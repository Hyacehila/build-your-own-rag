"""Original-node navigation shared by deterministic B/C and explicit D tools."""
import copy
import re
from collections import defaultdict

from .common import RagError
from .facts import ASSETS, BODY_KINDS, HEADINGS
from .sources import Reader

POLICY = {'short_section_body_tokens': 4000}

class Structure:
    def __init__(self, catalog, tokenizer):
        self.catalog, self.nodes, self.tokenizer = catalog, catalog.nodes, tokenizer
        self.by_ref = catalog.by_ref
        self.segments, self.segment_of = [], {}
        ordered = sorted((n for n in self.nodes.values() if n.get('reading_order') is not None),
                         key=lambda n: (n['parse_id'], n['reading_order'], n['id']))
        previous = None
        for n in ordered:
            key = (n['parse_id'], n.get('section_id'))
            if key != previous or n['kind'] in HEADINGS:
                self.segments.append([])
            self.segments[-1].append(n['id'])
            self.segment_of[n['id']] = len(self.segments)-1
            previous = key
        self.lengths = {}

    def node(self, nid):
        if nid not in self.nodes:
            raise RagError("Node outside frozen original index")
        return self.nodes[nid]

    def targets(self, node_id=None, chunk_id=None):
        if (node_id is None) == (chunk_id is None):
            raise RagError("Supply exactly one original node_id or chunk_id")
        ids = [node_id] if node_id is not None else self.catalog.chunks.get(chunk_id, {}).get('node_ids')
        if not ids:
            raise RagError("Chunk outside frozen original index or without nodes")
        for nid in ids:
            self.node(nid)
        return list(dict.fromkeys(ids))

    def related(self, ids):
        result, gaps = [], []
        for nid in ids:
            n = self.node(nid)
            result.append(nid)
            for ref in n.get('related_refs', []):
                other = self.by_ref.get((n['parse_id'], ref))
                if other:
                    # Only explicit original relationships, never inferred relevance.
                    result.append(other['id'])
                else:
                    gaps.append(
                        {
                            "reason": "unresolved_explicit_relation",
                            "node_id": nid,
                            "reference": ref,
                            "sources": n.get("sources", []),
                        }
                    )
        return list(dict.fromkeys(result)), gaps

    def select(self, anchors, *, whole=False, short=False, radius=8):
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            raise RagError('Radius must be a nonnegative integer')
        selected = []
        grouped = defaultdict(list)
        for nid in anchors:
            self.node(nid)
            if nid in self.segment_of:
                grouped[self.segment_of[nid]].append(nid)
            else:
                selected.append(nid)
        for sid, seeds in grouped.items():
            segment = self.segments[sid]
            body = [nid for nid in segment if self.nodes[nid]['kind'] in BODY_KINDS]
            if sid not in self.lengths:
                self.lengths[sid] = sum(self.tokenizer.count(self.nodes[nid]['text']) for nid in body)
            if whole or (short and self.lengths[sid] <= POLICY['short_section_body_tokens']):
                chosen = segment
            else:
                # Each anchor's window; never bridge distant anchors through a long section.
                chosen = []
                positions = {nid: i for i, nid in enumerate(segment)}
                for seed in seeds:
                    pos = positions[seed]
                    before = [nid for nid in body if positions[nid] < pos][-radius:] if radius else []
                    after = [nid for nid in body if positions[nid] > pos][:radius]
                    lo = min([pos, *[positions[n] for n in before]])
                    hi = max([pos, *[positions[n] for n in after]])
                    chosen.extend(segment[lo:hi+1])
            selected.extend(nid for nid in chosen if self.nodes[nid]['kind'] in BODY_KINDS | ASSETS | {'caption','footnote'})
        # Reading order within each selected range; overlapping windows only appear once.
        return list(dict.fromkeys(selected))


class StructuralReader(Reader):
    """Node evidence is canonical; table truncation preserves complete row boundaries."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alias_intervals = defaultdict(list)

    def _text(self, key, text, sources, offset, kind):
        aliases = self.alias_intervals.get(key, [])
        for start, end, eid in aliases:
            if start <= offset < end:
                return {'evidence_id':eid, 'cached':True, 'next_offset':end if end<len(text) else None}
        original_limit = self._read_limit
        end = min([a for a,_,_ in aliases if a > offset] or [len(text)])
        allowance = min(self.remaining, self._read_limit-self.text_used)
        selected = self.tokenizer.prefix(text[offset:end], allowance)
        if kind == 'table' and offset+len(selected) < len(text):
            boundary = selected.rfind('\n')
            selected = selected[:boundary+1] if boundary >= 0 else ''
        if not selected and offset < len(text):
            return {'status':'text_budget_exhausted', 'next_offset':offset,
                    'row_boundary_preserved':kind=='table'}
        self._read_limit = min(original_limit, self.text_used+self.tokenizer.count(selected))
        try:
            result = super()._text(key,text,sources,offset,kind)
            if result.get('evidence_id') and key in getattr(self, 'structure_nodes', {}):
                self.evidence[result['evidence_id']]['heading_path'] = self.structure_nodes[key].get('headings', [])
            return result
        finally:
            self._read_limit = original_limit

    def adopt(self, old, structure):
        for attr in ['evidence','images','seen','text_used','preview_used','question']:
            setattr(self, attr, copy.deepcopy(getattr(old,attr)))
        self.structure_nodes = structure.nodes
        # Preserve immutable first packet while avoiding redisplaying its original spans.
        for e in self.evidence.values():
            if e["type"] != "text" or not e.get("chunk_id"):
                continue
            chunk = structure.catalog.chunks[e['chunk_id']]
            snippet = e['text']
            header = '\n'.join(chunk.get('headings', []))
            if header and snippet.startswith(header):
                snippet = snippet[len(header) :].lstrip("\n")
            normalized = re.sub(r'\s+', ' ', snippet).strip()
            for nid in chunk['node_ids']:
                text=structure.nodes[nid]['text']
                if text and re.sub(r'\s+',' ',text).strip() in normalized:
                    self.alias_intervals[nid].append((0,len(text),e['id']))
                elif snippet and snippet in text:
                    start = text.index(snippet)
                    self.alias_intervals[nid].append((start, start + len(snippet), e["id"]))

    def content(self, only_ids=None):
        parts=super().content(only_ids)
        for part in parts:
            if part.get("type") != "text":
                continue
            match=re.match(r'\[(E\d+)\]',part['text'])
            path=self.evidence.get(match[1],{}).get('heading_path',[]) if match else []
            if path:
                head,sep,body=part['text'].partition('\n')
                part['text']=head+'\nOriginal heading path: '+' > '.join(path)+sep+body
        return parts


def read_nodes(reader, structure, ids, *, include_images=True, node_start=0, offset=0):
    if node_start < 0 or node_start > len(ids):
        raise RagError("Node cursor outside selection")
    results, unread = [], []
    reader.structure_nodes = structure.nodes
    for i,nid in enumerate(ids[node_start:],node_start):
        node = structure.node(nid)
        cursor = offset if i == node_start else 0
        while True:
            if reader.remaining<=0:
                unread.append({"node_id": nid, "offset": cursor, "node_start": i})
                break
            # The selected list already includes explicit relations. A continuation's
            # offset applies only to this node, never to its shorter caption/footnote.
            before=set(reader.evidence)
            reader._read_limit=float('inf') if reader.unbounded else reader.text_used+reader.remaining
            response={'results':reader._node(nid,cursor,include_images and node['kind'] in ASSETS)}
            reader.trace.append({'tool':'read','arguments':{'node_id':nid,'offset':cursor,
                'include_images':include_images and node['kind'] in ASSETS},'response':response,
                'new_evidence':sorted(set(reader.evidence)-before),'status':'ok'})
            results.extend(response.get('results', []))
            if response.get("error"):
                raise RagError(response["error"])
            part=next((x for x in response.get('results',[]) if x.get('node_id')==nid),{})
            nxt=part.get('next_offset')
            if nxt is None:
                break
            if nxt<=cursor:
                unread.append({"node_id": nid, "offset": cursor, "node_start": i})
                break
            cursor=nxt
    return {'results':results,'selected_node_ids':ids,'unread_ranges':unread,
            'next_cursor':unread[0] if unread else None,
            'remaining_text_tokens':None if reader.unbounded else reader.remaining,
            'remaining_page_images':max(0,reader.image_limit-len(reader.images))}
