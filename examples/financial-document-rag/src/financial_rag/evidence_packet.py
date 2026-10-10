"""One deterministic original-evidence assembler shared by B, C and pre-D."""
from .common import RagError, digest
from .summary_windows import ordered_sections, original_chunks


class OriginalCatalog:
    def __init__(self, store, index):
        self.store, self.index = store, index
        self.chunks = {c['id']: c for c in original_chunks(store, index['id'])}
        self.nodes = {n['id']: n for pid in index['parse_ids'] for n in store.nodes(pid)}
        self.by_ref = {(n['parse_id'], n['ref']): n for n in self.nodes.values()}
        self.neighbors = {}
        for chunks in ordered_sections(list(self.chunks.values()), self.nodes).values():
            for i, c in enumerate(chunks):
                chosen = [x for j, x in enumerate(chunks) if abs(j-i) == 1]
                def contiguous(other):
                    positions = [self.nodes[n]['reading_order'] for v in [c, other] for n in v['node_ids']]
                    parse = self.nodes[c['node_ids'][0]]['parse_id']
                    if self.nodes[c['node_ids'][0]].get('section_id') is not None: return True
                    return not any(n['parse_id']==parse and n['kind']=='section_header'
                                   and min(positions)<n['reading_order']<max(positions) for n in self.nodes.values()
                                   if n.get('reading_order') is not None)
                self.neighbors[c['id']] = [x['id'] for x in chosen if contiguous(x)]

    def resolve(self, hit, retrieval_index):
        ids = []
        for cid in hit.get('matched_chunk_ids', [hit['id']]):
            c = hit if cid == hit['id'] else self.store.get('chunks', cid)
            if c['index_id'] != retrieval_index['id']:
                raise RagError('Retrieval hit belongs to another index')
            bound = c.get('source_chunk_ids')
            if bound is None:
                if c.get('enrichment_id') or retrieval_index['id'] != self.index['id']:
                    raise RagError('Summary/original entry lacks an explicit original binding')
                bound = [cid]
            if any(i not in self.chunks for i in bound):
                raise RagError('Source binding is outside the original B index')
            ids.extend(bound)
        return list(dict.fromkeys(ids))

    def assemble(self, reader, hits, retrieval_index, *, expand=True, expand_neighbors=None):
        anchors, original_assets = [], []
        for hit in hits:
            ids = self.resolve(hit, retrieval_index)
            anchors.extend(ids)
            if not ids:
                original_assets.extend(hit.get('node_ids', []))
        anchors = list(dict.fromkeys(anchors))
        truncated = []

        def read_chunk(cid):
            chunk = self.chunks[cid]
            if reader.remaining <= 0:
                truncated.append(cid)
                return
            reader.read(chunk_id=cid, include_images=expand, max_tokens=reader.config.query.text_tokens)
            for nid in chunk['node_ids']:
                node = self.nodes[nid]
                for ref in node.get('related_refs', []):
                    related = self.by_ref.get((node['parse_id'], ref))
                    if related and related['kind'] in {'caption', 'footnote'} and reader.remaining > 0:
                        # These are original related nodes, never generated summaries.
                        reader.read(node_id=related['id'], include_images=False,
                                    max_tokens=reader.config.query.text_tokens)

        for cid in anchors:
            read_chunk(cid)
        for nid in dict.fromkeys(original_assets):
            if nid not in self.nodes:
                raise RagError('Asset outside original fact release')
            reader.read(node_id=nid, include_images=expand, max_tokens=reader.config.query.text_tokens)
        neighbors = []
        if expand if expand_neighbors is None else expand_neighbors:
            neighbors = list(dict.fromkeys(n for cid in anchors for n in self.neighbors.get(cid, []) if n not in anchors))
            for cid in neighbors:
                read_chunk(cid)
        return {'contract': 'original-anchors-then-one-neighbor-v1', 'anchor_ids': anchors,
                'neighbor_ids': neighbors, 'unread_anchor_or_neighbor_ids': truncated,
                'truncated_evidence_ids': [e['id'] for e in reader.evidence.values() if e.get('truncated')],
                'original_index_id': self.index['id'], 'retrieval_index_id': retrieval_index['id']}


def snapshot(reader, question, rewritten_query, ranking, assembly):
    packet = {'question': question, 'rewritten_query': rewritten_query,
              'original_index_id': reader.index['id'], 'initial_page_ranking': ranking,
              'assembly': assembly, 'evidence': reader.evidence, 'images': reader.images,
              'seen': reader.seen, 'text_used': reader.text_used}
    # The serialized packet is immutable even while the D reader keeps growing.
    import json
    packet = json.loads(json.dumps(packet))
    return {'sha256': digest(packet), 'packet': packet}


def restore(reader, saved):
    import copy
    p = saved['packet']
    if p['original_index_id'] != reader.index['id']:
        raise RagError('Initial evidence snapshot identity mismatch')
    reader.evidence = copy.deepcopy(p['evidence'])
    reader.images = copy.deepcopy(p['images'])
    reader.seen = copy.deepcopy(p['seen'])
    reader.text_used = p['text_used']
