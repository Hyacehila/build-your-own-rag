"""B/C-only coverage selection and deterministic five-stage source restoration."""
import re
from collections import defaultdict

from .common import RagError
from .facts import ASSETS, BODY_KINDS, HEADINGS
from .parsing import page_source
from .structural_expansion import Structure, read_nodes

POLICY = {'version':'BC-source-priority-v3','rrf_k':20,'sparse_k':50,'dense_k':50,
          'anchor_groups':20,'text_tokens':25000,'page_images':8,'body_radius_each_side':8,
          'short_section_body_tokens':4000,'long_section_intro_nodes':8,'title_sections':0,'bound_page_tokens':4000,
          'parent_boundary':'nearest_heading_parent','source_pages':'assets_continuations_or_structural_gaps',
          'default_adjacent_pages':False,'recursive_expansion':False}
REFERENCE = re.compile(r'\b(table|figure|note|section|appendix)\s+([0-9]+(?:\.[0-9]+)*[a-z]?|[ivxlcdm]+|[a-z])\b', re.I)


def subtract_interval(start,end,covered):
    """Exact half-open interval subtraction; no similarity threshold."""
    pending=[(start,end)]
    for lo,hi in sorted(covered):
        pending=[piece for a,b in pending for piece in
                 ([(a,b)] if hi<=a or lo>=b else [(a,min(b,lo)),(max(a,hi),b)]) if piece[0]<piece[1]]
    return pending


def select_spans(candidates,limit):
    """Input pairs (hit, exact coverage spans); independently testable on fragments."""
    used = defaultdict(list)
    selected = []
    log = []
    for position,(hit,spans) in enumerate(candidates,1):
        new=[]
        for s in spans:
            key=(s['doc_version'],s['parse_id'],s['source_id'],s['representation'])
            for a, b in subtract_interval(s["start"], s["end"], used[key]):
                new.append({**s, "start": a, "end": b})
        keep=bool(new) or not spans
        log.append({'candidate_id':hit['id'],'entry_type':hit.get('summary_type',hit.get('kind')),
                    'is_summary':bool(hit.get('enrichment_id')),'rrf_score':hit.get('score'),
                    'fused_rank':position,'base_coverage':spans,'new_coverage':new,
                    'overlap_coverage':[dict(s,overlap=[list(p) for p in subtract_interval(s['start'],s['end'],
                         [(x['start'],x['end']) for x in new if all(x[k]==s[k] for k in ['doc_version','parse_id','source_id','representation'])])]) for s in spans],
                    'kept':keep,'reason':'new_original_coverage' if new else 'unknown_coverage_retained' if keep else 'fully_covered',
                    'selected_position':len(selected)+1 if keep else None})
        if keep:
            selected.append({**hit,'matched_chunk_ids':[hit['id']]})
            for s in new:
                used[(s["doc_version"], s["parse_id"], s["source_id"], s["representation"])].append(
                    (s["start"], s["end"])
                )
            if len(selected) == limit:
                break
    return selected,log


def base_nodes(hit,catalog,index):
    # Resolve one entrance independently: old matched-entrance groups must not
    # silently accumulate scores or conceal partial overlap before selection.
    item={**hit,'matched_chunk_ids':[hit['id']]}
    # C retains actual B entries; never disguise them as enriched-index hits.
    owner = catalog.index if hit.get('index_id') == getattr(catalog,'index',{}).get('id') and hasattr(catalog,'index') else index
    members=catalog.resolve(item,owner)
    nodes=list(dict.fromkeys(n for cid in members for n in catalog.chunks[cid]['node_ids']))
    if not nodes:
        nodes = list(hit.get("node_ids", []))
    for nid in nodes:
        if nid not in catalog.nodes:
            raise RagError("Coverage outside original B facts")
    related=[]
    for nid in nodes:
        n = catalog.nodes[nid]
        related.append(nid)
        for ref in n.get('related_refs',[]):
            r=catalog.by_ref.get((n['parse_id'],ref))
            if r and r["kind"] in {"caption", "footnote"}:
                related.append(r["id"])
    return members,list(dict.fromkeys(related))


def base_coverage(hit,catalog,index):
    _, ids = base_nodes(hit, catalog, index)
    spans = []
    for nid in ids:
        n=catalog.nodes[nid]
        versions=sorted({s['doc_version'] for s in n.get('sources',[])})
        for version in versions:
            if n.get('text'):
                spans.append({'doc_version':version,'parse_id':n['parse_id'],'source_id':nid,
                              'representation':'original_node_text','start':0,'end':len(n['text'])})
            if n['kind'] in ASSETS:
                spans.append({'doc_version':version,'parse_id':n['parse_id'],'source_id':nid,
                              'representation':'original_asset','start':0,'end':1})
    return spans


def select_coverage(hits,limit,catalog,index):
    return select_spans(((h,base_coverage(h,catalog,index)) for h in hits),limit)


def select_supplement(hits,limit,catalog,index,primary):
    """Protect B's selected originals; only summaries may add new base coverage."""
    if limit == 0:
        return [], []
    summaries=[h for h in hits if h.get('enrichment_id')]
    candidates=[(h,base_coverage(h,catalog,catalog.index)) for h in primary]
    candidates += [(h,base_coverage(h,catalog,index)) for h in summaries]
    selected,log=select_spans(candidates,len(primary)+limit)
    assert [h['id'] for h in selected[:len(primary)]]==[h['id'] for h in primary]
    ranks={h['id']:i for i,h in enumerate(hits,1)}
    for entry in log[len(primary):]:
        entry['fused_rank']=ranks[entry['candidate_id']]
        if entry["selected_position"] is not None:
            entry["selected_position"] -= len(primary)
    return selected[len(primary):],log[len(primary):]


class Navigation:
    def __init__(self,structure):
        self.s = structure
        self.nodes = structure.nodes
        self.ordered=sorted(self.nodes,key=self.order)
        self._section = {}
        self._parent = {}
        self._direct = {}
        self._sequence = {}
        self.children=defaultdict(list)
        for nid in self.ordered:
            if self.nodes[nid]["kind"] in HEADINGS:
                self.children[self.parent(nid)].append(nid)

    def order(self,nid):
        n = self.nodes[nid]
        return (n["parse_id"], n.get("reading_order") if n.get("reading_order") is not None else -1, nid)

    def section(self,nid):
        if nid in self._section:
            return self._section[nid]
        n=self.nodes[nid]
        if n["kind"] in HEADINGS:
            out = nid
        elif n.get("section_id") in self.nodes and self.nodes[n["section_id"]]["kind"] in HEADINGS:
            out = n["section_id"]
        else:
            out = self.parent(nid)
        self._section[nid] = out
        return out

    def parent(self,nid):
        if nid in self._parent:
            return self._parent[nid]
        parse = self.nodes[nid]["parse_id"]
        p = self.nodes[nid].get("parent_id")
        seen = set()
        while p in self.nodes:
            if p in seen:
                raise RagError("Cycle in source heading ancestry")
            seen.add(p)
            n = self.nodes[p]
            if n["parse_id"] != parse:
                raise RagError("Source ancestry crosses document identity")
            if n["kind"] in HEADINGS:
                break
            p=n.get('parent_id')
        else:
            p = None
        self._parent[nid] = p
        return p

    def within(self,nid,parent):
        p = nid
        seen = set()
        while p in self.nodes:
            if p == parent:
                return True
            if p in seen:
                raise RagError("Cycle in original tree")
            seen.add(p)
            p = self.nodes[p].get("parent_id")
        return False

    def direct(self,heading):
        # Continuous specific section: stop at its first child heading, never
        # skip a child and return to later direct content.
        if heading not in self._direct:
            first=min([self.order(c) for c in self.children.get(heading,[])] or [('\U0010ffff',float('inf'),'')])
            self._direct[heading]=[n for n in self.ordered if n!=heading and self.section(n)==heading and self.order(n)<first
                                  and self.nodes[n]['kind'] in BODY_KINDS|ASSETS|{'caption','footnote'}]
        return self._direct[heading]

    def parent_of_anchor(self,nid):
        section=self.section(nid)
        return self.parent(section) if section else None

    def adjacent(self,nid):
        parent=self.parent_of_anchor(nid)
        if not parent:
            return [], None
        if parent not in self._sequence:
            self._sequence[parent] = [n for n in self.ordered if self.within(n, parent)]
        sequence=self._sequence[parent]
        if nid not in sequence:
            return [], parent
        pos = sequence.index(nid)
        body = [i for i, n in enumerate(sequence) if self.nodes[n]["kind"] in BODY_KINDS]
        before = [i for i in body if i < pos][-8:]
        after = [i for i in body if i > pos][:8]
        lo = min([pos, *before])
        hi = max([pos, *after])
        return [n for n in sequence[lo:hi+1] if self.nodes[n]['kind'] in BODY_KINDS|ASSETS|{'caption','footnote'}],parent

    def bounded_section(self,heading):
        ids = self.direct(heading)
        body = [n for n in ids if self.nodes[n]["kind"] in BODY_KINDS]
        if sum(self.s.tokenizer.count(self.nodes[n]["text"]) for n in body) <= 4000:
            return ids
        chosen=body[:8]
        # Include interleaved assets in the selected introductory window, but do
        # not include every table in a long section merely due to its heading.
        if chosen:
            last = self.order(chosen[-1])
            chosen = [n for n in ids if self.order(n) <= last]
        return list(dict.fromkeys(chosen))


def read_bound_page(reader,source):
    """Exact original PDF lines; geometry exclusion only for fully read single-page originals."""
    import pymupdf

    version, page = source["doc_version"], source["page_index"]
    doc = reader._document(version)
    covered = []
    uncertain = []
    for e in reader.evidence.values():
        if e["type"] != "text":
            continue
        loc=[s for s in e['sources'] if s['doc_version']==version and s['page_index']==page and s.get('bbox')]
        # Partial node/line text cannot justify excluding its entire rectangle.
        complete=not e.get('truncated') and e.get('offset',0)==0
        pages={(s['doc_version'],s['page_index']) for s in e['sources']}
        if complete and len(pages) == 1:
            covered.extend(pymupdf.Rect(s["bbox"]) for s in loc)
        elif loc:
            uncertain.append(e["id"])
    results = []
    skipped = []
    unread = []
    before = set(reader.evidence)
    text_before = reader.text_used
    with pymupdf.open(doc['pdf_path']) as pdf:
        for bi,block in enumerate(pdf[page].get_text('dict',sort=True)['blocks']):
            for li,line in enumerate(block.get('lines',[])):
                key = f"raw-page-line:{version}:{page}:{bi}:{li}"
                rect = pymupdf.Rect(line["bbox"])
                if any(r.contains(rect) for r in covered):
                    skipped.append({"source_key": key, "reason": "fully_read_original_geometry"})
                    continue
                text=''.join(s['text'] for s in line['spans'])
                if not text.strip():
                    continue
                loc={**page_source(doc,page),'bbox':list(rect),'coordinate_system':'PDF points, top-left'}
                if reader.remaining<=0:
                    unread.append(
                        {"source_key": key, "offset": 0, "end_offset": len(text), "reason": "text_budget"}
                    )
                    continue
                reader._read_limit=reader.text_used+reader.remaining
                result = reader._text(key, text, [loc], 0, "page_line")
                results.append(result)
                if result.get("next_offset") is not None:
                    unread.append(
                        {
                            "source_key": key,
                            "offset": result["next_offset"],
                            "end_offset": len(text),
                            "reason": "text_budget",
                        }
                    )
    image_result=reader._image(page_source(doc,page))
    return {'source':source,'reason':'selected_entrance_bound_source_page','results':results,
            'new_evidence':sorted(set(reader.evidence)-before),'new_tokens':reader.text_used-text_before,
            'skipped_exact_geometry':skipped,'uncertain_overlap_evidence_ids':uncertain,
            'unread_ranges':unread,'image_result':image_result,'actually_read':len(reader.evidence)>len(before)}


def assemble_restored(reader,hits,retrieval_index,catalog):
    s = Structure(catalog, reader.tokenizer)
    nav = Navigation(s)
    reader.structure_nodes = s.nodes
    reader.config=reader.config.model_copy(deep=True)
    reader.config.query.text_tokens=min(reader.config.query.text_tokens,reader.config.query.bc_text_tokens,POLICY['text_tokens'])
    reader.image_limit=min(reader.image_limit,8)
    anchors = []
    primary_anchors = []
    members = []
    groups = []
    events = []
    gaps = []
    phases = []
    unread = []
    # Primary B anchors always precede the capped summary supplements.
    for hit in hits:
        bound, ids = base_nodes(hit, catalog, retrieval_index)
        members.extend(bound)
        anchors.extend(ids)
        if not hit.get("enrichment_id"):
            primary_anchors.extend(ids)
        groups.append({'entrance_id':hit['id'],'members':bound,'node_ids':ids,'summary':bool(hit.get('enrichment_id'))})
    anchors = list(dict.fromkeys(anchors))
    anchor_related, missing = s.related(anchors)
    gaps.extend(missing)
    selected_all=[]
    def read_phase(name,ids,relations):
        ids = list(dict.fromkeys(ids))
        before = set(reader.evidence)
        text_before = reader.text_used
        result = read_nodes(reader, s, ids, include_images=True)
        selected_all.extend(ids)
        unread.extend(result["unread_ranges"])
        events.append({'phase':name,'relations':relations,'selected_node_ids':ids,'results':result['results'],
                       'new_evidence':sorted(set(reader.evidence)-before),'new_tokens':reader.text_used-text_before,
                       'unread_ranges':result['unread_ranges'],'stop_reason':'text_budget_or_row_boundary' if result['unread_ranges'] else None})
        phases.append({'phase':name,'text_tokens':reader.text_used,'evidence_count':len(reader.evidence),'images':len(reader.images)})
    protected,_=s.related(list(dict.fromkeys(primary_anchors)))
    first=sorted(protected,key=lambda n:s.nodes[n]['kind'] not in {'caption','footnote'})
    first+=sorted([n for n in anchor_related if n not in set(protected)],key=lambda n:s.nodes[n]['kind'] not in {'caption','footnote'})
    read_phase('anchors_and_related',first,[{'anchor':n,'relation':'base_original'} for n in anchors])

    parents = []
    parent_triggers = {}
    sections = set()
    for nid in anchors:
        sec=nav.section(nid)
        if sec:
            sections.add(sec)
        parent=nav.parent_of_anchor(nid)
        if parent:
            if parent not in parent_triggers:
                parents.append(parent)
                parent_triggers[parent] = nid
        else:
            gaps.append({"node_id": nid, "reason": "no_parent_heading"})
    overview = []
    overview_rel = []
    explicit_nodes = []
    explicit_sections = []
    explicit_rel = []
    for parent in parents:
        direct = nav.direct(parent)
        overview.extend(direct)
        overview_rel.append({'anchor':parent_triggers[parent],'relation':'parent_direct_overview','parent':parent,'nodes':direct})
    # References are extracted only from original anchors, never newly expanded text.
    for nid in anchors:
        parent=nav.parent_of_anchor(nid)
        if not parent:
            continue
        candidates=nav.children.get(parent,[])
        for ref in s.nodes[nid].get('related_refs',[]):
            target=s.by_ref.get((s.nodes[nid]['parse_id'],ref))
            if not target:
                continue
            sec=nav.section(target['id'])
            if sec in candidates and sec not in sections:
                explicit_sections.append(sec)
                explicit_nodes.append(target["id"])
                explicit_rel.append({'anchor':nid,'relation':'explicit_structure','target':target['id'],'section':sec})
        refkeys={(a.casefold(),b.casefold()) for a,b in REFERENCE.findall(s.nodes[nid].get('text',''))}
        for key in sorted(refkeys):
            matches=[]
            for target in nav.ordered:
                sec=nav.section(target)
                if sec not in candidates:
                    continue
                n=s.nodes[target]
                if n["kind"] not in HEADINGS | ASSETS | {"caption"}:
                    continue
                title=n.get('text','').split('\n',1)[0]
                if key in {(a.casefold(), b.casefold()) for a, b in REFERENCE.findall(title)}:
                    matches.append(target)
            if len(matches)==1:
                target = matches[0]
                explicit_nodes.append(target)
                if s.nodes[target]["kind"] in HEADINGS:
                    explicit_sections.append(target)
                explicit_rel.append({'anchor':nid,'relation':'unique_original_reference','reference':list(key),'target':target})
            else:
                gaps.append(
                    {
                        "node_id": nid,
                        "reference": list(key),
                        "reason": "reference_ambiguous" if matches else "reference_unresolved",
                        "candidates": matches,
                    }
                )
    associated = list(dict.fromkeys(explicit_sections))
    explicit_scope = list(explicit_nodes)
    for sec in associated:
        chosen = nav.bounded_section(sec)
        explicit_scope.extend(chosen)
        explicit_rel.append({'relation':'explicit_associated_section','section':sec,'nodes':chosen,
                             'trigger_anchors':[n for n in anchors if nav.parent_of_anchor(n)==nav.parent(sec)]})
    required, missing = s.related(list(dict.fromkeys(explicit_scope)))
    gaps.extend(missing)
    read_phase('explicit_relations',required,explicit_rel)
    local, missing = s.related(s.select(anchors, short=True))
    gaps.extend(missing)
    read_phase('local_structure',local,[{'anchor':n,'relation':'same_section_eight_or_short'} for n in anchors])
    stage3, missing = s.related(list(dict.fromkeys(overview)))
    gaps.extend(missing)
    read_phase('parent_overview',stage3,overview_rel)

    # Prioritize original structural relationships inside25k; lexical-title
    # matches are intentionally disabled by the user's budget revision.
    stage4 = []
    rels = []
    for nid in anchors:
        window, parent = nav.adjacent(nid)
        stage4.extend(window)
        rels.append({'anchor':nid,'relation':'cross_local_heading_eight','parent':parent,'nodes':window})
    stage4, missing = s.related(stage4)
    gaps.extend(missing)
    read_phase('associated_sections_and_cross_heading',stage4,rels)

    page_sources = {}
    skipped_page_anchors = []
    # Normal single-page prose is covered through original nodes/windows, not
    # automatically expanded into whole pages. Assets, explicit continuations
    # and unresolved original structural relationships retain bound-page access.
    page_seeds=list(dict.fromkeys(anchors+explicit_nodes))
    gap_nodes={g.get('node_id') for g in gaps if g.get('reason')=='unresolved_explicit_relation'}
    for nid in page_seeds:
        node = s.nodes[nid]
        locs = node.get("sources", [])
        pages_for_node={(x['doc_version'],x['page_index']) for x in locs}
        reason=('original_asset' if node['kind'] in ASSETS else
                'bound_multi_page_continuation' if len(pages_for_node)>1 else
                'original_structural_gap' if node.get('structural_coverage_missing') or nid in gap_nodes else None)
        if not reason:
            skipped_page_anchors.append({"node_id": nid, "reason": "ordinary_prose_no_automatic_page"})
            continue
        for source in locs:
            key=(source['doc_version'],source['page_index'])
            if key not in page_sources:
                page_sources[key] = {"source": source, "triggers": []}
            page_sources[key]['triggers'].append({'node_id':nid,'reason':reason})
    pages = []
    before = reader.text_used
    overall_limit = reader.config.query.text_tokens
    reader.config.query.text_tokens=min(overall_limit,before+POLICY['bound_page_tokens'])
    try:
        for entry in page_sources.values():
            source=entry['source']
            if reader.remaining<=0:
                result={'source':source,'reason':'eligible_bound_page_budget_exhausted','new_evidence':[],
                        'new_tokens':0,'actually_read':False,'results':[],
                        'uncertain_overlap_evidence_ids':[],
                        'unread_ranges':[{'doc_version':source['doc_version'],'page_index':source['page_index'],
                                          'reason':'bound_page_4k_or_total_25k_budget'}]}
            else:
                result = read_bound_page(reader, source)
            result["triggers"] = entry["triggers"]
            pages.append(result)
            unread.extend(result["unread_ranges"])
    finally:
        reader.config.query.text_tokens = overall_limit
    events.append({'phase':'bound_original_pages','candidate_pages':[{'doc_version':v,'page_index':p} for v,p in page_sources],
                   'skipped_ordinary_prose':skipped_page_anchors,'stage_text_limit':POLICY['bound_page_tokens'],
                   'pages':pages,'new_tokens':reader.text_used-before})
    phases.append({'phase':'bound_original_pages','text_tokens':reader.text_used,'evidence_count':len(reader.evidence),'images':len(reader.images)})
    for group in groups:
        group['unread_node_ranges']=[x for x in unread if x.get('node_id') in group['node_ids']]
    return {'contract':POLICY['version'],'policy':{**POLICY,'text_tokens':reader.config.query.text_tokens},'anchor_ids':list(dict.fromkeys(members)),
            'anchor_node_ids':anchors,'protected_original_node_ids':list(dict.fromkeys(primary_anchors)),
            'resolved_groups':groups,'selected_node_ids':list(dict.fromkeys(selected_all)),
            'phases':phases,'events':events,'page_fallbacks':pages,'structural_gaps':gaps,
            'unread_ranges':unread,'title_matches':[],'removed_expansions':['lexical_title_only_sections','ordinary_single_page_prose_whole_pages'],
            'truncated_evidence_ids':[e['id'] for e in reader.evidence.values() if e.get('truncated')],
            'original_index_id':reader.index['id'],'retrieval_index_id':retrieval_index['id']}
