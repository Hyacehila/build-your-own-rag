"""B/C coverage selection and bounded original-context reading."""

from collections import defaultdict

from financial_rag.common import RagError
from financial_rag.document.facts import ASSETS
from financial_rag.query.structural_expansion import Structure, read_nodes

POLICY = {"version": "BC-local-context-v1", "context_radius": 8}

def subtract_interval(start, end, covered):
    """Exact half-open interval subtraction; no similarity threshold."""
    pending = [(start, end)]
    for lo, hi in sorted(covered):
        pending = [
            piece
            for a, b in pending
            for piece in ([(a, b)] if hi <= a or lo >= b else [(a, min(b, lo)), (max(a, hi), b)])
            if piece[0] < piece[1]
        ]
    return pending


def select_spans(candidates, limit):
    """Input pairs (hit, exact coverage spans); independently testable on fragments."""
    used = defaultdict(list)
    selected = []
    log = []
    for position, (hit, spans) in enumerate(candidates, 1):
        new = []
        for s in spans:
            key = (s["doc_version"], s["parse_id"], s["source_id"], s["representation"])
            for a, b in subtract_interval(s["start"], s["end"], used[key]):
                new.append({**s, "start": a, "end": b})
        keep = bool(new) or not spans
        log.append(
            {
                "candidate_id": hit["id"],
                "entry_type": hit.get("summary_type", hit.get("kind")),
                "is_summary": bool(hit.get("enrichment_id")),
                "rrf_score": hit.get("score"),
                "fused_rank": position,
                "base_coverage": spans,
                "new_coverage": new,
                "overlap_coverage": [
                    dict(
                        s,
                        overlap=[
                            list(p)
                            for p in subtract_interval(
                                s["start"],
                                s["end"],
                                [
                                    (x["start"], x["end"])
                                    for x in new
                                    if all(
                                        x[k] == s[k]
                                        for k in ["doc_version", "parse_id", "source_id", "representation"]
                                    )
                                ],
                            )
                        ],
                    )
                    for s in spans
                ],
                "kept": keep,
                "reason": "new_original_coverage"
                if new
                else "unknown_coverage_retained"
                if keep
                else "fully_covered",
                "selected_position": len(selected) + 1 if keep else None,
            }
        )
        if keep:
            selected.append({**hit, "matched_chunk_ids": [hit["id"]]})
            for s in new:
                used[(s["doc_version"], s["parse_id"], s["source_id"], s["representation"])].append(
                    (s["start"], s["end"])
                )
            if len(selected) == limit:
                break
    return selected, log


def base_nodes(hit, catalog, index):
    # Resolve one entrance independently: old matched-entrance groups must not
    # silently accumulate scores or conceal partial overlap before selection.
    item = {**hit, "matched_chunk_ids": [hit["id"]]}
    # C retains actual B entries; never disguise them as enriched-index hits.
    owner = (
        catalog.index
        if hit.get("index_id") == getattr(catalog, "index", {}).get("id") and hasattr(catalog, "index")
        else index
    )
    members = catalog.resolve(item, owner)
    nodes = list(dict.fromkeys(n for cid in members for n in catalog.chunks[cid]["node_ids"]))
    if not nodes:
        nodes = list(hit.get("node_ids", []))
    for nid in nodes:
        if nid not in catalog.nodes:
            raise RagError("Coverage outside original B facts")
    related = []
    for nid in nodes:
        n = catalog.nodes[nid]
        related.append(nid)
        for ref in n.get("related_refs", []):
            r = catalog.by_ref.get((n["parse_id"], ref))
            if r and r["kind"] in {"caption", "footnote"}:
                related.append(r["id"])
    return members, list(dict.fromkeys(related))


def base_coverage(hit, catalog, index):
    _, ids = base_nodes(hit, catalog, index)
    spans = []
    for nid in ids:
        n = catalog.nodes[nid]
        versions = sorted({s["doc_version"] for s in n.get("sources", [])})
        for version in versions:
            if n.get("text"):
                spans.append(
                    {
                        "doc_version": version,
                        "parse_id": n["parse_id"],
                        "source_id": nid,
                        "representation": "original_node_text",
                        "start": 0,
                        "end": len(n["text"]),
                    }
                )
            if n["kind"] in ASSETS:
                spans.append(
                    {
                        "doc_version": version,
                        "parse_id": n["parse_id"],
                        "source_id": nid,
                        "representation": "original_asset",
                        "start": 0,
                        "end": 1,
                    }
                )
    return spans


def select_coverage(hits, limit, catalog, index):
    return select_spans(((h, base_coverage(h, catalog, index)) for h in hits), limit)


def select_supplement(hits, limit, catalog, index, primary):
    """Protect B's selected originals; only summaries may add new base coverage."""
    if limit == 0:
        return [], []
    summaries = [h for h in hits if h.get("enrichment_id")]
    candidates = [(h, base_coverage(h, catalog, catalog.index)) for h in primary]
    candidates += [(h, base_coverage(h, catalog, index)) for h in summaries]
    selected, log = select_spans(candidates, len(primary) + limit)
    assert [h["id"] for h in selected[: len(primary)]] == [h["id"] for h in primary]
    ranks = {h["id"]: i for i, h in enumerate(hits, 1)}
    for entry in log[len(primary) :]:
        entry["fused_rank"] = ranks[entry["candidate_id"]]
        if entry["selected_position"] is not None:
            entry["selected_position"] -= len(primary)
    return selected[len(primary) :], log[len(primary) :]



def assemble_restored(reader, hits, retrieval_index, catalog):
    """Read original hits first, then a bounded window in the same section."""
    structure = Structure(catalog, reader.tokenizer)
    reader.structure_nodes = structure.nodes
    reader.config = reader.config.model_copy(deep=True)
    reader.config.query.text_tokens = min(reader.config.query.text_tokens, reader.config.query.bc_text_tokens)
    anchors, members, groups, primary = [], [], [], []
    for hit in hits:
        bound, ids = base_nodes(hit, catalog, retrieval_index)
        members.extend(bound)
        anchors.extend(ids)
        if not hit.get("enrichment_id"):
            primary.extend(ids)
        groups.append({"entrance_id": hit["id"], "members": bound, "node_ids": ids,
                       "summary": bool(hit.get("enrichment_id"))})
    anchors = list(dict.fromkeys(anchors))
    primary = list(dict.fromkeys(primary))
    first, missing_first = structure.related(primary)
    supplement, missing_supplement = structure.related([nid for nid in anchors if nid not in first])
    first.extend(nid for nid in supplement if nid not in first)
    nearby, missing_nearby = structure.related(structure.select(anchors, radius=POLICY["context_radius"]))
    nearby = [nid for nid in nearby if nid not in first]
    phases, events, unread = [], [], []
    for name, ids in (("originals", first), ("local_context", nearby)):
        before = set(reader.evidence)
        used = reader.text_used
        result = read_nodes(reader, structure, ids, include_images=True)
        unread.extend(result["unread_ranges"])
        events.append({"phase": name, "selected_node_ids": ids, "results": result["results"],
                       "new_evidence": sorted(set(reader.evidence) - before),
                       "new_tokens": reader.text_used - used, "unread_ranges": result["unread_ranges"]})
        phases.append({"phase": name, "text_tokens": reader.text_used,
                       "evidence_count": len(reader.evidence), "images": len(reader.images)})
    for group in groups:
        group["unread_node_ranges"] = [r for r in unread if r.get("node_id") in group["node_ids"]]
    return {
        "contract": POLICY["version"],
        "policy": {**POLICY, "text_tokens": reader.config.query.text_tokens,
                   "page_images": reader.image_limit},
        "anchor_ids": list(dict.fromkeys(members)),
        "anchor_node_ids": anchors,
        "protected_original_node_ids": primary,
        "resolved_groups": groups,
        "selected_node_ids": list(dict.fromkeys(first + nearby)),
        "phases": phases,
        "events": events,
        "structural_gaps": missing_first + missing_supplement + missing_nearby,
        "unread_ranges": unread,
        "truncated_evidence_ids": [e["id"] for e in reader.evidence.values() if e.get("truncated")],
        "original_index_id": reader.index["id"],
        "retrieval_index_id": retrieval_index["id"],
    }
