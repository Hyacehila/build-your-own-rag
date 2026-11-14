"""Project uniquely aligned text fragments to their actual source pages."""


def localize_fragment(chunk, nodes):
    if chunk["kind"] != "text" or not any(len({s["page_index"] for s in n["sources"]}) > 1 for n in nodes):
        return chunk
    compact, locations = [], []
    for node in nodes:
        original = node.get("raw", {}).get("orig", node["text"])
        provs = node.get("raw", {}).get("prov", [])
        if not provs or any(not 0 <= p["charspan"][0] <= p["charspan"][1] <= len(original) for p in provs):
            return {**chunk, "source_alignment": "invalid_or_missing_original_spans"}
        for i, char in enumerate(original):
            if not char.isspace():
                compact.append(char)
                locations.append(
                    (
                        node["id"],
                        i,
                        {p["page_no"] - 1 for p in provs if p["charspan"][0] <= i < p["charspan"][1]},
                    )
                )
    prefix = "\n".join(chunk.get("headings") or [])
    body = (
        chunk["text"][len(prefix) :].lstrip("\n")
        if prefix and chunk["text"].startswith(prefix)
        else chunk["text"]
    )
    needle = "".join(body.split())
    haystack = "".join(compact)
    start = haystack.find(needle)
    if not needle or start < 0 or haystack.find(needle, start + 1) >= 0:
        return {**chunk, "source_alignment": "not_uniquely_aligned"}
    covered = locations[start : start + len(needle)]
    if any(not loc[2] for loc in covered):
        return {**chunk, "source_alignment": "unlocated_characters"}
    allowed = {(nid, page) for nid, _, pages in covered for page in pages}
    # The node's PDF box remains a node-level box. Do not pretend that character
    # alignment provides a more precise geometric rectangle.
    sources = [s for n in nodes for s in n["sources"] if (n["id"], s["page_index"]) in allowed]
    unique = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    spans = []
    for nid in dict.fromkeys(n for n, _, _ in covered):
        positions = [(i, ps) for n, i, ps in covered if n == nid]
        spans.append(
            {
                "node_id": nid,
                "orig_start": min(i for i, _ in positions),
                "orig_end": max(i for i, _ in positions) + 1,
                "page_indices": sorted(set().union(*(ps for _, ps in positions))),
            }
        )
    return {
        **chunk,
        "sources": unique,
        "node_sources": chunk["sources"],
        "fragment_spans": spans,
        "source_alignment": "unique_original_text",
        "bbox_precision": "node",
    }
