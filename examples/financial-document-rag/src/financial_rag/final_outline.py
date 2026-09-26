"""Final-tree review packets and fail-closed publication checks; no LLM calls."""

import argparse
from collections import Counter
from pathlib import Path

from .common import RagError, digest, read_json, sha256, write_json
from .storage import Store

HEADINGS = {"title", "section_header"}
REVIEW_CONTRACT = "final-heading-semantics-v1"


def heading_rows(nodes):
    by_id = {n["id"]: n for n in nodes}
    rows = []
    for node in nodes:
        if node["kind"] not in HEADINGS or node["raw"].get("content_layer", "body") != "body":
            continue
        parent = by_id.get(node.get("parent_id"))
        chain, seen, scope = [], {node["id"]}, None
        while parent:
            if parent["id"] in seen:
                raise RagError("Cycle in final outline ancestry.")
            seen.add(parent["id"])
            if parent["kind"] in HEADINGS:
                chain.append(parent)
            if parent["raw"].get("name", "").startswith("source-range:"):
                scope = parent["raw"]["name"]
            parent = by_id.get(parent.get("parent_id"))
        rows.append(
            {
                "ref": node["ref"],
                "text": node["text"],
                "kind": node["kind"],
                "pages": sorted({s["page_number"] for s in node["sources"]}),
                "level": node["raw"].get("level"),
                "parent_ref": node["parent_ref"],
                "parent_heading_ref": chain[0]["ref"] if chain else None,
                "parent_heading_text": chain[0]["text"] if chain else None,
                "ancestor_refs": [n["ref"] for n in reversed(chain)],
                "heading_depth": len(chain) + 1,
                "scope": scope,
            }
        )
    return rows


def outline(store, row):
    parsed = store.get("parses", row["parse_id"])
    if sha256(Path(parsed["snapshot"])) != parsed["snapshot_sha256"]:
        raise RagError("Final outline snapshot hash mismatch.")
    rows = heading_rows(store.nodes(parsed["id"]))
    return {
        "contract": REVIEW_CONTRACT,
        "doc_id": row["doc_id"],
        "parse_id": parsed["id"],
        "snapshot_sha256": parsed["snapshot_sha256"],
        "outline_sha256": digest(rows),
        "headings": rows,
    }


def packets(root, output, size=100):
    if not 1 <= size <= 250:
        raise RagError("Use 1 through 250 headings per review packet.")
    manifest = []
    with Store(root, read_only=True) as store:
        release = store.meta("fact_release")
        if not release:
            raise RagError("Build a candidate fact release first.")
        for row in release["documents"]:
            data = outline(store, row)
            dest = output / row["doc_id"]
            write_json(dest / "outline.json", data)
            entries = []
            for start in range(0, len(data["headings"]), size):
                subset = data["headings"][start : start + size]
                lines = [
                    f"{start + i + 1:04d} {n['ref']} p{','.join(map(str, n['pages']))} "
                    f"{'  ' * (n['heading_depth'] - 1)}{n['text']}\n"
                    f"     parent={n['parent_heading_ref']} {n['parent_heading_text'] or '(scope root)'}; "
                    f"scope={n['scope']}"
                    for i, n in enumerate(subset)
                ]
                name = f"packet-{start // size + 1:03d}.txt"
                (dest / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
                entries.append(
                    {"name": name, "sha256": sha256(dest / name), "refs": [n["ref"] for n in subset]}
                )
            packet_manifest = {
                **{k: v for k, v in data.items() if k != "headings"},
                "packets": entries,
                "heading_count": len(data["headings"]),
            }
            write_json(dest / "packets.json", packet_manifest)
            manifest.append(packet_manifest)
    return {"release_id": release["release_id"], "documents": manifest}


def validate_review(data, receipt, page_count):
    for key in ("contract", "doc_id", "parse_id", "snapshot_sha256", "outline_sha256"):
        if receipt.get(key) != data[key]:
            raise RagError(f"Final outline review has stale or missing {key}.")
    if not isinstance(receipt.get("reviewer"), str) or not receipt["reviewer"].strip():
        raise RagError("Final outline review needs an identified reviewer.")
    if receipt.get("final_tree_reviewed") is not True:
        raise RagError("Final generated tree has not been reviewed.")
    refs = [n["ref"] for n in data["headings"]]
    covered = receipt.get("reviewed_heading_refs", [])
    if Counter(covered) != Counter(refs):
        raise RagError("Final outline review coverage is incomplete, duplicated or foreign.")
    for field in ("contents_pages_viewed", "representative_pages_viewed", "suspicious_pages_viewed"):
        pages = receipt.get(field)
        if not isinstance(pages, list) or any(type(p) is not int or not 1 <= p <= page_count for p in pages):
            raise RagError(f"Invalid or missing {field} evidence.")
    if not receipt["contents_pages_viewed"] and not receipt.get("no_contents_reason"):
        raise RagError("Contents evidence or an explicit no-contents reason is required.")
    if not receipt["representative_pages_viewed"]:
        raise RagError("Representative source pages were not reviewed.")
    if receipt.get("confirmed_wrong_parent_remaining") != []:
        raise RagError("Confirmed wrong-parent findings remain unresolved or unreported.")
    if not isinstance(receipt.get("uncertain_exceptions"), list):
        raise RagError("Review must report uncertain exceptions explicitly.")
    expected = receipt.get("expected_relations")
    if not isinstance(expected, list) or not expected:
        raise RagError("Source-grounded expected parent relations are required.")
    by_ref = {n["ref"]: n for n in data["headings"]}
    for relation in expected:
        node = by_ref.get(relation.get("child_ref"))
        field = relation.get("relation", "parent_heading_ref")
        if field not in {"parent_ref", "parent_heading_ref"} or node is None:
            raise RagError("Expected relation addresses a missing heading or invalid relation.")
        evidence = relation.get("evidence_pages", [])
        if not evidence or any(type(p) is not int or not 1 <= p <= page_count for p in evidence):
            raise RagError("Expected relation lacks valid source evidence.")
        if "expected_parent_ref" not in relation or node[field] != relation["expected_parent_ref"]:
            raise RagError(f"Expected parent failed for {node['ref']}: {node[field]}")
    return {
        "doc_id": data["doc_id"],
        "headings_reviewed": len(refs),
        "expected_relations": len(expected),
        "uncertain_exceptions": len(receipt["uncertain_exceptions"]),
        "receipt_sha256": digest(receipt),
    }


def verify_stored_acceptance(store):
    release = store.meta("fact_release")
    accepted = store.meta("outline_acceptance")
    if not accepted or accepted.get("release_id") != release["release_id"]:
        raise RagError("Final outline semantic acceptance is required before publication.")
    results = []
    for row in release["documents"]:
        locator = f"outline-review/{row['doc_id']}/receipt.json"
        resource = store.db.execute(
            "SELECT sha256,data FROM resources WHERE locator=?", (locator,)
        ).fetchone()
        if not resource:
            raise RagError("Missing final outline review receipt.")
        import hashlib
        import json

        if hashlib.sha256(resource[1]).hexdigest() != resource[0]:
            raise RagError("Stored review resource checksum mismatch.")
        results.append(validate_review(outline(store, row), json.loads(resource[1]), row["source_pages"]))
    if results != accepted["documents"]:
        raise RagError("Final outline acceptance metadata differs from its receipts.")
    return accepted


def accept(root, reviews):
    from .cleaning import _resource

    with Store(root) as store:
        release = store.meta("fact_release")
        if not release:
            raise RagError("Build the candidate first.")
        pending, results = [], []
        for row in release["documents"]:
            data = outline(store, row)
            receipt = read_json(reviews / row["doc_id"] / "receipt.json")
            results.append(validate_review(data, receipt, row["source_pages"]))
            pending.append((row["doc_id"], data, receipt))
        # Nothing is published until every document's final receipt passes.
        for doc_id, data, receipt in pending:
            _resource(store, f"outline-review/{doc_id}/outline.json", data)
            _resource(store, f"outline-review/{doc_id}/receipt.json", receipt)
        accepted = {"contract": REVIEW_CONTRACT, "release_id": release["release_id"], "documents": results}
        _resource(store, "outline-acceptance.json", accepted)
        store.set_meta("outline_acceptance", accepted)
        release.update(
            publication_status="accepted", semantic_status="final_heading_tree_reviewed_with_exceptions"
        )
        _resource(store, "release.json", release)
        store.set_meta("fact_release", release)
        return verify_stored_acceptance(store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["packets", "accept", "verify"])
    parser.add_argument("--root", type=Path, default=Path(".local/facts-v2-release"))
    parser.add_argument("--reviews", type=Path, default=Path(".local/final-outline-v2"))
    parser.add_argument("--packet-size", type=int, default=100)
    args = parser.parse_args()
    if args.action == "packets":
        result = packets(args.root, args.reviews, args.packet_size)
    elif args.action == "accept":
        result = accept(args.root, args.reviews)
    else:
        with Store(args.root, read_only=True) as store:
            result = verify_stored_acceptance(store)
    from .common import canonical

    print(canonical(result))


if __name__ == "__main__":
    main()
