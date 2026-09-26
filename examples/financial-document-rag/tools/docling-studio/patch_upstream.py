"""Compatibility patch for pinned Studio; preserve explicit Docling subtrees."""

import hashlib
import sys
from pathlib import Path


def patch(root):
    path = root / "document-parser/services/chunk_service.py"
    text = path.read_text(encoding="utf-8")
    old = 'if label_type in {"list", "group", "form_area", "key_value_area"}:'
    new = 'if label_type in {"list", "group", "section", "title", "section_header", "form_area", "key_value_area"}:'
    heading_old = "node = _make_node(ref, label_type, item, inline_meta, tree_reader)\n                stack[-1][1].append(node)"
    heading_new = "node = _build_item_subtree(ref, item, by_ref, skip_refs, inline_meta, tree_reader)\n                stack[-1][1].append(node)"
    if new in text and heading_new in text:
        return
    if text.count(old) != 1 or text.count(heading_old) != 1:
        raise ValueError("Studio tree implementation changed; review the patch before applying.")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(text.replace(old, new).replace(heading_old, heading_new), encoding="utf-8")
    print("Applied explicit-tree compatibility patch; upstream SHA256:", original_hash)


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
