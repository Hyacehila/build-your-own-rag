"""Hash-bound, source-preserving document review decisions."""

from docling_core.types.doc import ContentLayer, DocItemLabel, SectionHeaderItem, TextItem, TitleItem

from .common import RagError
from .outline import apply_item_reparents, apply_outline_patches, fingerprint_content, rebuild_scoped
from .page_exclusions import exclude_pages


def apply_profile(document, profile):
    excluded = {x["page_number"] for x in profile["excluded_pages"]}
    if excluded:
        doc, exclusions = exclude_pages(document, excluded)
        mapping = exclusions["old_to_new_refs"]
    else:
        doc = document.model_copy(deep=True)
        mapping = {
            x.self_ref: x.self_ref
            for name in ("texts", "groups", "tables", "pictures")
            for x in getattr(doc, name)
        }
        exclusions = {"old_to_new_refs": mapping, "removed_refs": []}
    before = fingerprint_content(doc)
    boundaries = sorted(set(profile["boundaries"]))
    if (
        not boundaries
        or boundaries[0] != 1
        or any(type(p) is not int or p not in doc.pages for p in boundaries)
    ):
        raise RagError("Profile boundaries must start at physical page 1 and stay within the PDF.")
    raw_refs = {x.self_ref for x in document.texts}
    decisions = (
        set(profile["heading_levels"]) | set(profile.get("demote", [])) | set(profile.get("furniture", []))
    )
    if not decisions <= raw_refs:
        raise RagError("Profile refers to missing raw text nodes.")
    remaining = {
        h.self_ref for h in document.texts if isinstance(h, SectionHeaderItem) and h.self_ref in mapping
    }
    if not remaining <= decisions:
        raise RagError("Profile must explicitly classify every retained section heading.")
    former = {h.self_ref for h in doc.texts if isinstance(h, (SectionHeaderItem, TitleItem))}
    by_ref = {x.self_ref: x for x in doc.texts}
    for raw_ref, level in profile["heading_levels"].items():
        if type(level) is not int or not 1 <= level <= 12:
            raise RagError("Heading level must be an integer from 1 through 12.")
        ref = mapping.get(raw_ref)
        if ref and isinstance(by_ref[ref], SectionHeaderItem):
            by_ref[ref].level = level
    demote = {
        mapping[r]
        for r in [
            *profile.get("demote", []),
            *profile.get("furniture", []),
            *profile.get("navigation_refs", []),
        ]
        if r in mapping
    }
    for i, item in enumerate(doc.texts):
        if item.self_ref in demote and isinstance(item, (SectionHeaderItem, TitleItem)):
            data = item.model_dump(mode="json")
            data.pop("level", None)
            data["label"] = DocItemLabel.TEXT
            doc.texts[i] = TextItem.model_validate(data)
    rebuild_scoped(doc, boundaries, former)
    by_ref = {x.self_ref: x for x in doc.texts}
    all_by_ref = {
        x.self_ref: x for name in ("texts", "tables", "pictures", "groups") for x in getattr(doc, name)
    }
    for raw_ref in [*profile.get("furniture", []), *profile.get("navigation_refs", [])]:
        if raw_ref not in mapping:
            if raw_ref in raw_refs:
                continue
            raise RagError("Navigation ref missing from retained facts.")
        item = all_by_ref[mapping[raw_ref]]
        parent = item.parent.resolve(doc)
        parent.children = [r for r in parent.children if r.cref != item.self_ref]
        if raw_ref in profile.get("furniture", []) and item.children:
            raise RagError("Furniture reclassification must not remove a retained subtree.")
        item.parent = doc.furniture.get_ref()
        if type(item) is TextItem:
            item.label = DocItemLabel.PAGE_HEADER
        item.content_layer = ContentLayer.FURNITURE
        doc.furniture.children.append(item.get_ref())
    for promotion in profile.get("promote", []):
        ref = mapping.get(promotion["ref"])
        anchor_ref = mapping.get(promotion["before_ref"])
        if not ref or not anchor_ref or ref == anchor_ref:
            raise RagError("Promotion needs retained source and insertion anchor.")
        item, anchor = all_by_ref[ref], all_by_ref[anchor_ref]
        level = promotion["level"]
        evidence = promotion.get("evidence_pages", [])
        if (
            not isinstance(item, TextItem)
            or item.children
            or not item.prov
            or not anchor.prov
            or item.prov[0].page_no != anchor.prov[0].page_no
            or item.prov[0].page_no not in evidence
            or type(level) is not int
            or not 1 <= level <= 12
        ):
            raise RagError("Promotion requires exact leaf text, valid level and same-page evidence/anchor.")
        old_parent = item.parent.resolve(doc)
        old_parent.children = [r for r in old_parent.children if r.cref != ref]
        data = item.model_dump(mode="json")
        data.update(label=DocItemLabel.SECTION_HEADER, level=level, content_layer=ContentLayer.BODY)
        promoted = SectionHeaderItem.model_validate(data)
        doc.texts[int(ref.rsplit("/", 1)[1])] = promoted
        parent = anchor.parent.resolve(doc)
        position = next(i for i, r in enumerate(parent.children) if r.cref == anchor_ref)
        parent.children.insert(position, promoted.get_ref())
        promoted.parent = parent.get_ref()
        all_by_ref[ref] = promoted
    if profile.get("promote"):
        rebuild_scoped(doc, boundaries)
    proposal = {"nest": []}
    for p in profile.get("nest", []):
        if p["child_ref"] not in mapping or p["parent_ref"] not in mapping:
            raise RagError("Outline patch refers to excluded content.")
        proposal["nest"].append(
            {**p, "child_ref": mapping[p["child_ref"]], "parent_ref": mapping[p["parent_ref"]]}
        )
    if proposal["nest"]:
        doc, patch_report = apply_outline_patches(doc, proposal, boundaries)
    else:
        patch_report = {"applied": []}
    moves = []
    for move in profile.get("reparent", []):
        if move["child_ref"] not in mapping or move["parent_ref"] not in mapping:
            raise RagError("Item reparent refers to excluded content.")
        moves.append(
            {**move, "child_ref": mapping[move["child_ref"]], "parent_ref": mapping[move["parent_ref"]]}
        )
    doc, reparent_report = apply_item_reparents(doc, moves, boundaries)
    if fingerprint_content(doc) != before:
        raise RagError("Profile changed source content or provenance.")
    return doc, {
        "exclusions": exclusions,
        "patches": patch_report,
        "content_unchanged": True,
        "furniture_reclassified": len(profile.get("furniture", [])),
        "demoted": len(profile.get("demote", [])),
        "promoted": len(profile.get("promote", [])),
        "reparented": len(reparent_report["applied"]),
        "reparent_patches": reparent_report,
        "boundaries": boundaries,
    }
