"""Small source-preserving Markdown fact bundle. Original code: MIT.

Adapted from the author's Lenny Compass fact builder and locator contracts;
see README.md for provenance and the deliberately reduced scope.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from io import BytesIO
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from docling.datamodel.backend_options import MarkdownBackendOptions
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import DocumentConverter, MarkdownFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.types.doc import (
    DocItemLabel, DoclingDocument, ImageRef, RefItem, SectionHeaderItem, Size, TitleItem,
)
from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class FactError(ValueError):
    """An input, persisted bundle, or locator cannot be trusted for this read."""


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    builder_version: Literal["0.1.0"] = "0.1.0"
    document_id: str = Field(min_length=1)
    source_name: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_versions: dict[str, str]
    structure_policy: Literal["nest-root-items-by-existing-heading-level"] = (
        "nest-root-items-by-existing-heading-level"
    )
    image_policy: Literal["references-only; markdown-alt-as-caption"] = (
        "references-only; markdown-alt-as-caption"
    )


class Locator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    self_ref: str = Field(pattern=r"^#/(texts|tables|pictures|groups)/[0-9]+$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def resolve_ref(document: DoclingDocument, reference: str):
    try:
        item = RefItem(cref=reference).resolve(document)
        if item.self_ref != reference:
            raise ValueError("reference and stored self_ref disagree")
        return item
    except (IndexError, KeyError, AttributeError, ValueError) as exc:
        raise FactError(f"MISSING_REF: {reference}") from exc


def walk_body(document: DoclingDocument):
    """Walk stored children in order; reject duplicate/cyclic or broken edges."""
    seen: set[str] = set()
    stack = [(document.body, 0)]
    while stack:
        item, depth = stack.pop()
        reference = item.self_ref
        if reference in seen:
            raise FactError(f"INVALID_TREE: repeated node {reference}")
        seen.add(reference)
        yield item, depth
        children = []
        for child_ref in item.children:
            child = resolve_ref(document, child_ref.cref)
            if child.parent is None or child.parent.cref != reference:
                raise FactError(f"INVALID_TREE: parent mismatch at {child.self_ref}")
            children.append((child, depth + 1))
        stack.extend(reversed(children))


def preserve_image_references(document: DoclingDocument, markdown: str) -> None:
    """Map literal Markdown image references and alt text, without fetching bytes.

    Alt-as-caption is an explicit example policy, not a universal equivalence.
    A 0x0 size and placeholder DPI carry no measured image geometry.
    """
    images = []

    def visit(tokens):
        for token in tokens:
            if token.type == "image":
                images.append((token.attrGet("src"), token.content))
            elif token.children:
                visit(token.children)

    visit(MarkdownIt().parse(markdown))
    if len(images) != len(document.pictures):
        raise FactError("IMAGE_MISMATCH: Markdown references and parsed pictures differ")
    for picture, (uri, alt) in zip(document.pictures, images):
        if not uri:
            raise FactError("IMAGE_MISMATCH: empty image URI")
        picture.image = ImageRef(
            mimetype=mimetypes.guess_type(urlsplit(uri).path)[0] or "application/octet-stream",
            dpi=72,
            size=Size(width=0, height=0),
            uri=uri,
        )
        existing = [resolve_ref(document, ref.cref).text for ref in picture.captions]
        expected = [alt] if alt else []
        if existing not in ([], expected):
            raise FactError("IMAGE_MISMATCH: unexpected caption produced by parser")
        if alt and not existing:
            # The pinned Markdown backend also emits the alt as the next text
            # sibling. Reuse that exact item instead of duplicating its text.
            siblings = resolve_ref(document, picture.parent.cref).children
            index = next(i for i, ref in enumerate(siblings) if ref.cref == picture.self_ref)
            following = resolve_ref(document, siblings[index + 1].cref) if index + 1 < len(siblings) else None
            if following is not None and getattr(following, "text", None) == alt:
                caption = following
                caption.label = DocItemLabel.CAPTION
            else:
                caption = document.add_text(label=DocItemLabel.CAPTION, text=alt, parent=picture)
            picture.captions.append(caption.get_ref())


def nest_sections(document: DoclingDocument) -> None:
    """Organize root siblings by explicit heading levels, without inventing text.

    Existing list/table/picture internals remain intact. This small adapter is
    for the pinned Markdown backend, not a general hierarchy recovery algorithm.
    """
    roots = [resolve_ref(document, ref.cref) for ref in document.body.children]
    document.body.children = []
    stack = [(-1, document.body)]
    for item in roots:
        level = 0 if isinstance(item, TitleItem) else (
            item.level if isinstance(item, SectionHeaderItem) else None
        )
        if level is not None:
            while len(stack) > 1 and stack[-1][0] >= level:
                stack.pop()
        parent = stack[-1][1]
        item.parent = parent.get_ref()
        parent.children.append(item.get_ref())
        if level is not None:
            stack.append((level, item))


def convert_markdown(source: bytes, name: str) -> DoclingDocument:
    if Path(name).suffix.lower() != ".md" or not source.strip():
        raise FactError("INVALID_SOURCE: expected a non-empty .md file")
    try:
        markdown = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FactError("INVALID_SOURCE: expected UTF-8") from exc
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: MarkdownFormatOption(
                pipeline_cls=SimplePipeline,
                backend_options=MarkdownBackendOptions(
                    enable_remote_fetch=False, enable_local_fetch=False, fetch_images=False,
                ),
            ),
        },
    )
    # Parse the same bytes we hash, rather than reopening a mutable source path.
    result = converter.convert(DocumentStream(name=name, stream=BytesIO(source)))
    if result.status is not ConversionStatus.SUCCESS:
        raise FactError(f"CONVERSION_FAILED: {result.status}")
    document = result.document
    list(walk_body(document))
    return document


def structure_text(document: DoclingDocument) -> str:
    rows = []
    for item, depth in walk_body(document):
        label = item.label.value
        description = getattr(item, "text", "") or getattr(item, "name", "") or ""
        rows.append(f"{'  ' * depth}{item.self_ref} [{label}] {description[:72]}".rstrip())
    return "\n".join(rows) + "\n"


def build_bundle(source_path: Path, output: Path, document_id: str) -> Manifest:
    """Write a new snapshot directory; existing snapshots are never overwritten."""
    if output.exists():
        raise FactError(f"OUTPUT_EXISTS: choose a new --output directory: {output}")
    source = source_path.read_bytes()
    document = convert_markdown(source, source_path.name)
    parsed_bytes = json_bytes(document.model_dump(mode="json", by_alias=True))
    preserve_image_references(document, source.decode("utf-8"))
    nest_sections(document)
    list(walk_body(document))
    serialized = json_bytes(document.model_dump(mode="json", by_alias=True))
    if DoclingDocument.model_validate_json(serialized) != document:
        raise FactError("ROUNDTRIP_MISMATCH: document changed during serialization")
    manifest = Manifest(
        document_id=document_id,
        source_name=source_path.name,
        source_sha256=sha256(source),
        document_sha256=sha256(serialized),
        parser_versions={name: version(name) for name in ("docling-slim", "docling-core")},
    )
    candidates = [item for item, _ in walk_body(document) if item.self_ref != "#/body"]
    if not candidates:
        raise FactError("EMPTY_DOCUMENT: no body items")
    selected = document.tables[0] if document.tables else candidates[0]
    locator = Locator(
        document_id=document_id,
        document_sha256=manifest.document_sha256,
        self_ref=selected.self_ref,
    )
    # Prepare all artifacts before publication. Manifest is the completion marker.
    artifacts = {
        "source.md": source,
        "parsed-document.json": parsed_bytes,
        "document.json": serialized,
        "locator.json": json_bytes(locator.model_dump()),
        "structure.txt": structure_text(document).encode("utf-8"),
        "preview.md": (document.export_to_markdown().rstrip() + "\n").encode("utf-8"),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_bytes(json_bytes(manifest.model_dump()))
    load_bundle(output)
    return manifest


def load_bundle(output: Path) -> tuple[Manifest, DoclingDocument]:
    """Check snapshot consistency, not truthfulness or cryptographic authenticity."""
    try:
        manifest = Manifest.model_validate_json((output / "manifest.json").read_bytes())
        source = (output / "source.md").read_bytes()
        serialized = (output / "document.json").read_bytes()
        if sha256(source) != manifest.source_sha256:
            raise FactError("SOURCE_MISMATCH: snapshot source bytes changed")
        if sha256(serialized) != manifest.document_sha256:
            raise FactError("DOCUMENT_MISMATCH: snapshot document bytes changed")
        document = DoclingDocument.model_validate_json(serialized)
        list(walk_body(document))
        return manifest, document
    except (OSError, ValidationError) as exc:
        raise FactError(f"INVALID_BUNDLE: {exc}") from exc


def read_node(output: Path, locator: Locator) -> dict:
    manifest, document = load_bundle(output)
    if (locator.document_id, locator.document_sha256) != (
        manifest.document_id, manifest.document_sha256,
    ):
        raise FactError("STALE_LOCATOR: locator belongs to a different document version")
    reachable = {item.self_ref for item, _ in walk_body(document)}
    if locator.self_ref not in reachable:
        raise FactError(f"MISSING_REF: {locator.self_ref}")
    node = resolve_ref(document, locator.self_ref)
    ancestors = []
    current = node
    while current.parent is not None:
        current = resolve_ref(document, current.parent.cref)
        ancestors.append({
            "self_ref": current.self_ref,
            "label": current.label.value,
            "text": getattr(current, "text", "") or getattr(current, "name", ""),
        })
    return {
        "locator": locator.model_dump(),
        "source_name": manifest.source_name,
        "source_sha256": manifest.source_sha256,
        "ancestors": list(reversed(ancestors)),
        "node": node.model_dump(mode="json", by_alias=True),
    }
