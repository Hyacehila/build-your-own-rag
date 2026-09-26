from __future__ import annotations

from .api import ModelAPI
from .chunking import make_chunk
from .common import RagError, digest
from .config import Config
from .parsing import active_parses
from .storage import Store
from .tokenization import Tokenizer, recursive_spans


def image_input(node: dict) -> bool:
    # Some original bitmap diagrams are detected as tables with no cells/text.
    # Keep the parser's label and provenance; describe the actual crop in C.
    return node["kind"] == "picture" or (node["kind"] == "table" and not node["text"].strip())


def enrichment_key(config: Config, node: dict) -> str:
    role = config.models.vision if image_input(node) else config.models.chat
    return digest(
        {
            "stage": "enrichment",
            "node": node,
            "model": role.identity(),
            "options": config.enrichment.model_dump(),
            "contract": "asset-descriptions-with-empty-table-image-v2",
            "tokenizer": config.tokenizer.model_dump(),
            "image_scale": config.query.image_scale,
        }
    )


def enrich(config: Config, store: Store, node_ids: list[str] | None = None) -> dict:
    from .dataset import dataset
    from .sources import render_source

    dataset(store, verify_pdfs=True)
    parses = active_parses(config, store, "structured")
    nodes = [n for p in parses for n in store.nodes(p["id"]) if n["kind"] in ("picture", "table")]
    if node_ids is not None:
        if not set(node_ids) <= {n["id"] for n in nodes}:
            raise RagError("Requested enrichment nodes are outside the active structured facts.")
        nodes = [n for n in nodes if n["id"] in node_ids]
    tokenizer = Tokenizer(config.tokenizer)
    pending = [n for n in nodes if store.cached(enrichment_key(config, n)) is None]
    for role in {"vision" if image_input(n) else "chat" for n in pending}:
        getattr(config.models, role).require(role)
    report = {
        "cached": len(nodes) - len(pending),
        "completed": [],
        "failed": [],
        "empty_table_image_fallbacks": [n["id"] for n in nodes if n["kind"] == "table" and image_input(n)],
    }
    for node in pending:
        key = enrichment_key(config, node)
        api = ModelAPI(config, store, "enrich:" + key)
        try:
            if not node["sources"]:
                raise RagError("Asset has no PDF page provenance.")
            if image_input(node):
                input_mode = "image" if node["kind"] == "picture" else "empty_table_image_fallback"
                prompt = (
                    "Describe this document figure for retrieval. Preserve labels, units, axes, relationships, "
                    "trends and numbers that are clearly visible. Do not guess unclear values. "
                    "Use only this original image and caption. Caption: " + node.get("caption", "")
                )
                if input_mode == "empty_table_image_fallback":
                    prompt += (
                        "\nThe parser marked this region as a table but extracted no cells. "
                        "Describe what is actually visible, whether a table, diagram or other figure. "
                        "Do not invent missing table cells."
                    )
                parts = [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
                for source in node["sources"]:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": render_source(store, source, config.query.image_scale, crop=True)
                            },
                        }
                    )
                answer = api.chat("vision", [{"role": "user", "content": parts}])["content"]
                segment_count = 1
            else:
                input_mode = "table_text"
                # Long tables are summarized in bounded segments, then combined using all segment summaries.
                # No question, answer or relevance file is loaded anywhere in this stage.
                spans = recursive_spans(node["text"], tokenizer, config.enrichment.max_input_tokens)
                if not spans:
                    raise RagError("Empty table cannot be summarized.")
                summaries = []
                prompt = (
                    "Summarize this financial table as a retrieval entry. Include subject, column names, "
                    "row categories, time periods and units. Do not invent facts. The full table will be "
                    "read separately for exact values. Original caption: " + node.get("caption", "") + "\n"
                )
                for a, b in spans:
                    summaries.append(
                        api.chat("chat", [{"role": "user", "content": prompt + node["text"][a:b]}])["content"]
                    )
                segment_count = len(summaries)
                # Hierarchical reduction prevents truncating later segments of large tables.
                while len(summaries) > 1:
                    joined = "\n\n".join(summaries)
                    groups = recursive_spans(joined, tokenizer, config.enrichment.max_input_tokens)
                    reduced = [
                        api.chat(
                            "chat",
                            [
                                {
                                    "role": "user",
                                    "content": "Combine these summaries of consecutive segments of ONE table into a concise whole-table "
                                    "retrieval summary. Preserve subjects, column names, periods and units.\n"
                                    + joined[a:b],
                                }
                            ],
                        )["content"]
                        for a, b in groups
                    ]
                    if len(reduced) >= len(summaries):
                        raise RagError(
                            "Table summaries did not shrink; increase max_input_tokens or limit model output."
                        )
                    summaries = reduced
                answer = summaries[0]
            if not answer.strip():
                raise RagError("Empty enrichment response.")
            store.cache(
                key,
                {
                    "id": key,
                    "node_id": node["id"],
                    "text": answer,
                    "sources": node["sources"],
                    "kind": node["kind"],
                    "segment_count": segment_count,
                    "input_mode": input_mode,
                },
            )
            report["completed"].append(node["id"])
        except Exception as e:
            report["failed"].append({"node_id": node["id"], "type": type(e).__name__, "error": str(e)})
            store.set_meta("enrichment-error:" + key, report["failed"][-1])
    return report


def enriched_chunks(config: Config, store: Store, base: dict) -> tuple[list[dict], list[str]]:
    # Body text is reused byte for byte. Asset entries replace table-row / caption-only entries.
    output = [
        {k: v for k, v in c.items() if k not in {"id", "index_id"}}
        for c in store.chunks(base["id"])
        if c["kind"] not in ("table", "picture")
    ]
    enhancement_ids = []
    tokenizer = Tokenizer(config.tokenizer)
    for pid in base["parse_ids"]:
        nodes = store.nodes(pid)
        by_ref = {n["ref"]: n for n in nodes}
        for node in nodes:
            if node["kind"] not in ("table", "picture"):
                continue
            key = enrichment_key(config, node)
            record = store.cached(key)
            if record is None:
                raise RagError(
                    f"Missing enhancement for {node['ref']}; run enrich first (or estimate-embedding)."
                )
            enhancement_ids.append(key)
            text = "\n".join([*node["headings"], node.get("caption", ""), record["text"]]).strip()
            source_nodes = [node] + [by_ref[r] for r in node["related_refs"] if r in by_ref]
            output.append(
                make_chunk(
                    text,
                    source_nodes,
                    node["kind"],
                    node["headings"],
                    asset_ids=[node["id"]],
                    enrichment_id=key,
                    token_overflow=tokenizer.count(text) > config.chunking.max_tokens,
                )
            )
    return output, enhancement_ids
