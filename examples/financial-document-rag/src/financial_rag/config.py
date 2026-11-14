from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import RagError, digest

DATASET_REVISION = "7f432c176d82e27546501ad8064a713ac3071809"
DATASET_REPO = "vidore/vidore_v3_finance_en"
FINANCE_EXCLUDED_QUERY_IDS = (
    "0", "1", "54", "56", "80", "81", "82", "90", "133", "166", "180", "192", "257"
)
ARMS = {"A": "flat", "B": "structured", "C": "enriched", "pre-D": "structured", "D": "structured"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(Strict):
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    request_params: dict[str, Any] = Field(default_factory=dict)
    cache_revision: str = "1"
    timeout_seconds: float = Field(default=120, gt=0)
    retries: int = Field(default=2, ge=0, le=5)
    json_retries: int = Field(default=5, ge=0, le=5)
    json_mode: bool = False
    batch_size: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def safe_params(self):
        forbidden = {"input", "messages", "model", "tools", "tool_choice", "stream", "api_key", "n"}
        if forbidden & self.request_params.keys():
            raise ValueError(f"request_params may not override {sorted(forbidden)}")
        if self.request_params.get("encoding_format", "float") != "float":
            raise ValueError("Only float embeddings are supported")
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("base_url must be an HTTP(S) URL including the provider's API prefix")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Do not put credentials or query parameters in base_url")
        return self

    def identity(self) -> dict:
        # Credentials, retries and batch size don't change embedding semantics.
        return {
            "base_url": self.base_url.rstrip("/"),
            "model": self.model,
            "request_params": self.request_params,
            "cache_revision": self.cache_revision,
        }

    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def require(self, name: str) -> None:
        if not self.configured():
            raise RagError(f"Configure models.{name}.model and its base_url or base_url_env before this stage.")
        if self.api_key_env and not os.environ.get(self.api_key_env):
            raise RagError(f"Missing environment variable {self.api_key_env} for models.{name}.")


class Models(Strict):
    embedding: Role = Field(default_factory=Role)
    chat: Role = Field(default_factory=Role)
    vision: Role = Field(default_factory=Role)
    judge: Role = Field(default_factory=Role)


class Tokens(Strict):
    kind: Literal["tiktoken", "huggingface", "approx"] = "tiktoken"
    name: str = "cl100k_base"
    revision: str | None = None
    estimated: bool = True

    @model_validator(mode="after")
    def reproducible(self):
        if self.kind == "approx" and not self.estimated:
            raise ValueError("The approximate tokenizer must remain explicitly labeled estimated=true")
        if self.kind == "huggingface" and not Path(self.name).is_dir():
            if not self.revision or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
                raise ValueError("Pin the Hugging Face tokenizer revision to a full commit SHA")
        return self


class Chunking(Strict):
    max_tokens: int = Field(default=800, ge=32)
    overlap: int = Field(default=100, ge=0)
    source_scope: Literal["node", "fragment"] = "node"

    def semantic_options(self) -> dict:
        return self.model_dump()

    @model_validator(mode="after")
    def window(self):
        if self.overlap >= self.max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        return self


class Parsing(Strict):
    ocr: bool = False
    device: str = "cpu"
    threads: int = Field(default=4, ge=1)
    artifacts_path: str | None = None
    model_revision: str = "docling-defaults-2.120.1"
    table_mode: Literal["accurate", "fast"] = "accurate"
    compile_model: bool = False  # eager inference also works without a Windows C++ compiler
    heading_hierarchy: bool = True  # original bookmarks, numbering and font style; no generated titles
    backend: Literal["docling", "pdfium"] = "docling"
    table_cell_matching: bool = True
    force_backend_text: bool = False
    guarded_merges: bool = False
    ocr_engine: Literal["tesseract", "rapidocr_torch"] = "tesseract"
    ocr_mode: Literal["default", "full_page", "pdf_aware_layout_regions"] = "default"

    def semantic_options(self) -> dict:
        # Include nondefault parser choices in the parse identity.
        defaults = {
            "backend": "docling",
            "table_cell_matching": True,
            "force_backend_text": False,
            "guarded_merges": False,
            "ocr_engine": "tesseract",
            "ocr_mode": "default",
        }
        result = self.model_dump(exclude={"artifacts_path", *defaults})
        result.update(
            {
                k: getattr(self, k)
                for k, v in defaults.items()
                if getattr(self, k) != v and (self.ocr or not k.startswith("ocr_"))
            }
        )
        if self.guarded_merges:
            result["merge_contract"] = "full-width-zones-literal-hyphens-orig-spans-v1"
        return result


class Retrieval(Strict):
    sparse_k: int = Field(default=50, ge=1)
    dense_k: int = Field(default=50, ge=1)
    rrf_k: int = Field(default=20, ge=1)
    final_k: int = Field(default=20, ge=1)
    flat_final_k: int = Field(default=10, ge=1)
    summary_supplement_k: int = Field(default=5, ge=0, le=20)
    group_sources: bool = True


class Query(Strict):
    bc_text_tokens: int = Field(default=25000, ge=1)
    text_tokens: int = Field(default=40000, ge=1)
    page_images: int = Field(default=8, ge=0)
    image_scale: float = Field(default=1.5, gt=0, le=4)
    search_preview_tokens: int = Field(default=120, ge=0)
    preview_total_tokens: int = Field(default=3000, ge=0)
    read_max_tokens: int = Field(default=40000, ge=1)
    audit_requests: bool = True
    source_work_dir: Path | None = None
    context_window_tokens: int | None = Field(default=None, ge=1)
    input_estimate_multiplier: float = Field(default=1.2, ge=1, le=2)
    exploration_calls: int = Field(default=24, ge=1, le=100)
    agent_total_images: int = Field(default=16, ge=0, le=100)
    agent_request_images: int = Field(default=8, ge=0, le=100)


class Budget(Strict):
    # One shared ledger counts every provider role and retry before HTTP.
    ledger: Path | None = None
    batch: str = Field(default="default", min_length=1)
    request_limit: int | None = Field(default=None, ge=1)
    token_limit: int | None = Field(default=None, ge=1)
    unknown_usage_limit: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def configured_limit(self):
        if self.ledger is not None and self.request_limit is None:
            raise ValueError("budget.request_limit is required when budget.ledger is set")
        return self


class Enrichment(Strict):
    image_estimate_tokens: int = Field(default=180, ge=1)
    table_estimate_tokens: int = Field(default=220, ge=1)
    max_input_tokens: int = Field(default=12000, ge=100)
    prompt_version: str = "1"


class Pricing(Strict):
    embedding_per_million: float = Field(default=0.5, ge=0)
    currency: str = "CNY"
    label: str = "Planning assumption only; verify actual billing with the provider."


class Dataset(Strict):
    repo: str = DATASET_REPO
    revision: str = DATASET_REVISION
    expected_documents: int = 6
    expected_pages: int = 2942
    expected_english_queries: int = 296
    excluded_query_ids: list[str] | None = None

    @model_validator(mode="after")
    def pinned(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Dataset revision must be a full immutable commit SHA")
        if self.excluded_query_ids is None:
            self.excluded_query_ids = (
                list(FINANCE_EXCLUDED_QUERY_IDS)
                if (self.repo, self.revision) == (DATASET_REPO, DATASET_REVISION)
                else []
            )
        if len(set(self.excluded_query_ids)) != len(self.excluded_query_ids):
            raise ValueError("Excluded query IDs must be unique")
        return self


class Config(Strict):
    work_dir: Path = Path(".local/finance")
    dataset: Dataset = Field(default_factory=Dataset)
    models: Models = Field(default_factory=Models)
    tokenizer: Tokens = Field(default_factory=Tokens)
    chunking: Chunking = Field(default_factory=Chunking)
    parsing: Parsing = Field(default_factory=Parsing)
    retrieval: Retrieval = Field(default_factory=Retrieval)
    query: Query = Field(default_factory=Query)
    enrichment: Enrichment = Field(default_factory=Enrichment)
    pricing: Pricing = Field(default_factory=Pricing)
    budget: Budget = Field(default_factory=Budget)

    def portable_settings(self) -> dict:
        result = self.model_dump(mode="json", exclude={"work_dir", "budget"})
        result["parsing"] = self.parsing.semantic_options()
        result["chunking"] = self.chunking.semantic_options()
        return result

    def fingerprint(self) -> str:
        return digest(self.portable_settings())


def load_config(path: Path | None) -> Config:
    from dotenv import load_dotenv

    base = path.resolve().parent if path else Path.cwd()
    load_dotenv(base / ".env", override=False)
    settings = tomllib.loads(path.read_text(encoding="utf-8")) if path else {}
    # Data preparation does not need a model gateway. Validate a configured URL
    # here, and require a missing one only when the model is actually used.
    for name, role in settings.get("models", {}).items():
        if not isinstance(role, dict) or "base_url_env" not in role:
            continue
        variable = role.pop("base_url_env")
        if not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
            raise RagError(f"models.{name}.base_url_env must name an environment variable")
        if "base_url" in role:
            raise RagError(f"models.{name}: choose base_url or base_url_env, not both")
        value = os.environ.get(variable, "").strip()
        if not value:
            role["base_url"] = ""
            continue
        # Validate without echoing potentially credential-bearing input in errors.
        try:
            Role(base_url=value)
        except ValueError:
            raise RagError(f"{variable} must be an HTTP(S) API URL without credentials, "
                           "query parameters or fragments") from None
        role["base_url"] = value
    config = Config.model_validate(settings)
    config.work_dir = (base / config.work_dir).resolve()
    if config.query.source_work_dir:
        config.query.source_work_dir = (base / config.query.source_work_dir).resolve()
    if config.parsing.artifacts_path:
        config.parsing.artifacts_path = str((base / config.parsing.artifacts_path).resolve())
    if config.budget.ledger:
        config.budget.ledger = (base / config.budget.ledger).resolve()
    return config
