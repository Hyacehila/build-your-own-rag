"""Small, pinned local Docling model bundle, independent of paid model APIs."""

from pathlib import Path

from .common import RagError, read_json, sha256, write_json
from .config import Config
from .dataset import _verify_download, hub_local_dir

LAYOUT_REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
TABLE_REVISION = "fc0f2d45e2218ea24bce5045f58a389aed16dc23"


def verify_models(root: Path) -> dict | None:
    path = root / "manifest.json"
    if not path.exists():
        # Existing user-managed Docling artifacts remain supported.
        return None
    manifest = read_json(path)
    for entry in manifest["files"]:
        file = (root / entry["path"]).resolve()
        if not file.is_relative_to(root.resolve()) or not file.is_file() or sha256(file) != entry["sha256"]:
            raise RagError("Local Docling weights changed or are incomplete; rerun download-models.")
    return manifest


def download_models(config: Config) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    if not config.parsing.artifacts_path:
        raise RagError('Set parsing.artifacts_path (e.g. ".local/models/docling") before download-models.')
    root = Path(config.parsing.artifacts_path)
    bundles = [
        (
            "docling-project/docling-layout-heron",
            LAYOUT_REVISION,
            ["config.json", "preprocessor_config.json", "model.safetensors"],
        ),
        (
            "docling-project/docling-models",
            TABLE_REVISION,
            [
                f"model_artifacts/tableformer/{config.parsing.table_mode}/tm_config.json",
                f"model_artifacts/tableformer/{config.parsing.table_mode}/tableformer_{config.parsing.table_mode}.safetensors",
            ],
        ),
    ]
    manifest = {"models": [], "files": []}
    for repo, revision, names in bundles:
        try:
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise RagError("Hub returned a different local-model revision.")
            upstream = {r.rfilename: r for r in info.siblings}
            for name in names:
                downloaded = Path(
                    hf_hub_download(
                        repo, name, revision=revision, local_dir=hub_local_dir(root / repo.replace("/", "--"))
                    )
                )
                checksum = _verify_download(downloaded, upstream[name], name)
                file = root / repo.replace("/", "--") / name
                manifest["files"].append(
                    {
                        "path": file.relative_to(root).as_posix(),
                        "sha256": checksum,
                        "bytes": file.stat().st_size,
                    }
                )
            manifest["models"].append({"repo": repo, "revision": revision})
        except Exception as e:
            raise RagError(f"Docling model download failed ({type(e).__name__}); rerun to resume.") from e
    write_json(root / "manifest.json", manifest)
    verify_models(root)
    return {"artifacts_path": str(root), "manifest": str(root / "manifest.json"), **manifest}
