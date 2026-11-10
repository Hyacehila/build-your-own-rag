"""Exercise the public CLI across the new command groups without external services."""

import json
from pathlib import Path

import pytest

from financial_rag import cli
from financial_rag.config import Config, Role
from financial_rag.storage import Store
from financial_rag.validation import doctor


def test_cli_four_arm_and_read_only_audit(local, tmp_path, fake, monkeypatch, capsys):
    config, _ = local
    for name in ("embedding", "chat", "vision", "judge"):
        setattr(config.models, name, Role(base_url=fake.url, model="offline-double", retries=0))
    config.models.chat.request_params = {"max_tokens": 1024}
    proof = tmp_path / "synthetic-capacity.json"
    proof.write_text(json.dumps({
        "model": config.models.chat.model, "base_url": fake.url,
        "verification": "operator_configuration", "source_reference": "In-process CLI test provider",
        "context_window_tokens": 131072, "image_token_upper_bound": 4096,
        "message_overhead_tokens": 16, "request_overhead_tokens": 64, "text_encoding": "cl100k_base",
    }))
    config.query.capacity_evidence_path = proof
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    opened = []

    def open_store(path, *, read_only=False):
        opened.append(read_only)
        return Store(path, read_only=read_only)

    monkeypatch.setattr(cli, "Store", open_store)

    def invoke(*args):
        assert cli.main(list(args)) == 0
        output = capsys.readouterr().out
        # run prints per-question progress before its final JSON object.
        return json.loads(output[output.index("{") :])

    invoke("parse", "--parser", "flat")
    invoke("chunk", "--index", "flat")
    invoke("chunk", "--index", "structured")
    source = invoke("source", "--doc-id", "test-source", "--page", "1")
    assert Path(source["image"]).is_file()
    invoke("enrich")
    invoke("chunk", "--index", "enriched")
    invoke("embed", "--index", "all")
    result = invoke("run", "--run", "cli-fixture", "--arms", "A", "B", "C", "D", "--limit", "1")
    assert result["records"] == 4 and result["failures"] == 0
    assert result["synthetic"] and not result["full_benchmark"]
    scored = invoke("evaluate", "--run", "cli-fixture")
    assert scored["rows"] == 4 and scored["pending"] == 0
    report = invoke("report", "--run", "cli-fixture")
    assert Path(report["report"]).is_file()
    assert Path(invoke("usage")["path"]).is_file()
    requests = len(fake.requests)
    assert invoke("doctor")["dataset_check"] == "passed"
    assert len(fake.requests) == requests
    invoke("audit-data", "--no-gallery")
    assert opened[-1] is True
    assert len(fake.requests) == requests


def test_doctor_reports_unprepared_and_unreadable_inputs(local, tmp_path):
    empty = Config(work_dir=tmp_path / "empty")
    with Store(empty.work_dir) as store:
        report = doctor(empty, store)
        assert report["dataset_check"] == "not_prepared"
        assert not report["failures"]

    config, store = local
    labels = Path(store.meta("dataset")["prepared_dir"]) / "labels.jsonl"
    labels.unlink()
    report = doctor(config, store)
    assert report["dataset_check"] == "failed"
    assert "FileNotFoundError" in report["failures"][0]


def test_invalid_run_limit_does_not_open_database(tmp_path, monkeypatch):
    def unexpected_store(*args, **kwargs):
        pytest.fail("Invalid command arguments must not open a working database")

    monkeypatch.setattr(cli, "Store", unexpected_store)
    with pytest.raises(SystemExit) as failure:
        cli.main(["--config", str(tmp_path / "absent.toml"), "run", "--run", "bad", "--limit", "0"])
    assert failure.value.code == 2
