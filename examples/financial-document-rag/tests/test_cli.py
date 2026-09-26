"""Exercise the public CLI across the new command groups without external services."""

import json
from pathlib import Path

import pytest

from financial_rag import cli, common
from financial_rag.config import Config, Role, Tokens
from financial_rag.storage import Store


def test_cli_four_arm_fixture_and_read_only_audit(tmp_path, fake, monkeypatch, capsys):
    config = Config(work_dir=tmp_path / "work", tokenizer=Tokens(kind="approx", estimated=True))
    for name in ("embedding", "chat", "vision", "judge"):
        setattr(config.models, name, Role(base_url=fake.url, model="offline-double", retries=0))
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

    invoke("fixture", "--structured-snapshot")
    invoke("parse", "--parser", "flat")
    invoke("chunk", "--index", "flat")
    invoke("chunk", "--index", "structured")
    source = invoke("source", "--doc-id", "fictional-bank", "--page", "1")
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
    invoke("audit-data", "--no-gallery")
    assert opened[-1] is True
    assert len(fake.requests) == requests


def test_invalid_run_limit_does_not_open_database(tmp_path, monkeypatch):
    def unexpected_store(*args, **kwargs):
        pytest.fail("Invalid command arguments must not open a working database")

    monkeypatch.setattr(cli, "Store", unexpected_store)
    with pytest.raises(SystemExit) as failure:
        cli.main(["--config", str(tmp_path / "absent.toml"), "run", "--run", "bad", "--limit", "0"])
    assert failure.value.code == 2


def test_run_fingerprint_covers_nested_modules_with_same_filename(tmp_path, monkeypatch):
    package = tmp_path / "project/src/financial_rag"
    package.mkdir(parents=True)
    monkeypatch.setattr(common, "__file__", str(package / "common.py"))
    first = package / "commands/diagnostics.py"
    second = package / "diagnostics/diagnostics.py"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n", encoding="utf-8")
    original = common.implementation()
    first.write_text("value = 2\n", encoding="utf-8")
    changed = common.implementation()
    assert changed != original
    second.write_text("value = 3\n", encoding="utf-8")
    assert common.implementation() != changed
