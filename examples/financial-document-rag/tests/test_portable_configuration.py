"""Configuration is usable without a model service until a model is requested."""

import json

import pytest

from financial_rag import cli
from financial_rag.commands import data
from financial_rag.common import RagError
from financial_rag.config import Role, load_config


def test_environment_route_from_process_or_dotenv(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[models.chat]\nbase_url_env = "TEST_RAG_URL"\nmodel = "model"\n')
    (tmp_path / ".env").write_text("TEST_RAG_URL=https://dotenv.example/v1\n")
    monkeypatch.setenv("TEST_RAG_URL", "https://first.example/v1")
    assert (
        load_config(path).models.chat.identity()
        == Role(base_url="https://first.example/v1", model="model").identity()
    )
    monkeypatch.delenv("TEST_RAG_URL")
    assert load_config(path).models.chat.base_url == "https://dotenv.example/v1"
    monkeypatch.delenv("TEST_RAG_URL")


def test_missing_gateway_is_allowed_until_model_use(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[models.chat]\nbase_url_env = "TEST_RAG_URL"\nmodel = "model"\n')
    monkeypatch.delenv("TEST_RAG_URL", raising=False)
    cfg = load_config(path)
    assert not cfg.models.chat.configured()
    with pytest.raises(RagError, match="base_url"):
        cfg.models.chat.require("chat")


def test_data_command_does_not_require_model_gateway(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    path.write_text('work_dir = "workspace"\n[models.chat]\nbase_url_env = "TEST_RAG_URL"\nmodel = "model"\n')
    monkeypatch.delenv("TEST_RAG_URL", raising=False)
    monkeypatch.setattr(data, "download", lambda args, config, store: {"prepared": True})
    assert cli.main(["--config", str(path), "download"]) == 0
    assert json.loads(capsys.readouterr().out)["prepared"] is True


@pytest.mark.parametrize(
    "value",
    [
        "ftp://example.org",
        "https://user:private-value@example.org/v1",
        "https://example.org/v1?key=private-value",
    ],
)
def test_invalid_route_does_not_echo_secrets(tmp_path, monkeypatch, value):
    path = tmp_path / "config.toml"
    path.write_text('[models.chat]\nbase_url_env = "TEST_RAG_URL"\nmodel = "model"\n')
    monkeypatch.setenv("TEST_RAG_URL", value)
    with pytest.raises(RagError, match="TEST_RAG_URL") as error:
        load_config(path)
    assert "private-value" not in str(error.value)


def test_literal_route_and_offline_config(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_RAG_BASE_URL", "https://other.example/v1")
    path = tmp_path / "config.toml"
    path.write_text('[models.chat]\nbase_url = "https://model.example/v1"\nmodel = "model"\n')
    assert load_config(path).models.chat.base_url == "https://model.example/v1"
    path.write_text('work_dir = "offline"')
    assert not load_config(path).models.chat.configured()


def test_conflicting_route_fields_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RAG_URL", "https://model.example/v1")
    path = tmp_path / "config.toml"
    path.write_text('[models.chat]\nbase_url = "https://model.example/v1"\nbase_url_env = "TEST_RAG_URL"\n')
    with pytest.raises(RagError, match="not both"):
        load_config(path)
