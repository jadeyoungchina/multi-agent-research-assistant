from pathlib import Path

from app.config import Settings, get_settings


def test_settings_default_to_dashscope_qwen(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "app.db",
    )
    assert settings.chat_provider == "dashscope"
    assert settings.dashscope_chat_model == "qwen3.7-flash"
    assert settings.max_revision_iterations == 2
    assert settings.max_workflow_steps == 24
    assert settings.max_upload_file_bytes < settings.max_upload_total_bytes


def test_get_settings_reads_explicit_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHAT_PROVIDER=fake\nDASHSCOPE_CHAT_MODEL=qwen3.7-flash\n",
        encoding="utf-8",
    )
    assert get_settings(env_file).chat_provider == "fake"


def test_secret_values_are_hidden_from_repr() -> None:
    settings = Settings(_env_file=None, dashscope_api_key="secret-value")
    assert "secret-value" not in repr(settings)
