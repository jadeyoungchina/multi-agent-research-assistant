"""Run explicitly with pytest; these contracts live outside the frozen experiment."""
from pathlib import Path
import importlib.util


SPEC = importlib.util.spec_from_file_location("offline_demo", Path(__file__).with_name("run_demo.py"))
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def test_demo_ignores_environment_and_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAT_PROVIDER", "invalid-provider")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "invalid-provider")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "existing.db"))
    (tmp_path / "existing.db").write_bytes(b"existing-user-data")
    (tmp_path / ".env").write_text("CHAT_PROVIDER=invalid-provider\nUPLOAD_DIR=existing-uploads\n")
    settings = demo.demo_settings(tmp_path / "isolated")
    assert settings.chat_provider == settings.embedding_provider == "fake"
    assert settings.database_path == tmp_path / "isolated/demo.db"
    assert settings.upload_dir == tmp_path / "isolated/uploads"
    assert settings.dashscope_api_key is settings.openai_api_key is settings.deepseek_api_key is None
    assert (tmp_path / "existing.db").read_bytes() == b"existing-user-data"


def test_demo_uses_actual_upload_workflow_sse_and_persistence(tmp_path, monkeypatch):
    import socket

    original_connect = socket.socket.connect

    def deny_network(sock, address):
        # Windows asyncio creates a loopback socket pair for its internal wakeup.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        raise AssertionError("Offline demo attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    result = demo.check_demo(tmp_path / "session")
    assert result["completed"] and result["persisted_report_reopened"]
    assert result["citation_count"] == 2 and result["real_model_requests"] == 0
