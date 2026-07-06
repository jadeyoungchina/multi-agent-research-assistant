from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = "c9877e4d8788a0bb97502348dca8fbd78a6eaa73db0638221b3cb67422d30177"


def normalized_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return sha256(text.encode("utf-8")).hexdigest()


def test_upstream_license_is_vendored_verbatim() -> None:
    assert normalized_sha(ROOT / "LICENSE") == EXPECTED
    assert normalized_sha(ROOT / "THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt") == EXPECTED


def test_notice_pins_reused_notebooks() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "4c95ae14cc2462c442b5c064cccd74430d02bc46" in notice
    for name in (
        "scientific_paper_agent_langgraph.ipynb",
        "document_intake_agent_langgraph.ipynb",
        "EU_Green_Compliance_FAQ_Bot.ipynb",
        "multi_agent_collaboration_system.ipynb",
        "trace_based_agent_evaluation.ipynb",
    ):
        assert name in notice
