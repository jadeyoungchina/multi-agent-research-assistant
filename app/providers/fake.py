from collections import deque
from hashlib import sha256
from math import sqrt
import re
from time import perf_counter
from typing import Sequence, TypeVar

from pydantic import BaseModel

from app.domain.providers import ChatMessage, ProviderMetadata
from app.domain.research import Critique, DraftReport, ResearchPlan, ResearchSynthesis

T = TypeVar("T", bound=BaseModel)


class FakeChatProvider:
    def __init__(
        self,
        text_responses: Sequence[str] = (),
        structured_responses: Sequence[BaseModel] = (),
    ) -> None:
        self._text = deque(text_responses)
        self._structured = deque(structured_responses)

    def generate(self, messages: Sequence[ChatMessage]) -> tuple[str, ProviderMetadata]:
        started = perf_counter()
        if not self._text:
            raise AssertionError("no fake text response queued")
        value = self._text.popleft()
        return value, ProviderMetadata(
            provider="fake",
            model="fake-chat",
            latency_ms=int((perf_counter() - started) * 1000),
        )

    def generate_structured(
        self, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, ProviderMetadata]:
        started = perf_counter()
        if not self._structured:
            raise AssertionError(f"no fake response queued for {schema.__name__}")
        value = self._structured.popleft()
        if not isinstance(value, schema):
            raise AssertionError(f"expected {schema.__name__}, got {type(value).__name__}")
        return value, ProviderMetadata(
            provider="fake",
            model="fake-chat",
            latency_ms=int((perf_counter() - started) * 1000),
        )


class DemoFakeChatProvider:
    """Offline extraction for demonstrations, with no benchmark answers or model claims.

    Only runtime messages are inspected. Token usage stays zero (no tokenizer or
    real model is involved), while each call still produces provider metadata.
    """

    @staticmethod
    def _metadata() -> ProviderMetadata:
        return ProviderMetadata(provider="fake", model="fake-chat", latency_ms=0)

    @staticmethod
    def _question(messages: Sequence[ChatMessage]) -> str:
        text = next((message.content for message in reversed(messages) if message.role == "user"), "")
        text = re.sub(r"^(?:Research question|Question):\s*\n", "", text)
        return text.split("\n\n", 1)[0].strip() or "Summarize supplied evidence"

    @staticmethod
    def _evidence(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
        text = next((message.content for message in reversed(messages) if message.role == "user"), "")
        if "Supplied evidence:\n" not in text:
            return []
        text = text.split("Supplied evidence:\n", 1)[1]
        blocks = re.findall(
            r"^Evidence ID: ([A-Za-z0-9_-]+)\nSource: [^\n]*\nContent: (.*?)(?=\n\nEvidence ID: |\Z)",
            text, flags=re.MULTILINE | re.DOTALL,
        )
        evidence = []
        seen = set()
        for evidence_id, content in blocks:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            # Turn source lines into separate sentences so extraction stays readable.
            sentences = [line.strip().lstrip("-*># ").strip() for line in content.splitlines() if line.strip()]
            narrative = "\n".join(line if line.endswith((".", "!", "?", "。", "！", "？")) else line + "." for line in sentences)
            if narrative:
                evidence.append((evidence_id, narrative))
        return evidence

    def generate(self, messages: Sequence[ChatMessage]) -> tuple[str, ProviderMetadata]:
        evidence = self._evidence(messages)
        answer = "\n".join(text for _, text in evidence) or "No supplied evidence is available in this offline demonstration."
        return answer, self._metadata()

    @staticmethod
    def _literal_markdown(text: str) -> str:
        # Source syntax must not become active citation tokens in the draft.
        return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    def generate_structured(self, messages: Sequence[ChatMessage], schema: type[T]) -> tuple[T, ProviderMetadata]:
        question = self._question(messages)
        evidence = self._evidence(messages)
        if schema is ResearchPlan:
            payload = dict(objective=question, subquestions=[question], search_queries=[question], completion_criteria=["Supplied evidence"])
        elif schema is ResearchSynthesis:
            payload = dict(findings=[
                dict(claim=f"Supplied evidence: {text}", supporting_evidence_ids=[evidence_id], confidence="high")
                for evidence_id, text in evidence
            ])
        elif schema is Critique:
            payload = dict(sufficient=bool(evidence), reason="Supplied evidence exists." if evidence else "No supplied evidence.")
        elif schema is DraftReport:
            payload = dict(
                title="Offline evidence summary", summary="Deterministic extraction of supplied evidence.",
                findings=[dict(heading="Evidence", narrative=text, evidence_ids=[evidence_id]) for evidence_id, text in evidence],
                limitations=["Offline extraction demonstrates the pipeline, not real-model research quality."],
                markdown="\n\n".join(f"{self._literal_markdown(text)}\n[[cite:{evidence_id}]]" for evidence_id, text in evidence) or "No supplied evidence.",
            )
        elif set(schema.model_fields) == {"queries"}:
            payload = dict(queries=[question])
        elif set(schema.model_fields) == {"answer_markdown", "cited_evidence_ids"}:
            payload = dict(
                answer_markdown="\n".join(text for _, text in evidence) or "No supplied evidence.",
                cited_evidence_ids=[evidence_id for evidence_id, _ in evidence],
            )
        else:
            raise ValueError("unsupported offline demonstration schema")
        return schema.model_validate(payload), self._metadata()


def build_demo_fake_provider() -> DemoFakeChatProvider:
    """Build an unqueued, evidence-aware fake for the demo and evaluation CLI."""
    return DemoFakeChatProvider()


class DeterministicEmbeddingProvider:
    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return f"fake-hash-{self._dimensions}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        encoded = text.casefold().encode("utf-8")
        for offset in range(0, len(encoded) or 1, 4):
            digest = sha256(encoded[offset : offset + 4]).digest()
            index = int.from_bytes(digest[:2], "big") % self._dimensions
            vector[index] += -1.0 if digest[2] & 1 else 1.0
        norm = sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ProviderMetadata]:
        started = perf_counter()
        vectors = [self._embed(text) for text in texts]
        return vectors, ProviderMetadata(
            provider="fake",
            model=self.model_name,
            latency_ms=int((perf_counter() - started) * 1000),
        )

    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]:
        started = perf_counter()
        return self._embed(text), ProviderMetadata(
            provider="fake",
            model=self.model_name,
            latency_ms=int((perf_counter() - started) * 1000),
        )
