import re

from app.domain.documents import EvidenceChunk
from app.domain.providers import ProviderMetadata, TokenUsage


def format_evidence(evidence: list[EvidenceChunk]) -> str:
    """Render evidence with the source details available to agent prompts."""
    return "\n\n".join(
        "\n".join(
            (
                f"Evidence ID: {chunk.id}",
                f"Source: {chunk.filename}, page {chunk.page_number if chunk.page_number is not None else 'unknown'}, chunk {chunk.chunk_index}",
                f"Content: {chunk.content}",
            )
        )
        for chunk in evidence
    )


class EvidenceLabels:
    """Request-local short labels with exact, reversible evidence identity.

    Models need not copy long hashes. Persisted traces retain original IDs. Only
    explicit ID fields and citation tokens are translated; no claim or citation
    is created, and unknown labels are left for the normal validator to reject.
    """

    def __init__(self, evidence: list[EvidenceChunk]):
        self.allowed = {item.id for item in evidence}
        self.forward = {}
        used = set(self.allowed)
        for index, item in enumerate(evidence):
            label = f"ref_{index + 1:04d}"
            while label in used:
                label += "_"
            if len(item.id) > 24:
                self.forward[item.id] = label
                used.add(label)
        self.reverse = {label: identifier for identifier, label in self.forward.items()}
        self.evidence = [item.model_copy(update={"id": self.forward.get(item.id, item.id)}, deep=True) for item in evidence]

    def original(self, value: str) -> str:
        candidate = re.sub(r"^(?:Evidence ID:|[Cc]ite:)\s*", "", value.strip()).strip()
        if candidate in self.reverse:
            return self.reverse[candidate]
        if candidate in self.allowed:
            return candidate
        if re.fullmatch(r"[0-9a-f]{64}", candidate) and f"ev_{candidate}" in self.allowed:
            return f"ev_{candidate}"
        return value

    def text(self, value: str, *, to_provider: bool = False) -> str:
        def replace(match):
            identifier = match.group(1)
            mapped = self.forward.get(identifier, identifier) if to_provider else self.original(identifier)
            return f"[[cite:{mapped}]]"
        return re.sub(r"\[\[cite:([^\]\r\n]+)\]\]", replace, value)

    def synthesis(self, value, *, to_provider: bool = False):
        result = value.model_copy(deep=True)
        translate = (lambda identifier: self.forward.get(identifier, identifier)) if to_provider else self.original
        for finding in result.findings:
            finding.claim = self.text(finding.claim, to_provider=to_provider)
            finding.supporting_evidence_ids = [translate(identifier) for identifier in finding.supporting_evidence_ids]
            finding.conflicting_evidence_ids = [translate(identifier) for identifier in finding.conflicting_evidence_ids]
        return result

    def draft(self, value, *, to_provider: bool = False):
        result = value.model_copy(deep=True)
        translate = (lambda identifier: self.forward.get(identifier, identifier)) if to_provider else self.original
        result.summary = self.text(result.summary, to_provider=to_provider)
        result.markdown = self.text(result.markdown, to_provider=to_provider)
        for finding in result.findings:
            finding.narrative = self.text(finding.narrative, to_provider=to_provider)
            finding.evidence_ids = [translate(identifier) for identifier in finding.evidence_ids]
        result.limitations = [self.text(text, to_provider=to_provider) for text in result.limitations]
        return result


def combine_provider_metadata(items: list[ProviderMetadata]) -> ProviderMetadata:
    combined = items[-1].model_copy(deep=True)
    combined.usage = TokenUsage(**{
        field: sum(getattr(item.usage, field) for item in items)
        for field in TokenUsage.model_fields
    })
    combined.latency_ms = sum(item.latency_ms for item in items)
    combined.retries = sum(item.retries for item in items)
    combined.attempt_count = sum(item.attempt_count for item in items)
    combined.attempt_ids = [identifier for item in items for identifier in item.attempt_ids]
    combined.usage_complete = all(item.usage_complete for item in items)
    return combined
