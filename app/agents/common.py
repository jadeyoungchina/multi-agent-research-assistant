from app.domain.documents import EvidenceChunk


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
