# Concept-only from EU_Green_Compliance_FAQ_Bot.ipynb cells 21, 25, and 36.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Corrected semantics: preserve source metadata, use reciprocal-rank scores, and ignore
# duplicate evidence IDs within each ranking.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from app.domain.documents import EvidenceChunk


def normalize_queries(
    question: str, expansions: list[str], max_expansions: int
) -> list[str]:
    if max_expansions < 0:
        raise ValueError("max_expansions must be non-negative")
    original = question.strip()
    if not original:
        raise ValueError("question must not be blank")

    result = [original]
    seen = {original.casefold()}
    accepted_expansions = 0
    for value in expansions:
        if accepted_expansions >= max_expansions:
            break
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
            accepted_expansions += 1
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def reciprocal_rank_fuse(
    ranked_lists: list[list[EvidenceChunk]], rrf_k: int, top_k: int
) -> list[EvidenceChunk]:
    rrf_k = _positive_integer(rrf_k, "rrf_k")
    top_k = _positive_integer(top_k, "top_k")
    scores: dict[str, float] = {}
    evidence: dict[str, EvidenceChunk] = {}

    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            if item.id in seen_in_list:
                continue
            seen_in_list.add(item.id)
            evidence[item.id] = item
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores, key=lambda evidence_id: (-scores[evidence_id], evidence_id))
    return [
        evidence[evidence_id].model_copy(update={"score": scores[evidence_id]})
        for evidence_id in ordered[:top_k]
    ]
