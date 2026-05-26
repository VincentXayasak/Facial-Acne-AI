"""Filter retrieved chunks so RAG stays on acne-related papers."""

from __future__ import annotations

import os
import re

# Cosine distance in Chroma (lower = better). Good matches are often ~0.10–0.20.
MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.28"))

_ACNE_CORE = re.compile(
    r"acne\s*vulgaris|acne\s+treatment|facial\s+acne|acne\s+lesion|acne\s+patient",
    re.IGNORECASE,
)
_ACNE_ANY = re.compile(r"\bacne\b", re.IGNORECASE)

_TREATMENT = re.compile(
    r"benzoyl\s*peroxide|\bBPO\b|retinoid|tretinoin|azelaic\s*acid|isotretinoin|"
    r"topical\s+treatment|antibiotic.*acne|acne\s+treatment|combination\s+therapy|"
    r"randomized|clinical\s+trial",
    re.IGNORECASE,
)

# Title/text hits these but not acne → likely wrong paper for most user questions.
_OFF_TOPIC_WITHOUT_ACNE = re.compile(
    r"eczema|atopic\s+dermatitis|tinea\s+pedis|irritant\s+contact\s+dermatitis|"
    r"vinegar.*folk|artificial\s+acid\s+water",
    re.IGNORECASE,
)


def is_treatment_query(question: str) -> bool:
    q = question.lower()
    return any(
        w in q
        for w in (
            "treat",
            "treatment",
            "routine",
            "product",
            "regimen",
            "skincare",
            "skin care",
            "benzoyl",
            "moisturizer",
            "cleanser",
            "otc",
            "over the counter",
            "how can i",
            "what should i use",
        )
    )


def is_causes_query(question: str) -> bool:
    q = question.lower()
    return any(
        w in q
        for w in ("cause", "why do i", "why am i", "what causes", "hormone", "genetic")
    )


def chunk_passes_relevance(
    text: str,
    title: str,
    distance: float,
    *,
    question: str,
) -> bool:
    """Drop weak or off-topic hits (e.g. eczema papers for 'acne routine')."""
    if distance > MAX_DISTANCE:
        return False

    combined = f"{title}\n{text}"
    has_acne = bool(_ACNE_ANY.search(combined))
    has_acne_core = bool(_ACNE_CORE.search(combined))

    if _OFF_TOPIC_WITHOUT_ACNE.search(combined) and not has_acne:
        return False

    if is_treatment_query(question):
        # Treatment questions need acne + some treatment signal in the excerpt.
        if not has_acne:
            return False
        if not (_TREATMENT.search(combined) or has_acne_core):
            return False
        # Reject generic barrier/eczema care without acne treatment focus.
        if re.search(
            r"irritant contact dermatitis|eczema score|vinegar",
            combined,
            re.IGNORECASE,
        ) and not _TREATMENT.search(combined):
            return False
        return True

    # General / causes: require at least mention of acne.
    return has_acne or has_acne_core
