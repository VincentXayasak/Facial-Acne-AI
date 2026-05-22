"""Corpus topic taxonomy — aligned with paper_topics.txt."""

from __future__ import annotations

# Keys must match what Gemini returns in relevant_research_topics.
CORPUS_TOPICS: dict[str, str] = {
    "hormones_genetics": (
        "Hormones & Genetics — puberty, androgens, genetics, sebocytes, hormonal acne"
    ),
    "bacteria_inflammation": (
        "Bacteria & Inflammation — Cutibacterium acnes, microbiome, inflammation, innate immunity"
    ),
    "environment_stress": (
        "Environment & Stress — barrier function, stress, sleep, hygiene, cosmetics, sweat"
    ),
    "medicine_skincare": (
        "Medicine & Skincare — retinoids, benzoyl peroxide, azelaic acid, antibiotics, treatments"
    ),
    "diet_habits": (
        "Diet & Habits — dairy, glycemic diet, omega-3, probiotics, sleep, cleansing habits"
    ),
}

# Extra search terms per topic to pull better-matching chunks from Chroma.
TOPIC_SEARCH_TERMS: dict[str, list[str]] = {
    "hormones_genetics": [
        "androgen",
        "hormone",
        "puberty",
        "genetics",
        "sebaceous",
        "sebum",
        "hyperandrogen",
    ],
    "bacteria_inflammation": [
        "Cutibacterium acnes",
        "Propionibacterium",
        "inflammation",
        "inflammatory",
        "microbiome",
        "innate immunity",
        "cytokine",
        "NLRP3",
    ],
    "environment_stress": [
        "stress",
        "sleep",
        "skin barrier",
        "hygiene",
        "cosmetic",
        "sweat",
        "environment",
        "transepidermal water loss",
    ],
    "medicine_skincare": [
        "benzoyl peroxide",
        "retinoid",
        "tretinoin",
        "azelaic acid",
        "treatment",
        "topical",
        "therapy",
        "antibiotic",
        "isotretinoin",
    ],
    "diet_habits": [
        "dairy",
        "diet",
        "glycemic",
        "omega-3",
        "probiotic",
        "nutrition",
        "sleep",
        "washing",
        "cleanser",
    ],
}

CORPUS_TOPICS_PROMPT_BLOCK = "\n".join(
    f'- "{key}": {desc}' for key, desc in CORPUS_TOPICS.items()
)


def infer_topics_from_observation(observation: dict) -> list[str]:
    """Fallback topic tags if Gemini omits relevant_research_topics."""
    text = " ".join(
        str(observation.get(k, ""))
        for k in (
            "acne_types",
            "severity_estimate",
            "inflammation_level",
            "other_observations",
            "scarring_or_post_inflammatory_marks",
        )
    ).lower()

    topics: list[str] = []
    if any(w in text for w in ("papule", "pustule", "nodule", "inflam", "erythem", "red", "swollen")):
        topics.append("bacteria_inflammation")
    if any(w in text for w in ("comedone", "blackhead", "whitehead")):
        topics.append("medicine_skincare")
    if observation.get("apparent_age_group") in ("adolescent", "young adult"):
        topics.append("hormones_genetics")
    if not topics:
        topics = ["bacteria_inflammation", "medicine_skincare"]
    return topics
