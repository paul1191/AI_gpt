"""
Trust Score Builder
===================
Combines signals from three explainability layers into a single composite
trust score with a full breakdown.

Formula:
  trust_score = (
      w_grounding       × grounding_ratio        [RAG Faithfulness]
    + w_similarity      × avg_max_similarity      [RAG Faithfulness]
    + w_coverage        × source_coverage         [RAG Faithfulness]
    + w_shap_alignment  × shap_query_alignment    [SHAP]
    + w_lime_stability  × lime_stability_score    [LIME]
  )

Default weights (sum to 1.0):
  grounding_ratio:       0.30  — most important: are sentences actually supported?
  avg_max_similarity:    0.25  — how strongly does each sentence match a source?
  source_coverage:       0.15  — did the answer use the retrieved evidence broadly?
  shap_query_alignment:  0.15  — do SHAP-important tokens appear in the answer?
  lime_stability:        0.15  — is retrieval stable to query rephrasing?

Why each signal catches what the others miss:
  grounding_ratio      — catches hallucinated sentences (no source match)
  avg_max_similarity   — catches weak paraphrase (sentence vaguely matches but not faithfully)
  source_coverage      — catches over-reliance on one chunk, ignoring others
  shap_query_alignment — catches answer drift (LLM answered a different question)
  lime_stability       — catches fragile answers dependent on exact query wording

Trust level bands:
  0.85 – 1.00  → HIGH    — safe to cite in compliance report
  0.65 – 0.84  → MEDIUM  — review flagged sentences before citing
  0.40 – 0.64  → LOW     — significant gaps, re-query with more context
  0.00 – 0.39  → VERY LOW — do not rely on this answer
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ── Weights ───────────────────────────────────────────────────────────────────
W_GROUNDING      = 0.30
W_SIMILARITY     = 0.25
W_COVERAGE       = 0.15
W_SHAP_ALIGNMENT = 0.15
W_LIME_STABILITY = 0.15

# Trust level bands
BANDS = [
    (0.85, "HIGH",     "#10B981", "Safe to cite in compliance documentation."),
    (0.65, "MEDIUM",   "#F97316", "Review flagged sentences before citing."),
    (0.40, "LOW",      "#EF4444", "Significant gaps — re-query with more context."),
    (0.00, "VERY LOW", "#7F1D1D", "Do not rely on this answer without verification."),
]


def compute_shap_query_alignment(
    shap_data: Dict[str, Any],
    answer: str,
) -> float:
    """
    Measures how well the SHAP top-features (tokens that drove retrieval)
    appear in the generated answer.

    Intuition: if SHAP says "CET1" and "capital ratio" were the most important
    retrieval tokens, and they also appear in the answer, the LLM answered
    the right question. If the answer contains none of those terms, it may
    have drifted to a different topic.

    Returns float 0.0–1.0.
    """
    top_features = shap_data.get("top_features", [])
    if not top_features or not answer.strip():
        return 0.5   # neutral — no signal

    answer_lower = answer.lower()
    top_n = min(10, len(top_features))
    top_words = [f["feature"].lower() for f in top_features[:top_n]]

    # Count how many top SHAP features appear in the answer
    hits = sum(1 for w in top_words if w in answer_lower)
    alignment = hits / top_n if top_n else 0.5

    # Weighted by importance: higher-ranked features count more
    weighted_hits = sum(
        (top_n - i) / top_n
        for i, w in enumerate(top_words)
        if w in answer_lower
    )
    weighted_score = (2 * weighted_hits / top_n) if top_n else 0.5

    # Blend raw hit rate and weighted score
    return round(min(1.0, (alignment + weighted_score) / 2), 4)


def build_trust_score(
    faithfulness: Dict[str, Any],
    shap_data:    Dict[str, Any],
    lime_data:    Dict[str, Any],
    answer:       str,
) -> Dict[str, Any]:
    """
    Compute composite trust score from all three explainability signals.

    Returns:
      trust_score     — float 0.0–1.0
      trust_level     — "HIGH" | "MEDIUM" | "LOW" | "VERY LOW"
      trust_color     — hex colour for UI
      trust_message   — plain-English interpretation
      components      — breakdown of each signal's contribution
      weights         — weights used (for transparency)
    """

    # ── Extract signals ───────────────────────────────────────────────────────
    grounding_ratio    = float(faithfulness.get("grounding_ratio",    0.5))
    avg_max_similarity = float(faithfulness.get("avg_max_similarity", 0.5))
    source_coverage    = float(faithfulness.get("source_coverage",    0.5))
    lime_stability     = float(lime_data.get("stability_score",       1.0))
    shap_alignment     = compute_shap_query_alignment(shap_data, answer)

    # ── Weighted sum ──────────────────────────────────────────────────────────
    raw_score = (
        W_GROUNDING      * grounding_ratio
      + W_SIMILARITY     * avg_max_similarity
      + W_COVERAGE       * source_coverage
      + W_SHAP_ALIGNMENT * shap_alignment
      + W_LIME_STABILITY * lime_stability
    )
    trust_score = round(min(1.0, max(0.0, raw_score)), 4)

    # ── Trust band ────────────────────────────────────────────────────────────
    trust_level   = "VERY LOW"
    trust_color   = "#7F1D1D"
    trust_message = "Do not rely on this answer without verification."
    for threshold, level, color, message in BANDS:
        if trust_score >= threshold:
            trust_level   = level
            trust_color   = color
            trust_message = message
            break

    # ── Per-component breakdown ───────────────────────────────────────────────
    components = [
        {
            "name":        "Sentence Grounding",
            "signal":      "grounding_ratio",
            "source":      "RAG Faithfulness",
            "raw_value":   round(grounding_ratio, 4),
            "weight":      W_GROUNDING,
            "contribution": round(W_GROUNDING * grounding_ratio, 4),
            "description": f"{round(grounding_ratio * 100)}% of answer sentences are supported or partially supported by source chunks.",
        },
        {
            "name":        "Source Similarity",
            "signal":      "avg_max_similarity",
            "source":      "RAG Faithfulness",
            "raw_value":   round(avg_max_similarity, 4),
            "weight":      W_SIMILARITY,
            "contribution": round(W_SIMILARITY * avg_max_similarity, 4),
            "description": f"Average best-match cosine similarity between answer sentences and source chunks: {round(avg_max_similarity, 3)}.",
        },
        {
            "name":        "Source Coverage",
            "signal":      "source_coverage",
            "source":      "RAG Faithfulness",
            "raw_value":   round(source_coverage, 4),
            "weight":      W_COVERAGE,
            "contribution": round(W_COVERAGE * source_coverage, 4),
            "description": f"{round(source_coverage * 100)}% of retrieved chunks were used as evidence for at least one sentence.",
        },
        {
            "name":        "Query Alignment",
            "signal":      "shap_query_alignment",
            "source":      "SHAP",
            "raw_value":   round(shap_alignment, 4),
            "weight":      W_SHAP_ALIGNMENT,
            "contribution": round(W_SHAP_ALIGNMENT * shap_alignment, 4),
            "description": f"SHAP top retrieval tokens appear in the answer at {round(shap_alignment * 100)}% alignment — measures whether the LLM answered the right question.",
        },
        {
            "name":        "Retrieval Stability",
            "signal":      "lime_stability",
            "source":      "LIME",
            "raw_value":   round(lime_stability, 4),
            "weight":      W_LIME_STABILITY,
            "contribution": round(W_LIME_STABILITY * lime_stability, 4),
            "description": f"Retrieval Jaccard stability across {lime_data.get('n_perturbations', 0)} query perturbations: {round(lime_stability, 3)}. Values below 0.7 indicate fragile retrieval.",
        },
    ]

    # Sort by contribution descending so UI can show biggest factors first
    components.sort(key=lambda x: x["contribution"], reverse=True)

    # ── Key terms and warnings ────────────────────────────────────────────────
    warnings = []
    if grounding_ratio < 0.5:
        n_unsupported = faithfulness.get("unsupported_count", 0)
        warnings.append(f"{n_unsupported} sentence(s) have no supporting source chunk — possible hallucination.")
    if lime_stability < 0.6:
        key_terms = lime_data.get("key_terms", [])
        terms_str = ", ".join(f'"{t}"' for t in key_terms) if key_terms else "specific query words"
        warnings.append(f"Retrieval is unstable — result depends heavily on {terms_str}. Try rephrasing.")
    if shap_alignment < 0.3:
        warnings.append("Answer may have drifted from the query — SHAP top tokens have low presence in the answer.")
    if source_coverage < 0.4:
        warnings.append("Answer draws heavily from a single source chunk — broader evidence may be available.")

    return {
        "trust_score":   trust_score,
        "trust_level":   trust_level,
        "trust_color":   trust_color,
        "trust_message": trust_message,
        "components":    components,
        "weights": {
            "grounding_ratio":    W_GROUNDING,
            "avg_max_similarity": W_SIMILARITY,
            "source_coverage":    W_COVERAGE,
            "shap_alignment":     W_SHAP_ALIGNMENT,
            "lime_stability":     W_LIME_STABILITY,
        },
        "warnings": warnings,
        "key_terms":     lime_data.get("key_terms", []),
        "stable_terms":  lime_data.get("stable_terms", []),
    }
