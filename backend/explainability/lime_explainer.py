"""
LIME Retrieval Explainer
========================
Explains how sensitive the retrieval result is to small changes in the query.

Approach — query perturbation (LIME-inspired):
  1. Tokenise the original query into words
  2. Generate N_PERTURBATIONS variants by randomly masking 1-2 words each time
  3. Run each variant through the DocumentStore retriever
  4. Compare the retrieved chunk sets to the original result
  5. Score each word by how much retrieval changes when it is removed

This answers: "Which words in your query are doing the actual work?"

Signals exported to TrustScoreBuilder:
  stability_score   — mean Jaccard similarity of perturbed results vs original
                      1.0 = retrieval is identical regardless of query wording
                      0.0 = every perturbation returns completely different chunks
  key_terms         — words whose removal most changes the result (high influence)
  stable_terms      — words whose removal has minimal effect (low influence)

Why this matters for trust:
  HIGH stability → the answer is robust to query phrasing — trustworthy
  LOW stability  → the answer depends critically on exact wording — fragile
                   a compliance reviewer should re-query with different phrasing

Design:
  - Uses the DocumentStore's existing hybrid retrieve() — no extra model
  - N_PERTURBATIONS=10 (user-selected): ~4s extra latency
  - Jaccard similarity on chunk ID sets (not text) — fast and exact
  - Words shorter than 3 chars and stopwords are excluded from perturbation
    (removing "the", "of" etc. is not informative)
  - Random seed is fixed per query for reproducibility
"""
import re
import random
import logging
import hashlib
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

N_PERTURBATIONS = 10

# Common English stopwords to skip when perturbing
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "from", "into", "through", "during", "before", "after", "above",
    "below", "between", "each", "that", "this", "these", "those",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "whether", "if", "then", "than", "as", "such", "what",
    "which", "who", "whom", "whose", "how", "when", "where", "why",
}


def _tokenise_query(query: str) -> List[str]:
    """Split query into meaningful word tokens, preserving order."""
    tokens = re.split(r"\s+", query.strip())
    return [t for t in tokens if t]


def _meaningful_tokens(tokens: List[str]) -> List[int]:
    """Return indices of tokens worth perturbing (not stopwords, not tiny)."""
    return [
        i for i, t in enumerate(tokens)
        if len(t) >= 3 and t.lower().strip(".,;:!?()") not in _STOPWORDS
    ]


def _chunk_ids(results: List[Dict]) -> Set[str]:
    """Extract a set of chunk identifiers from retrieval results."""
    return {f"{r.get('doc_id', '')}_{r.get('chunk_index', i)}" for i, r in enumerate(results)}


def _jaccard(a: Set, b: Set) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union        = len(a | b)
    return intersection / union if union else 1.0


class LIMEExplainer:
    """
    LIME-inspired retrieval stability explainer.

    Usage:
        explainer = LIMEExplainer()
        result = explainer.analyse(
            query=original_query,
            original_chunks=retrieved_chunks,
            doc_store=document_store_instance,
            top_k=5,
        )
    """

    def analyse(
        self,
        query: str,
        original_chunks: List[Dict],
        doc_store,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if not original_chunks or not query.strip():
            return self._empty(query)

        try:
            return self._run_perturbations(query, original_chunks, doc_store, top_k)
        except Exception as e:
            logger.warning(f"LIMEExplainer failed: {e!r}")
            return self._empty(query)

    # ── Core perturbation loop ────────────────────────────────────────────────

    def _run_perturbations(
        self,
        query: str,
        original_chunks: List[Dict],
        doc_store,
        top_k: int,
    ) -> Dict[str, Any]:
        tokens    = _tokenise_query(query)
        orig_ids  = _chunk_ids(original_chunks)
        candidates = _meaningful_tokens(tokens)

        if not candidates:
            # Query has no meaningful tokens to perturb
            return {
                "method":          "LIME perturbation",
                "query":           query,
                "tokens":          tokens,
                "n_perturbations": 0,
                "stability_score": 1.0,
                "word_influence":  [],
                "key_terms":       [],
                "stable_terms":    [],
                "perturbations":   [],
                "note":            "Query has no meaningful tokens to perturb.",
            }

        # Deterministic seed based on query content
        seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16) % (2**31)
        rng  = random.Random(seed)

        # Per-token influence accumulator: token_idx → list of (1 - jaccard)
        token_influence: Dict[int, List[float]] = {i: [] for i in candidates}
        perturbation_log = []

        # Generate N_PERTURBATIONS variants, each masking 1-2 meaningful tokens
        for _ in range(N_PERTURBATIONS):
            n_mask = min(rng.randint(1, 2), len(candidates))
            masked_indices = rng.sample(candidates, n_mask)

            # Build perturbed query by dropping masked tokens
            perturbed_tokens = [
                t for i, t in enumerate(tokens) if i not in masked_indices
            ]
            perturbed_query = " ".join(perturbed_tokens).strip()
            if not perturbed_query:
                continue

            # Retrieve with perturbed query
            try:
                perturbed_chunks = doc_store.retrieve(perturbed_query, top_k=top_k)
            except Exception:
                continue

            pert_ids  = _chunk_ids(perturbed_chunks)
            jaccard   = _jaccard(orig_ids, pert_ids)
            influence = round(1.0 - jaccard, 4)   # high = removal caused big change

            for idx in masked_indices:
                token_influence[idx].append(influence)

            perturbation_log.append({
                "masked_words":   [tokens[i] for i in masked_indices],
                "perturbed_query": perturbed_query,
                "jaccard_similarity": round(jaccard, 4),
                "retrieval_changed": jaccard < 0.8,
            })

        # ── Aggregate per-token influence ─────────────────────────────────────
        word_influence = []
        for idx in candidates:
            scores = token_influence[idx]
            if scores:
                mean_inf = round(sum(scores) / len(scores), 4)
            else:
                mean_inf = 0.0
            word_influence.append({
                "word":       tokens[idx],
                "influence":  mean_inf,
                "tests":      len(scores),
            })

        word_influence.sort(key=lambda x: x["influence"], reverse=True)

        # Stability = mean Jaccard across all perturbations
        all_jaccards = [p["jaccard_similarity"] for p in perturbation_log]
        stability_score = round(
            sum(all_jaccards) / len(all_jaccards), 4
        ) if all_jaccards else 1.0

        # Key terms = top 3 most influential (removal hurts retrieval most)
        key_terms   = [w["word"] for w in word_influence[:3] if w["influence"] > 0.1]
        # Stable terms = bottom 3 (removal has minimal effect)
        stable_terms = [w["word"] for w in reversed(word_influence) if w["influence"] < 0.05][:3]

        return {
            "method":           "LIME query perturbation (word masking)",
            "query":            query,
            "tokens":           tokens,
            "n_perturbations":  len(perturbation_log),
            "stability_score":  stability_score,
            "word_influence":   word_influence,
            "key_terms":        key_terms,
            "stable_terms":     stable_terms,
            "perturbations":    perturbation_log,
        }

    def _empty(self, query: str = "") -> Dict[str, Any]:
        return {
            "method":           "LIME perturbation",
            "query":            query,
            "tokens":           [],
            "n_perturbations":  0,
            "stability_score":  1.0,
            "word_influence":   [],
            "key_terms":        [],
            "stable_terms":     [],
            "perturbations":    [],
        }
