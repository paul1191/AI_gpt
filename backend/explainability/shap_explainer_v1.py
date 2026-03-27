"""
SHAP Explainability Module
==========================
Explains WHICH parts of the retrieved regulatory context most influenced the answer,
and WHICH tokens drove that influence.

Pipeline:
  1. TF-IDF vectorisation of query + all retrieved chunks (unigrams + bigrams)
  2. Cosine similarity of each chunk to query → pseudo-relevance score
  3. Binary labels: top-50% chunks = relevant (1), rest = not relevant (0)
  4. LogisticRegression classifier trained on TF-IDF features
  5. SHAP LinearExplainer on the LogisticRegression
  6. Per-chunk SHAP importance = mean |SHAP| across all features for that chunk
  7. Global top features = mean |SHAP| across all chunks

Why these design choices:
  ─ LogisticRegression (not LinearSVC): LinearSVC has no predict_proba, which
    SHAP LinearExplainer requires. LogisticRegression is compatible and
    produces the same linear decision boundary.
  ─ StandardScaler(with_mean=False): TF-IDF produces sparse matrices. Setting
    with_mean=True (the default) densifies the matrix and destroys sparsity.
    with_mean=False only scales variance, preserving sparsity.
  ─ TF-IDF max_features=500 with (1,2)-ngrams: captures both single keywords
    and two-word regulatory phrases (e.g. "capital requirements", "tier 1").
  ─ Safe preview slicing: uses min(len, 153) to avoid IndexError on short chunks.
"""
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import shap
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression   # FIX: was LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    SHAP_AVAILABLE = True
    logger.info("SHAP + sklearn loaded — full explainability enabled")
except ImportError as e:
    SHAP_AVAILABLE = False
    logger.warning(f"SHAP/sklearn not available ({e}) — using TF-IDF cosine fallback")


class SHAPExplainer:
    """
    SHAP-based feature importance for RAG regulatory answers.

    Returns:
      method               — description of technique used
      explanation          — plain-English interpretation guide
      chunk_importances    — per-chunk: similarity, SHAP score, top words
      top_features         — top TF-IDF tokens by mean |SHAP| globally
      raw_shap_preview     — normalised SHAP values for top chunk (for chart)
      total_chunks_analysed — count
    """

    def analyse(
        self,
        query: str,
        context_chunks: List[str],
        answer: str,
    ) -> Dict[str, Any]:
        if not context_chunks:
            return self._empty_result()

        if not SHAP_AVAILABLE or len(context_chunks) < 2:
            return self._cosine_fallback(query, context_chunks)

        try:
            return self._shap_analysis(query, context_chunks)
        except Exception as e:
            logger.warning(f"SHAP analysis failed ({e!r}), falling back to cosine")
            return self._cosine_fallback(query, context_chunks)

    # ── Full SHAP pipeline ────────────────────────────────────────────────────

    def _shap_analysis(self, query: str, context_chunks: List[str]) -> Dict:
        # ── Step 1: TF-IDF vectorisation ──────────────────────────────────────
        all_texts = [query] + context_chunks
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
        )
        X_all    = vectorizer.fit_transform(all_texts)
        X_query  = X_all[0]
        X_chunks = X_all[1:]

        feature_names = vectorizer.get_feature_names_out()

        # ── Step 2: Cosine similarity to query → pseudo-label ─────────────────
        sims = cosine_similarity(X_query, X_chunks)[0]   # shape (n_chunks,)

        # ── Step 3: Binary relevance labels ───────────────────────────────────
        threshold = float(np.median(sims))
        labels = (sims >= threshold).astype(int)

        # Guarantee both classes present for classifier
        if len(set(labels)) < 2:
            labels[int(np.argmin(sims))] = 0
            labels[int(np.argmax(sims))] = 1

        # ── Step 4: Scale (sparse-safe) + LogisticRegression ─────────────────
        # FIX: with_mean=False preserves sparsity of TF-IDF matrix
        scaler   = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_chunks)   # still sparse

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_scaled, labels)

        # ── Step 5: SHAP LinearExplainer ──────────────────────────────────────
        # masker = X_scaled as background distribution
        explainer  = shap.LinearExplainer(clf, X_scaled)
        shap_vals  = explainer.shap_values(X_scaled)

        # shap_vals shape: (n_chunks, n_features) for binary classification
        # For LogisticRegression binary, LinearExplainer returns array directly
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[1])   # class 1 = relevant
        else:
            sv = np.array(shap_vals)

        # ── Step 6: Per-chunk attribution ─────────────────────────────────────
        chunk_importances = []
        for i, (chunk, sim) in enumerate(zip(context_chunks, sims)):
            chunk_sv   = sv[i]
            importance = float(np.mean(np.abs(chunk_sv)))
            top_words  = self._top_words(chunk_sv, feature_names, n=5)
            # FIX: safe slice — never crashes on short chunks
            preview    = chunk[:min(len(chunk), 150)]
            if len(chunk) > 150:
                preview += "..."
            chunk_importances.append({
                "chunk_index":             i,
                "similarity":              round(float(sim), 4),
                "shap_importance":         round(importance, 4),
                "top_contributing_words":  top_words,
                "preview":                 preview,
            })

        # ── Step 7: Global top features ───────────────────────────────────────
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        top_idx       = np.argsort(mean_abs_shap)[::-1][:15]
        top_features  = [
            {"feature": feature_names[j], "importance": round(float(mean_abs_shap[j]), 4)}
            for j in top_idx
        ]

        # Normalised SHAP for top chunk (for frontend bar chart)
        best_chunk_idx = int(np.argmax([c["shap_importance"] for c in chunk_importances]))
        top_sv         = sv[best_chunk_idx]
        denom          = float(np.max(np.abs(top_sv))) + 1e-9
        raw_preview    = (top_sv / denom).tolist()[:20]

        return {
            "method": "SHAP LinearExplainer on TF-IDF (1-2 grams) + LogisticRegression",
            "explanation": (
                "Each chunk's SHAP importance = mean |SHAP| across all TF-IDF features. "
                "Top features are tokens with the highest mean |SHAP| globally. "
                "Positive SHAP → token increases relevance score; "
                "negative SHAP → token decreases it."
            ),
            "chunk_importances":      chunk_importances,
            "top_features":           top_features,
            "raw_shap_preview":       raw_preview,
            "total_chunks_analysed":  len(context_chunks),
        }

    # ── Fallback (SHAP unavailable or single chunk) ───────────────────────────

    def _cosine_fallback(self, query: str, context_chunks: List[str]) -> Dict:
        """
        TF-IDF cosine similarity as explainability proxy when SHAP is unavailable.
        Returns same schema so the frontend works identically.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
            X          = vectorizer.fit_transform([query] + context_chunks)
            sims       = cos_sim(X[0:1], X[1:])[0]
            names      = vectorizer.get_feature_names_out()

            q_vec    = X[0].toarray()[0]
            top_idx  = np.argsort(q_vec)[::-1][:10]
            top_features = [
                {"feature": names[j], "importance": round(float(q_vec[j]), 4)}
                for j in top_idx
            ]
            chunk_importances = []
            for i, (chunk, sim) in enumerate(zip(context_chunks, sims)):
                preview = chunk[:min(len(chunk), 150)]
                if len(chunk) > 150:
                    preview += "..."
                chunk_importances.append({
                    "chunk_index":             i,
                    "similarity":              round(float(sim), 4),
                    "shap_importance":         round(float(sim), 4),
                    "top_contributing_words":  [],
                    "preview":                 preview,
                })
        except Exception as e:
            logger.error(f"Cosine fallback also failed: {e}")
            chunk_importances = [
                {
                    "chunk_index": i,
                    "similarity":  0.5,
                    "shap_importance": 0.5,
                    "top_contributing_words": [],
                    "preview": chunk[:min(len(chunk), 150)],
                }
                for i, chunk in enumerate(context_chunks)
            ]
            top_features = []

        return {
            "method": "TF-IDF Cosine Similarity (SHAP unavailable or single chunk)",
            "explanation": (
                "SHAP not available. Chunk importance is approximated by cosine "
                "similarity between the query and each chunk's TF-IDF vector."
            ),
            "chunk_importances":     chunk_importances,
            "top_features":          top_features,
            "raw_shap_preview":      [],
            "total_chunks_analysed": len(context_chunks),
        }

    def _empty_result(self) -> Dict:
        return {
            "method": "N/A",
            "explanation": "No context chunks to analyse.",
            "chunk_importances": [],
            "top_features": [],
            "raw_shap_preview": [],
            "total_chunks_analysed": 0,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _top_words(shap_vals: np.ndarray, feature_names, n: int = 5) -> List[Dict]:
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:n]
        return [
            {"word": feature_names[j], "shap": round(float(shap_vals[j]), 4)}
            for j in top_idx
        ]
