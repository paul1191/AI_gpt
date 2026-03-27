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

v1.5 addition:
  query_alignment_tokens — the top N SHAP feature names, exported so TrustScoreBuilder
  can check how many appear in the generated answer (shap_query_alignment signal).

Design decisions:
  ─ LogisticRegression (not LinearSVC): compatible with SHAP LinearExplainer.
  ─ StandardScaler(with_mean=False): preserves TF-IDF sparsity.
  ─ TF-IDF max_features=500 with (1,2)-ngrams: captures regulatory phrases.
  ─ Safe preview slicing: uses min(len, 150) to avoid IndexError on short chunks.
"""
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import shap
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
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
      method                  — description of technique used
      explanation             — plain-English interpretation guide
      chunk_importances       — per-chunk: similarity, SHAP score, top words
      top_features            — top TF-IDF tokens by mean |SHAP| globally
      query_alignment_tokens  — top feature names for TrustScoreBuilder
      raw_shap_preview        — normalised SHAP values for top chunk (for chart)
      total_chunks_analysed   — count
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

        sims      = cosine_similarity(X_query, X_chunks)[0]
        threshold = float(np.median(sims))
        labels    = (sims >= threshold).astype(int)

        if len(set(labels)) < 2:
            labels[int(np.argmin(sims))] = 0
            labels[int(np.argmax(sims))] = 1

        scaler   = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_chunks)

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_scaled, labels)

        explainer = shap.LinearExplainer(clf, X_scaled)
        shap_vals = explainer.shap_values(X_scaled)

        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[1])
        else:
            sv = np.array(shap_vals)

        chunk_importances = []
        for i, (chunk, sim) in enumerate(zip(context_chunks, sims)):
            chunk_sv   = sv[i]
            importance = float(np.mean(np.abs(chunk_sv)))
            top_words  = self._top_words(chunk_sv, feature_names, n=5)
            preview    = chunk[:min(len(chunk), 150)]
            if len(chunk) > 150:
                preview += "..."
            chunk_importances.append({
                "chunk_index":            i,
                "similarity":             round(float(sim), 4),
                "shap_importance":        round(importance, 4),
                "top_contributing_words": top_words,
                "preview":                preview,
            })

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        top_idx       = np.argsort(mean_abs_shap)[::-1][:15]
        top_features  = [
            {"feature": feature_names[j], "importance": round(float(mean_abs_shap[j]), 4)}
            for j in top_idx
        ]

        # v1.5: export top feature names for TrustScoreBuilder query alignment
        query_alignment_tokens = [f["feature"] for f in top_features[:10]]

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
            "chunk_importances":       chunk_importances,
            "top_features":            top_features,
            "query_alignment_tokens":  query_alignment_tokens,
            "raw_shap_preview":        raw_preview,
            "total_chunks_analysed":   len(context_chunks),
        }

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _cosine_fallback(self, query: str, context_chunks: List[str]) -> Dict:
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
            query_alignment_tokens = [f["feature"] for f in top_features[:10]]

            chunk_importances = []
            for i, (chunk, sim) in enumerate(zip(context_chunks, sims)):
                preview = chunk[:min(len(chunk), 150)]
                if len(chunk) > 150:
                    preview += "..."
                chunk_importances.append({
                    "chunk_index":            i,
                    "similarity":             round(float(sim), 4),
                    "shap_importance":        round(float(sim), 4),
                    "top_contributing_words": [],
                    "preview":                preview,
                })
        except Exception as e:
            logger.error(f"Cosine fallback also failed: {e}")
            chunk_importances       = []
            top_features            = []
            query_alignment_tokens  = []

        return {
            "method": "TF-IDF Cosine Similarity (SHAP unavailable or single chunk)",
            "explanation": (
                "SHAP not available. Chunk importance is approximated by cosine "
                "similarity between the query and each chunk's TF-IDF vector."
            ),
            "chunk_importances":      chunk_importances,
            "top_features":           top_features,
            "query_alignment_tokens": query_alignment_tokens,
            "raw_shap_preview":       [],
            "total_chunks_analysed":  len(context_chunks),
        }

    def _empty_result(self) -> Dict:
        return {
            "method":                 "N/A",
            "explanation":            "No context chunks to analyse.",
            "chunk_importances":      [],
            "top_features":           [],
            "query_alignment_tokens": [],
            "raw_shap_preview":       [],
            "total_chunks_analysed":  0,
        }

    @staticmethod
    def _top_words(shap_vals: np.ndarray, feature_names, n: int = 5) -> List[Dict]:
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:n]
        return [
            {"word": feature_names[j], "shap": round(float(shap_vals[j]), 4)}
            for j in top_idx
        ]
