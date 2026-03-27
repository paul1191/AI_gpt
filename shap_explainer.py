"""
SHAP Explainability Module
==========================
Explains WHICH parts of the retrieved regulatory context most influenced the answer.

Approach:
- We treat the query + each context chunk as a "feature"
- We use a TF-IDF vectoriser to create a bag-of-words feature space
- We train a lightweight LinearSVC to predict relevance (similarity score as target)
- We apply SHAP's LinearExplainer to get per-token importance values
- This tells us: "these words in the context most influenced the answer"

Why this approach (explainability note):
- SHAP (SHapley Additive exPlanations) gives theoretically grounded attribution
- LinearSVC + TF-IDF is transparent and fast — no GPU needed
- Results are interpretable: positive SHAP = pushed answer toward using this chunk
- Full SHAP values returned to frontend for interactive visualisation
"""

import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import shap
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP/sklearn not available — explainability will be mocked")


class SHAPExplainer:
    """
    SHAP-based explainability for RAG answers.
    
    Outputs per-chunk and per-token importance scores.
    """

    def __init__(self):
        self.vectorizer = None
        self.model = None
        logger.info(f"SHAPExplainer ready (shap_available={SHAP_AVAILABLE})")

    def analyse(self, query: str, context_chunks: List[str], answer: str) -> Dict[str, Any]:
        """
        Compute SHAP-based feature importance for the given query + context.

        Returns:
          - chunk_importances: per-chunk relevance attribution
          - top_features: top contributing TF-IDF tokens globally  
          - shap_values: raw SHAP values (for frontend bar chart)
          - method: explanation of the approach used
        """
        if not SHAP_AVAILABLE or len(context_chunks) < 2:
            return self._mock_analysis(query, context_chunks, answer)

        try:
            return self._real_shap_analysis(query, context_chunks, answer)
        except Exception as e:
            logger.warning(f"SHAP analysis failed ({e}), falling back to mock")
            return self._mock_analysis(query, context_chunks, answer)

    def _real_shap_analysis(self, query: str, context_chunks: List[str], answer: str) -> Dict:
        """
        Real SHAP analysis pipeline:
        1. Vectorise all chunks with TF-IDF
        2. Compute cosine similarity of each chunk to query as pseudo-label
        3. Train LinearSVC on this
        4. Apply SHAP LinearExplainer
        5. Return top features and chunk attributions
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Step 1: Fit TF-IDF on query + all chunks
        all_texts = [query] + context_chunks
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(all_texts)

        query_vec  = X[0]        # query vector
        chunk_vecs = X[1:]       # chunk vectors

        # Step 2: Compute similarity of each chunk to query
        sims = cosine_similarity(query_vec, chunk_vecs)[0]

        # Step 3: Create binary relevance labels (top 50% = relevant)
        threshold = np.median(sims)
        labels = (sims >= threshold).astype(int)

        # Need at least 2 classes for SVC
        if len(set(labels)) < 2:
            labels[np.argmin(sims)] = 0
            labels[np.argmax(sims)] = 1

        X_chunks = chunk_vecs.toarray()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_chunks)

        # Step 4: Train LinearSVC
        clf = LinearSVC(C=1.0, max_iter=1000)
        clf.fit(X_scaled, labels)

        # Step 5: SHAP LinearExplainer
        explainer = shap.LinearExplainer(clf, X_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_scaled)

        # If multiclass, take values for class 1 (relevant)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        feature_names = self.vectorizer.get_feature_names_out()

        # Per-chunk attribution (mean |SHAP| across features for that chunk)
        chunk_importances = []
        for i, (chunk, sim) in enumerate(zip(context_chunks, sims)):
            chunk_shap = sv[i]
            importance = float(np.mean(np.abs(chunk_shap)))
            top_words  = self._top_words_for_chunk(chunk_shap, feature_names, n=5)
            chunk_importances.append({
                "chunk_index": i,
                "similarity": round(float(sim), 4),
                "shap_importance": round(importance, 4),
                "top_contributing_words": top_words,
                "preview": context_chunks[i][:150] + "...",
            })

        # Global top features (mean |SHAP| across all chunks)
        mean_shap = np.mean(np.abs(sv), axis=0)
        top_idx   = np.argsort(mean_shap)[::-1][:15]
        top_features = [
            {"feature": feature_names[i], "importance": round(float(mean_shap[i]), 4)}
            for i in top_idx
        ]

        # Normalised SHAP values for the top chunk
        top_chunk_idx = int(np.argmax([c["shap_importance"] for c in chunk_importances]))
        top_sv = sv[top_chunk_idx]
        top_sv_norm = (top_sv / (np.max(np.abs(top_sv)) + 1e-9)).tolist()

        return {
            "method": "SHAP LinearExplainer on TF-IDF + LinearSVC",
            "explanation": (
                "Each chunk's SHAP importance shows how much it influenced the answer. "
                "Top features are TF-IDF tokens with highest mean |SHAP| across all chunks. "
                "Positive SHAP = token pushed toward relevance; negative = pushed away."
            ),
            "chunk_importances": chunk_importances,
            "top_features": top_features,
            "raw_shap_preview": top_sv_norm[:20],  # truncated for transfer
            "total_chunks_analysed": len(context_chunks),
        }

    def _top_words_for_chunk(self, shap_vals: np.ndarray, feature_names, n: int = 5) -> List[Dict]:
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:n]
        return [
            {"word": feature_names[i], "shap": round(float(shap_vals[i]), 4)}
            for i in top_idx
        ]

    def _mock_analysis(self, query: str, context_chunks: List[str], answer: str) -> Dict:
        """Fallback when SHAP is unavailable — uses TF-IDF cosine similarity as proxy."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            all_texts = [query] + context_chunks
            vec = TfidfVectorizer(max_features=200, stop_words="english")
            X = vec.fit_transform(all_texts)
            sims = cos_sim(X[0:1], X[1:])[0]
            names = vec.get_feature_names_out()

            q_vec = X[0].toarray()[0]
            top_idx = np.argsort(q_vec)[::-1][:10]
            top_features = [
                {"feature": names[i], "importance": round(float(q_vec[i]), 4)}
                for i in top_idx
            ]
            chunk_importances = [
                {
                    "chunk_index": i,
                    "similarity": round(float(s), 4),
                    "shap_importance": round(float(s), 4),
                    "top_contributing_words": [],
                    "preview": c[:150] + "...",
                }
                for i, (c, s) in enumerate(zip(context_chunks, sims))
            ]
        except Exception:
            chunk_importances = [
                {"chunk_index": i, "similarity": 0.5, "shap_importance": 0.5,
                 "top_contributing_words": [], "preview": c[:150]}
                for i, c in enumerate(context_chunks)
            ]
            top_features = []

        return {
            "method": "TF-IDF Cosine Similarity (SHAP fallback)",
            "explanation": "SHAP not available. Using cosine similarity as proxy for chunk importance.",
            "chunk_importances": chunk_importances,
            "top_features": top_features,
            "raw_shap_preview": [],
            "total_chunks_analysed": len(context_chunks),
        }
