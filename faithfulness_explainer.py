"""
RAG Faithfulness Explainer
==========================
Scores each sentence in the LLM answer against the retrieved source chunks.

For each sentence it determines:
  - supported    (max cosine sim >= HIGH_THRESHOLD)   → green
  - partial      (max cosine sim >= LOW_THRESHOLD)    → amber
  - unsupported  (max cosine sim <  LOW_THRESHOLD)    → red

Aggregate signals fed into TrustScoreBuilder:
  grounding_ratio     — fraction of sentences that are supported or partial
  avg_max_similarity  — mean of each sentence's best chunk match
  source_coverage     — fraction of retrieved chunks used as best match by ≥1 sentence

Design:
  - Uses SentenceTransformer directly (already loaded in DocumentStore) via
    a shared embedder passed in at call time — no second model load
  - Falls back to TF-IDF cosine if embedder unavailable
  - Sentence splitting is regex-based — handles "e.g.", "i.e.", "Art. 3" safely
  - Strips the References Used footer before analysis so it doesn't inflate scores
"""
import re
import logging
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

HIGH_THRESHOLD = float(0.60)   # supported
LOW_THRESHOLD  = float(0.35)   # partial (below = unsupported)

# Regex to split answer into sentences without breaking common abbreviations
_SENT_SPLIT = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s+'
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering very short fragments."""
    # Strip the References Used footer — everything after the --- divider
    clean = text.split("\n\n---\n")[0].strip()
    # Remove markdown bold markers and reference pills like [1], [2]
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
    clean = re.sub(r"\[\d+\]", "", clean)
    sentences = _SENT_SPLIT.split(clean)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class FaithfulnessExplainer:
    """
    Sentence-level faithfulness scorer for RAG answers.

    Usage:
        explainer = FaithfulnessExplainer()
        result = explainer.analyse(
            answer=answer_text,
            chunks=retrieved_chunks,      # list of {"text": ..., "filename": ...}
            embedder=sentence_transformer  # shared SentenceTransformer instance
        )
    """

    def analyse(
        self,
        answer: str,
        chunks: List[Dict],
        embedder=None,
    ) -> Dict[str, Any]:
        if not chunks or not answer.strip():
            return self._empty()

        sentences   = _split_sentences(answer)
        chunk_texts = [c["text"] for c in chunks]

        if not sentences:
            return self._empty()

        try:
            if embedder is not None:
                return self._embed_analysis(sentences, chunk_texts, chunks, embedder)
            else:
                return self._tfidf_analysis(sentences, chunk_texts, chunks)
        except Exception as e:
            logger.warning(f"FaithfulnessExplainer failed: {e!r}")
            return self._empty()

    # ── Embedding-based analysis (primary) ───────────────────────────────────

    def _embed_analysis(
        self,
        sentences: List[str],
        chunk_texts: List[str],
        chunks: List[Dict],
        embedder,
    ) -> Dict[str, Any]:
        # Encode sentences and chunks in one batch each
        all_texts  = sentences + chunk_texts
        all_embs   = embedder.encode(all_texts, show_progress_bar=False)
        sent_embs  = all_embs[:len(sentences)]
        chunk_embs = all_embs[len(sentences):]

        # Cosine similarity matrix: (n_sentences, n_chunks)
        sent_norms  = np.linalg.norm(sent_embs,  axis=1, keepdims=True) + 1e-9
        chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9
        sim_matrix  = (sent_embs / sent_norms) @ (chunk_embs / chunk_norms).T

        return self._build_result(sentences, chunks, sim_matrix, method="sentence-transformers cosine")

    # ── TF-IDF fallback ───────────────────────────────────────────────────────

    def _tfidf_analysis(
        self,
        sentences: List[str],
        chunk_texts: List[str],
        chunks: List[Dict],
    ) -> Dict[str, Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_texts  = sentences + chunk_texts
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        X          = vectorizer.fit_transform(all_texts)
        sent_X     = X[:len(sentences)]
        chunk_X    = X[len(sentences):]
        sim_matrix = cosine_similarity(sent_X, chunk_X)

        return self._build_result(sentences, chunks, sim_matrix, method="TF-IDF cosine (fallback)")

    # ── Result builder ────────────────────────────────────────────────────────

    def _build_result(
        self,
        sentences: List[str],
        chunks: List[Dict],
        sim_matrix: np.ndarray,
        method: str,
    ) -> Dict[str, Any]:
        scored_sentences = []
        used_chunk_indices = set()

        for i, sentence in enumerate(sentences):
            sims         = sim_matrix[i]           # shape (n_chunks,)
            best_idx     = int(np.argmax(sims))
            best_sim     = float(sims[best_idx])

            if best_sim >= HIGH_THRESHOLD:
                label = "supported"
                color = "green"
            elif best_sim >= LOW_THRESHOLD:
                label = "partial"
                color = "amber"
            else:
                label = "unsupported"
                color = "red"

            if label in ("supported", "partial"):
                used_chunk_indices.add(best_idx)

            best_chunk = chunks[best_idx]
            scored_sentences.append({
                "sentence":        sentence,
                "label":           label,
                "color":           color,
                "best_similarity": round(best_sim, 4),
                "best_chunk_idx":  best_idx,
                "best_source":     best_chunk.get("filename", "unknown"),
                "best_page":       best_chunk.get("page_number", 0),
            })

        # ── Aggregate signals for TrustScoreBuilder ───────────────────────────
        n = len(scored_sentences)
        n_supported = sum(1 for s in scored_sentences if s["label"] == "supported")
        n_partial   = sum(1 for s in scored_sentences if s["label"] == "partial")
        n_unsupported = sum(1 for s in scored_sentences if s["label"] == "unsupported")

        grounding_ratio    = round((n_supported + n_partial) / n, 4) if n else 0.0
        avg_max_similarity = round(
            float(np.mean([s["best_similarity"] for s in scored_sentences])), 4
        ) if n else 0.0
        source_coverage    = round(len(used_chunk_indices) / len(chunks), 4) if chunks else 0.0

        return {
            "method":              method,
            "sentences":           scored_sentences,
            "total_sentences":     n,
            "supported_count":     n_supported,
            "partial_count":       n_partial,
            "unsupported_count":   n_unsupported,
            # Signals exported to TrustScoreBuilder
            "grounding_ratio":     grounding_ratio,
            "avg_max_similarity":  avg_max_similarity,
            "source_coverage":     source_coverage,
            # Thresholds used (for UI display)
            "high_threshold":      HIGH_THRESHOLD,
            "low_threshold":       LOW_THRESHOLD,
        }

    def _empty(self) -> Dict[str, Any]:
        return {
            "method":             "N/A",
            "sentences":          [],
            "total_sentences":    0,
            "supported_count":    0,
            "partial_count":      0,
            "unsupported_count":  0,
            "grounding_ratio":    0.0,
            "avg_max_similarity": 0.0,
            "source_coverage":    0.0,
            "high_threshold":     HIGH_THRESHOLD,
            "low_threshold":      LOW_THRESHOLD,
        }
