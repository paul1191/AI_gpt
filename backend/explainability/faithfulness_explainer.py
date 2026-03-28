"""
RAG Faithfulness Explainer
==========================
Scores each sentence in the LLM answer against the retrieved source chunks.

For each sentence it determines:
  - supported    (max similarity >= HIGH_THRESHOLD)   → green
  - partial      (max similarity >= LOW_THRESHOLD)    → amber
  - unsupported  (max similarity <  LOW_THRESHOLD)    → red

Supported metrics (set via FAITHFULNESS_METRIC env var or metric= parameter):
  "cosine"     — sentence-transformers cosine similarity (default, fast, ~0s extra)
  "bertscore"  — BERTScore F1 using bert-base-uncased (accurate, slow on CPU ~15-30s extra)

BERTScore notes:
  - Requires: pip install bert-score
  - Uses bert-base-uncased by default (configurable via BERTSCORE_MODEL env var)
  - Significantly more accurate for paraphrase detection than cosine
  - NOT recommended as default on CPU — use cosine for interactive queries,
    bertscore for offline batch analysis or if you have a GPU
  - BERTScore thresholds are naturally higher than cosine (F1 scores tend to be
    0.85+ even for unrelated text) so HIGH/LOW thresholds are adjusted automatically

Aggregate signals fed into TrustScoreBuilder:
  grounding_ratio     — fraction of sentences that are supported or partial
  avg_max_similarity  — mean of each sentence's best chunk match
  source_coverage     — fraction of retrieved chunks used as best match by ≥1 sentence

Design:
  - Shared embedder from DocumentStore (no second model load) for cosine path
  - Falls back to TF-IDF cosine if embedder unavailable and metric=cosine
  - Sentence splitting handles "e.g.", "i.e.", "Art. 3" without breaking
  - Strips the References Used footer before analysis
"""
import os
import re
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
# Cosine similarity thresholds (sentence-transformers / TF-IDF)
COSINE_HIGH = 0.60
COSINE_LOW  = 0.35

# BERTScore F1 thresholds — naturally higher than cosine, needs different band
# bert-base-uncased F1: ~0.85 for related, ~0.88+ for well-supported sentences
BERT_HIGH   = 0.88
BERT_LOW    = 0.84

# Default metric — override with FAITHFULNESS_METRIC=bertscore env var
DEFAULT_METRIC   = os.getenv("FAITHFULNESS_METRIC", "cosine").lower()
BERTSCORE_MODEL  = os.getenv("BERTSCORE_MODEL", "bert-base-uncased")

# Try importing bert_score at module load — warn if unavailable
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
    logger.info("bert-score available — BERTScore metric enabled")
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.info("bert-score not installed — BERTScore metric unavailable (pip install bert-score)")

# Regex to split answer into sentences without breaking common abbreviations
_SENT_SPLIT = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s+'
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering very short fragments."""
    clean = text.split("\n\n---\n")[0].strip()
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)
    clean = re.sub(r"\[\d+\]", "", clean)
    sentences = _SENT_SPLIT.split(clean)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class FaithfulnessExplainer:
    """
    Sentence-level faithfulness scorer for RAG answers.

    Usage:
        explainer = FaithfulnessExplainer()

        # Default (cosine, fast):
        result = explainer.analyse(answer=answer, chunks=chunks, embedder=embedder)

        # BERTScore (slower but more accurate):
        result = explainer.analyse(answer=answer, chunks=chunks, embedder=embedder,
                                   metric="bertscore")

        # Or set env var FAITHFULNESS_METRIC=bertscore to make it the default
    """

    def analyse(
        self,
        answer: str,
        chunks: List[Dict],
        embedder=None,
        metric: str = None,       # None → uses DEFAULT_METRIC from env
    ) -> Dict[str, Any]:
        if not chunks or not answer.strip():
            return self._empty()

        sentences   = _split_sentences(answer)
        chunk_texts = [c["text"] for c in chunks]

        if not sentences:
            return self._empty()

        # Resolve metric
        resolved_metric = (metric or DEFAULT_METRIC).lower()
        if resolved_metric == "bertscore" and not BERTSCORE_AVAILABLE:
            logger.warning(
                "BERTScore requested but bert-score not installed. "
                "Falling back to cosine. Run: pip install bert-score"
            )
            resolved_metric = "cosine"

        try:
            return self._embed_analysis(
                sentences, chunk_texts, chunks, embedder, resolved_metric
            )
        except Exception as e:
            logger.warning(f"FaithfulnessExplainer ({resolved_metric}) failed: {e!r} — trying TF-IDF fallback")
            try:
                return self._tfidf_analysis(sentences, chunk_texts, chunks)
            except Exception as e2:
                logger.error(f"TF-IDF fallback also failed: {e2!r}")
                return self._empty()

    # ── Primary analysis dispatcher ───────────────────────────────────────────

    def _embed_analysis(
        self,
        sentences: List[str],
        chunk_texts: List[str],
        chunks: List[Dict],
        embedder,
        metric: str,
    ) -> Dict[str, Any]:
        if metric == "bertscore":
            return self._bertscore_analysis(sentences, chunk_texts, chunks)
        else:
            return self._cosine_analysis(sentences, chunk_texts, chunks, embedder)

    # ── Cosine similarity path (default, fast) ────────────────────────────────

    def _cosine_analysis(
        self,
        sentences: List[str],
        chunk_texts: List[str],
        chunks: List[Dict],
        embedder,
    ) -> Dict[str, Any]:
        if embedder is None:
            # No embedder available — fall through to TF-IDF
            return self._tfidf_analysis(sentences, chunk_texts, chunks)

        all_texts  = sentences + chunk_texts
        all_embs   = embedder.encode(all_texts, show_progress_bar=False)
        sent_embs  = all_embs[:len(sentences)]
        chunk_embs = all_embs[len(sentences):]

        sent_norms  = np.linalg.norm(sent_embs,  axis=1, keepdims=True) + 1e-9
        chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9
        sim_matrix  = (sent_embs / sent_norms) @ (chunk_embs / chunk_norms).T

        return self._build_result(
            sentences, chunks, sim_matrix,
            method="sentence-transformers cosine",
            high_threshold=COSINE_HIGH,
            low_threshold=COSINE_LOW,
        )

    # ── BERTScore path (accurate, slower) ────────────────────────────────────

    def _bertscore_analysis(
        self,
        sentences: List[str],
        chunk_texts: List[str],
        chunks: List[Dict],
    ) -> Dict[str, Any]:
        """
        Computes BERTScore F1 for every (sentence, chunk) pair.

        BERTScore uses contextual BERT embeddings to measure token-level
        recall/precision between two texts — much better than cosine for
        detecting paraphrase and indirect support.

        Implementation detail:
          bert_score.score() takes flat lists of candidates and references.
          We build all (sentence × chunk) pairs, score them in one batch,
          then reshape to (n_sentences, n_chunks) matrix.

        Performance on CPU:
          ~2-4s per pair with bert-base-uncased.
          With 10 sentences × 5 chunks = 50 pairs → ~15-30s total.
          Set BERTSCORE_MODEL=distilbert-base-uncased for ~2× speedup.
        """
        n_sent   = len(sentences)
        n_chunks = len(chunk_texts)

        # Build flat (candidate, reference) pair lists
        # candidate = answer sentence, reference = source chunk
        candidates = [s for s in sentences for _ in chunk_texts]   # repeat each sentence n_chunks times
        references = chunk_texts * n_sent                           # repeat chunk list n_sent times

        logger.info(
            f"BERTScore: scoring {len(candidates)} pairs "
            f"({n_sent} sentences × {n_chunks} chunks) "
            f"with {BERTSCORE_MODEL}"
        )

        _, _, F1 = bert_score_fn(
            candidates,
            references,
            lang="en",
            model_type=BERTSCORE_MODEL,
            batch_size=16,
            verbose=False,
        )

        # Reshape flat F1 tensor → (n_sentences, n_chunks) matrix
        sim_matrix = F1.view(n_sent, n_chunks).numpy()

        logger.info(
            f"BERTScore complete. Score range: "
            f"{sim_matrix.min():.3f} – {sim_matrix.max():.3f}"
        )

        return self._build_result(
            sentences, chunks, sim_matrix,
            method=f"BERTScore F1 ({BERTSCORE_MODEL})",
            high_threshold=BERT_HIGH,
            low_threshold=BERT_LOW,
        )

    # ── TF-IDF fallback (no embedder) ────────────────────────────────────────

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

        return self._build_result(
            sentences, chunks, sim_matrix,
            method="TF-IDF cosine (fallback — no embedder)",
            high_threshold=COSINE_HIGH,
            low_threshold=COSINE_LOW,
        )

    # ── Result builder ────────────────────────────────────────────────────────

    def _build_result(
        self,
        sentences: List[str],
        chunks: List[Dict],
        sim_matrix: np.ndarray,
        method: str,
        high_threshold: float,
        low_threshold: float,
    ) -> Dict[str, Any]:
        scored_sentences   = []
        used_chunk_indices = set()

        for i, sentence in enumerate(sentences):
            sims     = sim_matrix[i]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= high_threshold:
                label = "supported"
                color = "green"
            elif best_sim >= low_threshold:
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

        n             = len(scored_sentences)
        n_supported   = sum(1 for s in scored_sentences if s["label"] == "supported")
        n_partial     = sum(1 for s in scored_sentences if s["label"] == "partial")
        n_unsupported = sum(1 for s in scored_sentences if s["label"] == "unsupported")

        grounding_ratio    = round((n_supported + n_partial) / n, 4) if n else 0.0
        avg_max_similarity = round(
            float(np.mean([s["best_similarity"] for s in scored_sentences])), 4
        ) if n else 0.0
        source_coverage    = round(len(used_chunk_indices) / len(chunks), 4) if chunks else 0.0

        return {
            "method":             method,
            "metric":             "bertscore" if "BERTScore" in method else "cosine",
            "sentences":          scored_sentences,
            "total_sentences":    n,
            "supported_count":    n_supported,
            "partial_count":      n_partial,
            "unsupported_count":  n_unsupported,
            # Signals for TrustScoreBuilder
            "grounding_ratio":    grounding_ratio,
            "avg_max_similarity": avg_max_similarity,
            "source_coverage":    source_coverage,
            # Thresholds used (exposed so UI can show correct scale)
            "high_threshold":     high_threshold,
            "low_threshold":      low_threshold,
        }

    def _empty(self) -> Dict[str, Any]:
        return {
            "method":             "N/A",
            "metric":             DEFAULT_METRIC,
            "sentences":          [],
            "total_sentences":    0,
            "supported_count":    0,
            "partial_count":      0,
            "unsupported_count":  0,
            "grounding_ratio":    0.0,
            "avg_max_similarity": 0.0,
            "source_coverage":    0.0,
            "high_threshold":     COSINE_HIGH,
            "low_threshold":      COSINE_LOW,
        }
