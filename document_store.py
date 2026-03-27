"""
Document Store — ChromaDB + BM25 Hybrid Retrieval

v1.4 changes (retrieval quality fixes):
  - Hybrid search: BM25 keyword + ChromaDB semantic, merged with Reciprocal Rank Fusion
  - Similarity threshold: chunks below MIN_SIMILARITY are discarded rather than returned
  - BM25 index rebuilt in-memory on startup from ChromaDB stored documents
  - Multi-pass retrieval: fetches 3× top_k candidates before re-ranking (wider net)
  - debug_retrieve(): shows full scoring breakdown for diagnosing missed keywords
  - Page number + clause ref extraction preserved from v1.2

Why hybrid search fixes the observed failures:
  SHORT KEYWORDS (1-2 words): semantic embeddings of very short queries produce
    low-confidence vectors that don't reliably match anything. BM25 is exact
    keyword matching — "CET1" scores highly on any chunk containing "CET1".
  PARAPHRASED QUERIES: semantic catches these well. BM25 catches literal matches.
    RRF fusion means a chunk that ranks well in EITHER method gets promoted.

Reciprocal Rank Fusion formula:
  RRF(chunk) = 1/(k + semantic_rank) + 1/(k + bm25_rank)
  k=60 is standard — dampens the effect of very high ranks.
  A chunk ranked #1 in BM25 and #20 in semantic beats one ranked #5 in both.
"""
import io
import os
import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_PATH    = os.getenv("CHROMA_PATH",    "./chroma_db")
EMBED_MODEL    = os.getenv("EMBED_MODEL",    "all-MiniLM-L6-v2")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE",     "800"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP",  "200"))
MAX_PARA_CHARS = 1000

# Minimum cosine similarity to include a chunk in results.
# Chunks below this are discarded even if they are top_k.
# 0.25 is conservative — lower means more recall, higher means more precision.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

# RRF constant — standard value, rarely needs tuning
RRF_K = 60

# Candidate multiplier — fetch this many × top_k before re-ranking
CANDIDATE_FACTOR = 3


class DocumentStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="regulations",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready at '{CHROMA_PATH}' | {self.collection.count()} chunks")

        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedding model ready")

        # BM25 index — built in-memory from stored documents on startup
        # Rebuilt automatically after each ingest via _rebuild_bm25()
        self._bm25        = None   # BM25Okapi instance
        self._bm25_ids    = []     # chunk IDs in BM25 index order (for RRF lookup)
        self._bm25_corpus = []     # tokenised corpus parallel to _bm25_ids
        self._rebuild_bm25()

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> str:
        try:
            count = self.collection.count()
            bm25_size = len(self._bm25_ids) if self._bm25 else 0
            return f"ok ({count} chunks, BM25 index: {bm25_size})"
        except Exception as e:
            return f"error: {e}"

    def count_documents(self) -> int:
        return self.collection.count()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_document(
        self, doc_id: str, filename: str, content: bytes, ext: str
    ) -> None:
        """
        Full ingestion pipeline:
          1. Extract text (with page numbers for PDFs)
          2. Paragraph-aware chunking with overlap
          3. Batch embed all chunks
          4. Upsert into ChromaDB
          5. Rebuild BM25 index to include new chunks
        """
        logger.info(f"Ingesting '{filename}' ({len(content):,} bytes)")

        if ext == ".pdf":
            chunk_metas = self._extract_chunks_with_meta(content)
            chunks = [m["text"] for m in chunk_metas]
        else:
            text   = self._extract_text(content, ext)
            chunks = self._chunk_text(text)
            chunk_metas = [{"text": c, "page_number": 0, "page_label": "", "clause_refs": ""} for c in chunks]

        logger.info(f"  → {len(chunks)} chunks extracted")

        if not chunks:
            logger.warning(f"  → No usable text extracted from '{filename}'")
            return

        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()

        now = datetime.utcnow().isoformat()
        ids       = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id":       doc_id,
                "filename":     filename,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "ingested_at":  now,
                "char_count":   len(chunk),
                "page_number":  chunk_metas[i].get("page_number", 0),
                "page_label":   chunk_metas[i].get("page_label", ""),
                "clause_refs":  chunk_metas[i].get("clause_refs", ""),
            }
            for i, chunk in enumerate(chunks)
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(f"  → Indexed: '{filename}' ({len(chunks)} chunks)")

        # Rebuild BM25 index to include new chunks
        self._rebuild_bm25()

    # ── Hybrid Retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: BM25 keyword search + ChromaDB semantic search,
        merged with Reciprocal Rank Fusion (RRF).

        Pipeline:
          1. Fetch CANDIDATE_FACTOR × top_k candidates from ChromaDB (semantic)
          2. Fetch top-k candidates from BM25 (keyword)
          3. Compute RRF score for all candidates
          4. Filter by MIN_SIMILARITY threshold
          5. Return top_k by RRF score

        Falls back to semantic-only if BM25 index is empty.
        """
        total = self.collection.count()
        if total == 0:
            return []

        # ── Semantic search ───────────────────────────────────────────────────
        n_candidates = min(top_k * CANDIDATE_FACTOR, total)
        query_embedding = self.embedder.encode(query).tolist()

        where = {"doc_id": doc_filter} if doc_filter else None
        sem_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        # Build semantic rank map: chunk_id → (rank, similarity, text, meta)
        sem_ranked: Dict[str, Tuple] = {}
        for rank, (text, meta, dist) in enumerate(zip(
            sem_results["documents"][0],
            sem_results["metadatas"][0],
            sem_results["distances"][0],
        )):
            chunk_id   = f"{meta['doc_id']}_chunk_{meta['chunk_index']}"
            similarity = max(0.0, min(1.0, round(1.0 - dist, 4)))
            sem_ranked[chunk_id] = (rank, similarity, text, meta)

        # ── BM25 keyword search ───────────────────────────────────────────────
        bm25_ranked: Dict[str, int] = {}   # chunk_id → rank
        if self._bm25 and self._bm25_ids:
            bm25_candidates = min(top_k * CANDIDATE_FACTOR, len(self._bm25_ids))
            tokens     = _tokenise(query)
            bm25_scores = self._bm25.get_scores(tokens)
            # Get indices sorted by score descending
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:bm25_candidates]
            for rank, idx in enumerate(top_indices):
                if bm25_scores[idx] > 0:   # only include chunks with non-zero BM25 score
                    bm25_ranked[self._bm25_ids[idx]] = rank

        # ── RRF fusion ────────────────────────────────────────────────────────
        # Collect all candidate chunk IDs from both methods
        all_ids = set(sem_ranked.keys()) | set(bm25_ranked.keys())

        scored: List[Dict] = []
        for chunk_id in all_ids:
            sem_rank  = sem_ranked.get(chunk_id, (9999, 0.0, None, None))[0]
            bm25_rank = bm25_ranked.get(chunk_id, 9999)

            rrf_score  = (1.0 / (RRF_K + sem_rank)) + (1.0 / (RRF_K + bm25_rank))
            similarity = sem_ranked.get(chunk_id, (0, 0.0, None, None))[1]

            # If chunk only appears in BM25 (not in semantic candidates),
            # fetch its text and metadata from ChromaDB
            if chunk_id not in sem_ranked:
                try:
                    fetched = self.collection.get(
                        ids=[chunk_id],
                        include=["documents", "metadatas"],
                    )
                    if fetched["documents"]:
                        text = fetched["documents"][0]
                        meta = fetched["metadatas"][0]
                        # Compute actual similarity for this chunk
                        chunk_emb = self.embedder.encode(text).tolist()
                        from numpy import dot
                        from numpy.linalg import norm
                        q_emb = query_embedding
                        cos   = dot(q_emb, chunk_emb) / (norm(q_emb) * norm(chunk_emb) + 1e-8)
                        similarity = max(0.0, min(1.0, round(float(cos), 4)))
                    else:
                        continue
                except Exception:
                    continue
            else:
                text = sem_ranked[chunk_id][2]
                meta = sem_ranked[chunk_id][3]

            # Apply minimum similarity threshold
            if similarity < MIN_SIMILARITY and bm25_rank == 9999:
                # Only discard if BOTH methods scored it poorly
                # (BM25 hit overrides low semantic similarity — exact keyword match wins)
                continue

            scored.append({
                "text":          text,
                "filename":      meta.get("filename", "unknown"),
                "doc_id":        meta.get("doc_id"),
                "chunk_index":   meta.get("chunk_index"),
                "total_chunks":  meta.get("total_chunks"),
                "page_number":   meta.get("page_number", 0),
                "page_label":    meta.get("page_label", ""),
                "clause_refs":   meta.get("clause_refs", ""),
                "similarity":    similarity,
                "relevance_pct": round(similarity * 100, 1),
                "rrf_score":     round(rrf_score, 6),
                "bm25_hit":      chunk_id in bm25_ranked,
            })

        # Sort by RRF score and return top_k
        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        results = scored[:top_k]

        logger.info(
            f"Hybrid retrieve '{query[:40]}' → "
            f"{len(sem_ranked)} semantic, {len(bm25_ranked)} BM25 hits, "
            f"{len(results)} returned (threshold={MIN_SIMILARITY})"
        )
        return results

    def debug_retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Diagnostic retrieval — returns full scoring breakdown.
        Use GET /debug/retrieve?q=keyword to diagnose missed results.

        Shows:
          - Raw semantic scores for all candidates
          - BM25 scores for top candidates
          - RRF fusion score
          - Whether MIN_SIMILARITY threshold would filter it
          - Chunk text preview
        """
        total = self.collection.count()
        if total == 0:
            return {"error": "No documents indexed", "query": query}

        n_candidates = min(top_k * CANDIDATE_FACTOR, total)
        query_embedding = self.embedder.encode(query).tolist()

        sem_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        # BM25 scores
        bm25_scores_map: Dict[str, float] = {}
        if self._bm25 and self._bm25_ids:
            tokens      = _tokenise(query)
            raw_scores  = self._bm25.get_scores(tokens)
            for idx, chunk_id in enumerate(self._bm25_ids):
                if raw_scores[idx] > 0:
                    bm25_scores_map[chunk_id] = round(float(raw_scores[idx]), 4)

        candidates = []
        for rank, (text, meta, dist) in enumerate(zip(
            sem_results["documents"][0],
            sem_results["metadatas"][0],
            sem_results["distances"][0],
        )):
            chunk_id   = f"{meta['doc_id']}_chunk_{meta['chunk_index']}"
            similarity = max(0.0, min(1.0, round(1.0 - dist, 4)))
            bm25_score = bm25_scores_map.get(chunk_id, 0.0)
            bm25_rank  = (
                sorted(bm25_scores_map.values(), reverse=True).index(bm25_score)
                if bm25_score > 0 else 9999
            )
            rrf = round(
                (1.0 / (RRF_K + rank)) + (1.0 / (RRF_K + bm25_rank)), 6
            )
            above_threshold = similarity >= MIN_SIMILARITY or bm25_score > 0

            candidates.append({
                "rank_semantic":    rank + 1,
                "chunk_id":         chunk_id,
                "filename":         meta.get("filename"),
                "chunk_index":      meta.get("chunk_index"),
                "page_number":      meta.get("page_number", 0),
                "similarity":       similarity,
                "bm25_score":       bm25_score,
                "rrf_score":        rrf,
                "above_threshold":  above_threshold,
                "bm25_hit":         bm25_score > 0,
                "preview":          text[:200].replace("\n", " ") + "…",
            })

        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)

        return {
            "query":            query,
            "total_chunks_in_db": total,
            "bm25_index_size":  len(self._bm25_ids),
            "min_similarity":   MIN_SIMILARITY,
            "candidates_fetched": len(candidates),
            "candidates":       candidates,
            "config": {
                "CHUNK_SIZE":      CHUNK_SIZE,
                "CHUNK_OVERLAP":   CHUNK_OVERLAP,
                "MIN_SIMILARITY":  MIN_SIMILARITY,
                "EMBED_MODEL":     EMBED_MODEL,
                "RRF_K":           RRF_K,
                "CANDIDATE_FACTOR": CANDIDATE_FACTOR,
            },
        }

    # ── Document management ───────────────────────────────────────────────────

    def list_documents(self) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        seen: Dict[str, Dict] = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id":      doc_id,
                    "filename":    meta.get("filename"),
                    "chunks":      meta.get("total_chunks"),
                    "ingested_at": meta.get("ingested_at"),
                }
        return list(seen.values())

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted document: {doc_id}")
        self._rebuild_bm25()   # rebuild index after deletion

    # ── BM25 index management ─────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        """
        Load all stored documents from ChromaDB and build an in-memory BM25 index.
        Called on startup, after every ingest, and after every delete.
        BM25Okapi from rank-bm25 is pure Python — no build tools required.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank-bm25 not installed. BM25 keyword search disabled. "
                "Run: pip install rank-bm25"
            )
            self._bm25     = None
            self._bm25_ids = []
            return

        total = self.collection.count()
        if total == 0:
            self._bm25     = None
            self._bm25_ids = []
            logger.info("BM25: collection empty, index skipped")
            return

        try:
            # Fetch all documents (text + IDs) from ChromaDB
            all_data = self.collection.get(include=["documents", "metadatas"])
            texts    = all_data["documents"]
            metas    = all_data["metadatas"]

            # Build parallel arrays: chunk_ids ↔ tokenised corpus
            ids    = []
            corpus = []
            for text, meta in zip(texts, metas):
                chunk_id = f"{meta['doc_id']}_chunk_{meta['chunk_index']}"
                ids.append(chunk_id)
                corpus.append(_tokenise(text))

            self._bm25_ids    = ids
            self._bm25_corpus = corpus
            self._bm25        = BM25Okapi(corpus)
            logger.info(f"BM25 index rebuilt: {len(ids)} chunks")
        except Exception as e:
            logger.error(f"BM25 index build failed: {e}", exc_info=True)
            self._bm25     = None
            self._bm25_ids = []

    # ── Text extraction helpers ───────────────────────────────────────────────

    def _extract_text(self, content: bytes, ext: str) -> str:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext in (".txt", ".md"):
            return content.decode("utf-8", errors="replace")
        raise ValueError(f"Unsupported extension: {ext}")

    def _extract_chunks_with_meta(self, content: bytes) -> List[Dict]:
        """
        PDF-specific extractor that preserves page numbers and detects clause references.
        Returns list of {text, page_number, page_label, clause_refs} dicts per chunk.
        """
        from pypdf import PdfReader

        CLAUSE_PATTERN = re.compile(
            r"\b(Article\s+\d+[\w.]*|Section\s+\d+[\w.]*|Clause\s+\d+[\w.]*"
            r"|Para(?:graph)?\s+\d+[\w.]*|Rule\s+\d+[\w.]*"
            r"|Schedule\s+\d+[\w.]*|Annex\s+\d+[\w.]*"
            r"|\d{1,2}\.\d{1,2}(?:\.\d{1,2})*)",
            re.IGNORECASE,
        )

        reader = PdfReader(io.BytesIO(content))
        page_texts = []
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            if raw.strip():
                page_texts.append((page_num, raw))

        all_chunks: List[Dict] = []
        for page_num, page_text in page_texts:
            chunks = self._chunk_text(page_text)
            for chunk in chunks:
                clauses = CLAUSE_PATTERN.findall(chunk)
                unique_clauses = list(dict.fromkeys(clauses))
                all_chunks.append({
                    "text":        chunk,
                    "page_number": page_num,
                    "page_label":  f"Page {page_num}",
                    "clause_refs": ", ".join(unique_clauses[:5]),
                })

        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraphs: List[str] = []
        for para in raw_paras:
            if len(para) <= MAX_PARA_CHARS:
                paragraphs.append(para)
            else:
                for i in range(0, len(para), MAX_PARA_CHARS - CHUNK_OVERLAP):
                    sub = para[i: i + MAX_PARA_CHARS].strip()
                    if sub:
                        paragraphs.append(sub)

        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                overlap_seed = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
                current = (overlap_seed + "\n\n" + para).strip() if overlap_seed else para

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c.replace(" ", "").replace("\n", "")) > 50]


# ── Module-level tokeniser ────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokeniser for BM25.
    Lowercases and strips punctuation. Preserves regulatory terms like
    "CET1", "LCR", "Basel" as single tokens — doesn't split on numbers.
    """
    text = text.lower()
    # Split on whitespace and punctuation except hyphens (keep "risk-weighted")
    tokens = re.split(r"[^\w\-]+", text)
    # Filter very short tokens and pure punctuation
    return [t for t in tokens if len(t) >= 2]
