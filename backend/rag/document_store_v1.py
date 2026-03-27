"""
Document Store — ChromaDB-backed RAG layer

Handles ingestion, chunking, batch embedding, and retrieval of regulatory documents.

Design decisions:
  - SentenceTransformers all-MiniLM-L6-v2: free, local, 384-dim embeddings
  - Chunk size 800 chars with 150-char overlap to preserve context at boundaries
  - Hard cap on individual paragraph size prevents unbounded chunks
  - Batch encoding: all chunks embedded in one SentenceTransformer call (much faster)
  - _doc_registry is rebuilt from ChromaDB on startup — survives server restarts
  - retrieve() guards against empty collection to prevent ChromaDB ValueError
  - Cosine similarity space in ChromaDB: distance 0 = identical, 2 = opposite
    → similarity = 1 - distance (clamped to [0, 1])
"""
import io
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_PARA_CHARS = 800   # hard cap: paragraphs larger than this are split further


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
        logger.info(
            f"ChromaDB initialised at '{CHROMA_PATH}' | {self.collection.count()} chunks")

        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedding model ready")

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> str:
        try:
            self.collection.count()
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def count_documents(self) -> int:
        """Number of chunks in the collection (not unique documents)."""
        return self.collection.count()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_document(
        self, doc_id: str, filename: str, content: bytes, ext: str
    ) -> None:
        logger.info(f"Ingesting '{filename}' ({len(content):,} bytes)")

        # Returns list of dicts: {text, page_number, clause_refs}
        chunks_with_meta = self._extract_chunks_with_meta(content, ext)
        logger.info(f"  → {len(chunks_with_meta)} chunks")

        if not chunks_with_meta:
            logger.warning(f"  → No usable text extracted from '{filename}'")
            return

        texts = [c["text"] for c in chunks_with_meta]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        now = datetime.utcnow().isoformat()
        total = len(chunks_with_meta)
        ids = [f"{doc_id}_chunk_{i}" for i in range(total)]
        metadatas = [
            {
                "doc_id":       doc_id,
                "filename":     filename,
                "chunk_index":  i,
                "total_chunks": total,
                "ingested_at":  now,
                "char_count":   len(c["text"]),
                "page_number":  c.get("page_number", 0),
                "page_label":   c.get("page_label", ""),
                "clause_refs":  c.get("clause_refs", ""),
            }
            for i, c in enumerate(chunks_with_meta)
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(f"  → Indexed: '{filename}' ({total} chunks)")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search via ChromaDB cosine similarity.

        BUG FIX: ChromaDB raises ValueError if n_results > collection size or = 0.
        We clamp n_results to the available count before querying.

        Returns chunks sorted by similarity descending, each with:
          - text, filename, doc_id, chunk_index, total_chunks
          - similarity (0–1), relevance_pct (0–100)
        """
        total = self.collection.count()
        if total == 0:
            return []

        n_results = min(top_k, total)
        query_embedding = self.embedder.encode(query).tolist()

        where = {"doc_id": doc_filter} if doc_filter else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        chunks = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine distance in ChromaDB ∈ [0, 2]; clamp similarity to [0, 1]
            similarity = max(0.0, min(1.0, round(1.0 - dist, 4)))
            chunks.append({
                "text":         text,
                "filename":     meta.get("filename", "unknown"),
                "doc_id":       meta.get("doc_id"),
                "chunk_index":  meta.get("chunk_index"),
                "total_chunks": meta.get("total_chunks"),
                "similarity":   similarity,
                "relevance_pct": round(similarity * 100, 1),
                "page_number":  meta.get("page_number", 0),
                "page_label":   meta.get("page_label", ""),
                "clause_refs":  meta.get("clause_refs", ""),
            })

        chunks.sort(key=lambda x: x["similarity"], reverse=True)
        return chunks

    # ── Document management ───────────────────────────────────────────────────

    def list_documents(self) -> List[Dict]:
        """
        Returns one entry per unique document.
        Rebuilt from ChromaDB metadata on every call — survives server restarts.
        """
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
        """Delete all chunks belonging to a document from ChromaDB."""
        self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted document: {doc_id}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_chunks_with_meta(self, content: bytes, ext: str) -> list:
        """
        Extract text chunks WITH page numbers and clause references.

        Returns list of:
        {
            text:        str,
            page_number: int,   # 1-based page number
            page_label:  str,   # e.g. "Page 12" or "xii" for roman numerals
            clause_refs: str,   # detected clause/article/section refs e.g. "Article 5, Section 3.2"
        }
        """
        import re

        def extract_clause_refs(text: str) -> str:
            """
            Detect regulatory clause references in text.
            Covers: Article 5, Section 3.2, Clause 4(b), Para 12, Rule 7, Reg 14
            """
            patterns = [
                r"Article\s+\d+[\w\.\(\)]*",
                r"Section\s+\d+[\w\.\(\)]*",
                r"Clause\s+\d+[\w\.\(\)]*",
                r"Para(?:graph)?\s+\d+[\w\.\(\)]*",
                r"Rule\s+\d+[\w\.\(\)]*",
                r"Regulation\s+\d+[\w\.\(\)]*",
                r"Schedule\s+\d+[\w\.\(\)]*",
                r"Annex\s+\w+",
                r"\d+\.\d+(?:\.\d+)*",   # numbered clauses like 3.2.1
            ]
            found = []
            for pat in patterns:
                matches = re.findall(pat, text, re.IGNORECASE)
                found.extend(matches[:3])   # max 3 per pattern to avoid noise
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for f in found:
                if f.lower() not in seen:
                    seen.add(f.lower())
                    unique.append(f)
            return ", ".join(unique[:8])   # max 8 refs per chunk

        # ── PDF: extract page-by-page ─────────────────────────────────────────
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            # Build page-tagged chunks
            all_chunks = []
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                page_num = page_idx + 1
                page_label = f"Page {page_num}"

                # Chunk this page's text
                sub_chunks = self._chunk_text(page_text)
                for chunk in sub_chunks:
                    all_chunks.append({
                        "text":        chunk,
                        "page_number": page_num,
                        "page_label":  page_label,
                        "clause_refs": extract_clause_refs(chunk),
                    })
            return all_chunks

        # ── DOCX: extract paragraph-by-paragraph with heading detection ───────
        elif ext == ".docx":
            from docx import Document
            doc = Document(io.BytesIO(content))
            full_text = []
            para_num = 0
            for para in doc.paragraphs:
                if not para.text.strip():
                    continue
                para_num += 1
                full_text.append(para.text)

            # Chunk the full text
            combined = "\n\n".join(full_text)
            raw_chunks = self._chunk_text(combined)
            return [
                {
                    "text":        chunk,
                    "page_number": 0,
                    "page_label":  "See document",
                    "clause_refs": extract_clause_refs(chunk),
                }
                for chunk in raw_chunks
            ]

        # ── TXT / MD ──────────────────────────────────────────────────────────
        elif ext in (".txt", ".md"):
            text = content.decode("utf-8", errors="replace")
            raw_chunks = self._chunk_text(text)
            return [
                {
                    "text":        chunk,
                    "page_number": 0,
                    "page_label":  "See document",
                    "clause_refs": extract_clause_refs(chunk),
                }
                for chunk in raw_chunks
            ]

        raise ValueError(f"Unsupported extension: {ext}")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Paragraph-aware sliding-window chunker.

        Algorithm:
          1. Split on blank lines to get paragraphs
          2. Hard-split any paragraph exceeding MAX_PARA_CHARS (handles PDFs
             with no blank lines that would otherwise produce one giant chunk)
          3. Accumulate paragraphs into a chunk until CHUNK_SIZE is reached
          4. Start next chunk with CHUNK_OVERLAP chars from end of previous

        Returns only chunks with >50 non-whitespace characters.
        """
        # Step 1+2: split into paragraphs, hard-cap each one
        raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraphs: List[str] = []
        for para in raw_paras:
            if len(para) <= MAX_PARA_CHARS:
                paragraphs.append(para)
            else:
                # Hard-split long paragraphs at sentence boundaries if possible
                for i in range(0, len(para), MAX_PARA_CHARS - CHUNK_OVERLAP):
                    sub = para[i: i + MAX_PARA_CHARS].strip()
                    if sub:
                        paragraphs.append(sub)

        # Step 3+4: sliding window accumulation
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Seed next chunk with overlap tail from previous
                overlap_seed = current[-CHUNK_OVERLAP:] if len(
                    current) > CHUNK_OVERLAP else current
                current = (overlap_seed + "\n\n" +
                           para).strip() if overlap_seed else para

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c.replace(" ", "").replace("\n", "")) > 50]
