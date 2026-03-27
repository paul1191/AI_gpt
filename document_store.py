"""
Document Store — ChromaDB-backed RAG layer
Handles ingestion, chunking, embedding, and retrieval of regulatory documents.
"""
import os
import io
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")   # free local model
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


class DocumentStore:
    """
    Manages the ChromaDB vector store for regulatory documents.
    
    Design decisions (explainability):
    - SentenceTransformers all-MiniLM-L6-v2 for embeddings (free, runs locally)
    - Chunk size 800 tokens with 150 overlap to preserve context at boundaries
    - Stores metadata: source filename, page, chunk index, ingested_at
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="regulations",
            metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
        )
        logger.info(f"ChromaDB initialised at {CHROMA_PATH}")

        # Load embedding model once — cached in memory
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedding model ready")

        # In-memory doc registry (doc_id → metadata)
        self._doc_registry: Dict[str, Dict] = {}

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> str:
        try:
            self.collection.count()
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def count_documents(self) -> int:
        return self.collection.count()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_document(self, doc_id: str, filename: str, content: bytes, ext: str):
        """Parse, chunk, embed and store a document."""
        logger.info(f"Ingesting: {filename} ({len(content)} bytes)")

        text = self._extract_text(content, ext, filename)
        chunks = self._chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info(f"  → {len(chunks)} chunks created")

        ids, documents, metadatas, embeddings = [], [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embedder.encode(chunk).tolist()

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": datetime.utcnow().isoformat(),
                "char_count": len(chunk),
            })
            embeddings.append(embedding)

        # Batch upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._doc_registry[doc_id] = {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}
        logger.info(f"  → Indexed successfully: {filename}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5, doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Semantic search against ChromaDB.
        Returns ranked chunks with similarity scores and source metadata.
        
        Explainability note: we return raw similarity scores so the UI and
        SHAP analyser can reason about relevance.
        """
        query_embedding = self.embedder.encode(query).tolist()

        where = {"doc_id": doc_filter} if doc_filter else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        chunks = []
        docs   = results["documents"][0]
        metas  = results["metadatas"][0]
        dists  = results["distances"][0]

        for text, meta, dist in zip(docs, metas, dists):
            similarity = round(1 - dist, 4)   # cosine distance → similarity
            chunks.append({
                "text": text,
                "filename": meta.get("filename", "unknown"),
                "doc_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index"),
                "total_chunks": meta.get("total_chunks"),
                "similarity": similarity,
                "relevance_pct": round(similarity * 100, 1),
            })

        # Sort by similarity descending
        chunks.sort(key=lambda x: x["similarity"], reverse=True)
        return chunks

    # ── Document management ───────────────────────────────────────────────────

    def list_documents(self) -> List[Dict]:
        """List unique documents from the collection metadata."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        seen = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename"),
                    "chunks": meta.get("total_chunks"),
                    "ingested_at": meta.get("ingested_at"),
                }
        return list(seen.values())

    def delete_document(self, doc_id: str):
        """Delete all chunks belonging to a document."""
        self.collection.delete(where={"doc_id": doc_id})
        self._doc_registry.pop(doc_id, None)
        logger.info(f"Deleted document: {doc_id}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, content: bytes, ext: str, filename: str) -> str:
        """Extract raw text from PDF, DOCX, TXT, or MD."""
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages)
        elif ext == ".docx":
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        elif ext in (".txt", ".md"):
            return content.decode("utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Sliding-window character chunker.
        Respects paragraph boundaries where possible.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current += ("\n\n" + para) if current else para
            else:
                if current:
                    chunks.append(current)
                # Start new chunk with overlap from previous
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = overlap_text + "\n\n" + para if overlap_text else para

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c.strip()) > 50]  # skip trivially short chunks
