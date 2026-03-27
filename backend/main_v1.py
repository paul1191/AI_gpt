"""
Regulation Analyser & Assistant - Main FastAPI Application
Multi-Agent System with RAG, SHAP Explainability, Document References
and SQLite-backed Session History.

Routes:
  POST   /query                        → 5-agent pipeline + saves to SQLite
  POST   /documents/upload             → background ingest into ChromaDB
  GET    /documents                    → list indexed documents
  GET    /documents/{id}/status        → poll indexing status
  DELETE /documents/{id}               → remove from index
  GET    /health                       → Ollama + ChromaDB + DB liveness
  GET    /query/stream                 → SSE streaming answer
  GET    /history                      → recent messages across all sessions
  GET    /history/sessions             → all sessions with stats
  GET    /history/sessions/{id}        → full history for one session
  GET    /history/search?q=keyword     → full-text search past queries
  GET    /history/messages/{id}        → single message by id
  GET    /history/stats                → aggregate DB statistics
  DELETE /history/sessions/{id}        → delete a session
"""
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from agents.orchestrator import OrchestratorAgent
from rag.document_store import DocumentStore
from explainability.shap_explainer import SHAPExplainer
from db.session_store import (
    init_db, save_message,
    get_recent_messages, get_all_sessions, get_session_history,
    search_history, get_message_by_id, delete_session, get_stats,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Regulation Analyser API",
    description="On-premise multi-agent AI for regulatory document analysis with full explainability",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────
doc_store      = DocumentStore()
orchestrator   = OrchestratorAgent(doc_store)
shap_explainer = SHAPExplainer()

# Initialise SQLite session database (creates file + tables if not present)
init_db()

# In-memory indexing status: {doc_id: "indexing" | "ready" | "error:<msg>"}
_index_status: dict = {}

# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: Optional[str] = "full"   # full | quick | compare
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    agent_trace: list
    references: list
    shap_analysis: dict
    confidence: float
    mode: str
    message_id: Optional[int] = None   # DB row id for future retrieval

# ── Background ingest ─────────────────────────────────────────────────────────
def _ingest_with_status(doc_id: str, filename: str, content: bytes, ext: str):
    """Wraps ingest so upload status is trackable via /documents/{doc_id}/status."""
    try:
        _index_status[doc_id] = "indexing"
        doc_store.ingest_document(doc_id=doc_id, filename=filename, content=content, ext=ext)
        _index_status[doc_id] = "ready"
        logger.info(f"Indexing complete: {filename} ({doc_id})")
    except Exception as e:
        _index_status[doc_id] = f"error:{e}"
        logger.error(f"Indexing failed for {filename}: {e}", exc_info=True)

# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness probe — checks Ollama, ChromaDB, and SQLite connectivity."""
    stats = get_stats()
    return {
        "api":               "ok",
        "chroma":            doc_store.health_check(),
        "ollama":            orchestrator.llm_health_check(),
        "documents_indexed": doc_store.count_documents(),
        "saved_queries":     stats.get("total_messages", 0),
        "sessions":          stats.get("total_sessions", 0),
    }


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and asynchronously index a regulatory document into ChromaDB.
    Poll GET /documents/{doc_id}/status to know when indexing is complete.
    Supported formats: PDF, DOCX, TXT, MD. Max size: 50MB.
    """
    allowed = {".pdf", ".txt", ".docx", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
        )

    contents = await file.read()
    MAX_MB = 50
    if len(contents) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_MB}MB limit.")

    doc_id = str(uuid.uuid4())
    _index_status[doc_id] = "indexing"

    background_tasks.add_task(
        _ingest_with_status,
        doc_id=doc_id,
        filename=file.filename,
        content=contents,
        ext=ext,
    )
    logger.info(f"Upload queued: {file.filename} | doc_id={doc_id}")
    return {"doc_id": doc_id, "filename": file.filename, "status": "indexing"}


@app.get("/documents/{doc_id}/status")
async def document_status(doc_id: str):
    """Poll indexing progress for a recently uploaded document."""
    return {"doc_id": doc_id, "status": _index_status.get(doc_id, "unknown")}


@app.get("/documents")
async def list_documents():
    """List all documents currently indexed in ChromaDB."""
    docs = doc_store.list_documents()
    for doc in docs:
        doc["index_status"] = _index_status.get(doc["doc_id"], "ready")
    return docs


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document and all its chunks from ChromaDB."""
    doc_store.delete_document(doc_id)
    _index_status.pop(doc_id, None)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Main analysis endpoint.

    5-agent pipeline:
      1. QueryAnalyser  — intent classification + entity extraction (LLM)
      2. Retriever      — ChromaDB semantic search (SentenceTransformers)
      3. Synthesiser    — grounded answer with [N] inline citations (LLM)
      4. Critic         — confidence scoring + hallucination detection (LLM)
      5. Formatter      — structured output + reference list (rule-based)

    Plus:
      - SHAP analysis: TF-IDF + LogisticRegression + SHAP LinearExplainer
      - SQLite persistence: every query + response saved to sessions.db
    """
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"Query [{session_id[:8]}]: {req.query[:80]}...")

    try:
        result = await orchestrator.run(
            query=req.query,
            session_id=session_id,
            mode=req.mode,
            top_k=req.top_k,
        )

        shap_data = shap_explainer.analyse(
            query=req.query,
            context_chunks=[r["text"] for r in result["references"]],
            answer=result["answer"],
        )

        # ── Persist to SQLite ─────────────────────────────────────────────────
        message_id = save_message(
            session_id=session_id,
            query=req.query,
            answer=result["answer"],
            confidence=result["confidence"],
            mode=req.mode,
            references=result["references"],
            agent_trace=result["agent_trace"],
            shap_analysis=shap_data,
        )

        return QueryResponse(
            session_id=session_id,
            query=req.query,
            answer=result["answer"],
            agent_trace=result["agent_trace"],
            references=result["references"],
            shap_analysis=shap_data,
            confidence=result["confidence"],
            mode=req.mode,
            message_id=message_id,
        )
    except Exception as e:
        logger.error(f"Query failed [{session_id[:8]}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")


@app.get("/query/stream")
async def query_stream(query: str, session_id: Optional[str] = None):
    """Server-Sent Events streaming endpoint. Yields Ollama tokens in real time."""
    session_id = session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            async for token in orchestrator.stream(query=query, session_id=session_id):
                safe = token.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── History routes ────────────────────────────────────────────────────────────

@app.get("/history")
async def get_history(limit: int = 20):
    """Get the most recent queries across all sessions."""
    return get_recent_messages(limit=limit)


@app.get("/history/stats")
async def history_stats():
    """Aggregate statistics: total queries, sessions, avg confidence, date range."""
    return get_stats()


@app.get("/history/sessions")
async def list_sessions():
    """List all sessions with message count, avg confidence, and last activity."""
    return get_all_sessions()


@app.get("/history/sessions/{session_id}")
async def get_session(session_id: str):
    """Get the full conversation history for a specific session."""
    return get_session_history(session_id)


@app.delete("/history/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete all messages belonging to a session."""
    deleted = delete_session(session_id)
    return {"status": "deleted", "session_id": session_id, "messages_deleted": deleted}


@app.get("/history/search")
async def search_endpoint(q: str, limit: int = 20):
    """Full-text search across all stored queries and answers."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters.")
    return search_history(keyword=q.strip(), limit=limit)


@app.get("/history/messages/{message_id}")
async def get_message(message_id: int):
    """Retrieve a single saved message by its integer ID."""
    msg = get_message_by_id(message_id)
    if not msg:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found.")
    return msg
