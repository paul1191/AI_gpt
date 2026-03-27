"""
Regulation Analyser & Assistant - Main FastAPI Application v1.4
Multi-Agent System with RAG, SHAP Explainability, Document References,
SQLite Session History, Chat Mode, and Hybrid Retrieval.

New in v1.4:
  GET /debug/retrieve?q=keyword  → full scoring breakdown for retrieval diagnosis

All other routes unchanged from v1.3.
"""
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from agents.orchestrator import OrchestratorAgent
from rag.document_store import DocumentStore
from explainability.shap_explainer import SHAPExplainer
from db.session_store import (
    init_db, save_message, get_recent_turns_for_prompt,
    get_recent_messages, get_all_sessions, get_session_history,
    search_history, get_message_by_id, delete_session, get_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Regulation Analyser API",
    description="On-premise multi-agent AI for regulatory analysis — hybrid retrieval + chat memory",
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

doc_store      = DocumentStore()
orchestrator   = OrchestratorAgent(doc_store)
shap_explainer = SHAPExplainer()

init_db()

_index_status: dict = {}

CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "10"))

# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: Optional[str] = "full"
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
    message_id: Optional[int] = None
    turn_number: int = 1

# ── Background ingest ─────────────────────────────────────────────────────────

def _ingest_with_status(doc_id: str, filename: str, content: bytes, ext: str):
    try:
        _index_status[doc_id] = "indexing"
        doc_store.ingest_document(doc_id=doc_id, filename=filename, content=content, ext=ext)
        _index_status[doc_id] = "ready"
        logger.info(f"Indexing complete: {filename} ({doc_id})")
    except Exception as e:
        _index_status[doc_id] = f"error:{e}"
        logger.error(f"Indexing failed for {filename}: {e}", exc_info=True)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    stats = get_stats()
    return {
        "api":               "ok",
        "chroma":            doc_store.health_check(),
        "ollama":            orchestrator.llm_health_check(),
        "documents_indexed": doc_store.count_documents(),
        "saved_queries":     stats.get("total_messages", 0),
        "sessions":          stats.get("total_sessions", 0),
    }


@app.post("/chat/new")
async def new_chat_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    allowed = {".pdf", ".txt", ".docx", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
        )
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50MB limit.")

    doc_id = str(uuid.uuid4())
    _index_status[doc_id] = "indexing"
    background_tasks.add_task(
        _ingest_with_status,
        doc_id=doc_id, filename=file.filename, content=contents, ext=ext,
    )
    return {"doc_id": doc_id, "filename": file.filename, "status": "indexing"}


@app.get("/documents/{doc_id}/status")
async def document_status(doc_id: str):
    return {"doc_id": doc_id, "status": _index_status.get(doc_id, "unknown")}


@app.get("/documents")
async def list_documents():
    docs = doc_store.list_documents()
    for doc in docs:
        doc["index_status"] = _index_status.get(doc["doc_id"], "ready")
    return docs


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    doc_store.delete_document(doc_id)
    _index_status.pop(doc_id, None)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"Query [{session_id[:8]}]: {req.query[:80]}...")

    try:
        conversation_history = get_recent_turns_for_prompt(
            session_id=session_id,
            n=CHAT_HISTORY_TURNS,
        )
        turn_number = conversation_history.count("User:") + 1

        result = await orchestrator.run(
            query=req.query,
            session_id=session_id,
            mode=req.mode,
            top_k=req.top_k,
            conversation_history=conversation_history,
        )

        shap_data = shap_explainer.analyse(
            query=req.query,
            context_chunks=[r["text"] for r in result["references"]],
            answer=result["answer"],
        )

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
            turn_number=turn_number,
        )
    except Exception as e:
        logger.error(f"Query failed [{session_id[:8]}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")


@app.get("/query/stream")
async def query_stream(query: str, session_id: Optional[str] = None):
    session_id = session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            async for token in orchestrator.stream(query=query, session_id=session_id):
                yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Debug endpoint ────────────────────────────────────────────────────────────

@app.get("/debug/retrieve")
async def debug_retrieve(
    q: str = Query(..., description="Keyword or phrase to test retrieval for"),
    top_k: int = Query(10, description="Number of candidates to show"),
):
    """
    Diagnostic endpoint — shows full retrieval scoring for a query.

    Use this when a keyword you know is in a document isn't being found.
    The response shows:
      - semantic similarity score for each candidate chunk
      - BM25 keyword score
      - RRF fusion score
      - whether the chunk passes the MIN_SIMILARITY threshold
      - a text preview of each chunk

    Example: GET /debug/retrieve?q=CET1+capital+ratio&top_k=10

    Read the results in Swagger UI: http://localhost:8000/docs
    """
    if not q or len(q.strip()) < 1:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")
    return doc_store.debug_retrieve(query=q.strip(), top_k=top_k)


# ── History routes ────────────────────────────────────────────────────────────

@app.get("/history")
async def get_history(limit: int = 20):
    return get_recent_messages(limit=limit)

@app.get("/history/stats")
async def history_stats():
    return get_stats()

@app.get("/history/sessions")
async def list_sessions():
    return get_all_sessions()

@app.get("/history/sessions/{session_id}")
async def get_session(session_id: str):
    return get_session_history(session_id)

@app.delete("/history/sessions/{session_id}")
async def remove_session(session_id: str):
    deleted = delete_session(session_id)
    return {"status": "deleted", "session_id": session_id, "messages_deleted": deleted}

@app.get("/history/search")
async def search_endpoint(q: str, limit: int = 20):
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters.")
    return search_history(keyword=q.strip(), limit=limit)

@app.get("/history/messages/{message_id}")
async def get_message(message_id: int):
    msg = get_message_by_id(message_id)
    if not msg:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found.")
    return msg
