"""
Regulation Analyser & Assistant - Main FastAPI Application v1.3
Multi-Agent System with RAG, SHAP Explainability, Document References,
SQLite Session History, and Chat Mode with conversation memory.

Chat mode (new in v1.3):
  - POST /query now fetches the last 10 turns for the session from SQLite
  - Formatted history is passed to the orchestrator's Synthesiser agent
  - The LLM can resolve follow-up questions, pronouns, and implicit references
  - Frontend sends the same session_id across turns to maintain continuity
  - New endpoint: POST /chat/new → generates a fresh session_id

Routes:
  POST   /query                        → 5-agent pipeline + chat memory + saves to SQLite
  POST   /chat/new                     → start a new chat session (returns session_id)
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
    init_db, save_message, get_recent_turns_for_prompt,
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
    description="On-premise multi-agent AI for regulatory analysis with chat memory",
    version="1.3.0",
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

# Initialise SQLite session database
init_db()

# In-memory indexing status: {doc_id: "indexing" | "ready" | "error:<msg>"}
_index_status: dict = {}

# Max conversation turns to feed back to LLM
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "10"))

# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None   # if provided, history is loaded and used
    mode: Optional[str] = "full"       # full | quick | compare
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
    turn_number: int = 1               # which turn in this session this is

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

# ── Core routes ───────────────────────────────────────────────────────────────

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
    """
    Start a new chat session.
    Returns a fresh session_id the frontend should reuse for all turns
    in this conversation.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"New chat session: {session_id[:8]}")
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
    MAX_MB = 50
    if len(contents) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_MB}MB limit.")

    doc_id = str(uuid.uuid4())
    _index_status[doc_id] = "indexing"
    background_tasks.add_task(
        _ingest_with_status,
        doc_id=doc_id, filename=file.filename, content=contents, ext=ext,
    )
    logger.info(f"Upload queued: {file.filename} | doc_id={doc_id}")
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
    """
    Main analysis endpoint with chat memory.

    If session_id is provided and prior messages exist for that session,
    the last CHAT_HISTORY_TURNS turns are fetched from SQLite and injected
    into the Synthesiser prompt so the LLM can handle follow-up questions.

    Turn number is returned so the frontend can label messages chronologically.
    """
    # Use existing session_id (chat continuation) or create new one
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"Query [{session_id[:8]}]: {req.query[:80]}...")

    try:
        # ── Fetch conversation history from SQLite ────────────────────────────
        conversation_history = get_recent_turns_for_prompt(
            session_id=session_id,
            n=CHAT_HISTORY_TURNS,
        )
        turn_number = conversation_history.count("User:") + 1
        logger.info(
            f"Session {session_id[:8]} — turn {turn_number} "
            f"({'with history' if conversation_history else 'fresh session'})"
        )

        # ── Run 5-agent pipeline with history ────────────────────────────────
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

        # ── Save to SQLite ────────────────────────────────────────────────────
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
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters.")
    return search_history(keyword=q.strip(), limit=limit)

@app.get("/history/messages/{message_id}")
async def get_message(message_id: int):
    msg = get_message_by_id(message_id)
    if not msg:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found.")
    return msg
