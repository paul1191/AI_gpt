"""
Regulation Analyser & Assistant - Main FastAPI Application v1.5
Multi-Agent System with Hybrid Retrieval, Chat Memory, and
Multi-Framework Explainability (SHAP + RAG Faithfulness + LIME + Trust Score).

New in v1.5:
  - SHAP runs first (before orchestrator) so shap_data can be passed to the
    Critic/TrustScoreBuilder for query alignment scoring
  - QueryResponse adds: trust_score, trust_level, trust_data,
    faithfulness_data, lime_data
  - Critic is now fully algorithmic — no LLM call for confidence scoring
"""
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

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
    description="On-premise multi-agent regulatory AI — SHAP + Faithfulness + LIME + Trust Score",
    version="1.5.0",
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
    query:      str
    session_id: Optional[str] = None
    mode:       Optional[str] = "full"
    top_k:      Optional[int] = 5

class QueryResponse(BaseModel):
    session_id:        str
    query:             str
    answer:            str
    agent_trace:       list
    references:        list
    shap_analysis:     dict
    faithfulness_data: dict
    lime_data:         dict
    trust_data:        dict
    confidence:        float   # = trust_score (kept for backwards compat)
    trust_score:       float
    trust_level:       str
    mode:              str
    message_id:        Optional[int] = None
    turn_number:       int = 1

# ── Background ingest ─────────────────────────────────────────────────────────

def _ingest_with_status(doc_id: str, filename: str, content: bytes, ext: str):
    try:
        _index_status[doc_id] = "indexing"
        doc_store.ingest_document(doc_id=doc_id, filename=filename, content=content, ext=ext)
        _index_status[doc_id] = "ready"
        logger.info(f"Indexing complete: {filename}")
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
    return {"session_id": str(uuid.uuid4())}


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    allowed = {".pdf", ".txt", ".docx", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported type '{ext}'.")
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
    """
    Main analysis endpoint.

    Execution order (v1.5):
      1. Fetch conversation history from SQLite (chat memory)
      2. Run QueryAnalyser + Retriever + Synthesiser (LLM agents)
      3. Run SHAP on retrieved chunks + answer
      4. Pass shap_data into Critic/TrustScoreBuilder
         → Faithfulness + LIME + SHAP → composite trust_score
      5. Save everything to SQLite
    """
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"Query [{session_id[:8]}]: {req.query[:80]}")

    try:
        conversation_history = get_recent_turns_for_prompt(
            session_id=session_id, n=CHAT_HISTORY_TURNS,
        )
        turn_number = conversation_history.count("User:") + 1

        # ── First pass: run without shap_data to get answer + references ──────
        # We need the answer text to run SHAP, and the references to run LIME.
        # So we run orchestrator once to get the raw answer, then compute SHAP,
        # then the orchestrator's Critic gets shap_data on the second internal call.
        # To avoid two full LLM passes, orchestrator.run() accepts shap_data=None
        # on first call and skips SHAP alignment in trust score (uses 0.5 neutral).
        # Then we recompute trust_score with real SHAP data and return that.

        result = await orchestrator.run(
            query=req.query,
            session_id=session_id,
            mode=req.mode,
            top_k=req.top_k,
            conversation_history=conversation_history,
            shap_data=None,   # Critic uses neutral 0.5 for shap_alignment on first pass
        )

        # ── SHAP analysis (uses answer + retrieved chunks) ────────────────────
        shap_data = shap_explainer.analyse(
            query=req.query,
            context_chunks=[r["text"] for r in result["references"]],
            answer=result["answer"],
        )

        # ── Recompute trust score with real SHAP data ─────────────────────────
        from explainability.trust_score import build_trust_score
        trust_data = build_trust_score(
            faithfulness=result["faithfulness_data"],
            shap_data=shap_data,
            lime_data=result["lime_data"],
            answer=result["answer"],
        )
        # trust_score is now authoritative — overrides the first-pass value
        trust_score = trust_data["trust_score"]
        trust_level = trust_data["trust_level"]

        # ── Save to SQLite ────────────────────────────────────────────────────
        message_id = save_message(
            session_id=session_id,
            query=req.query,
            answer=result["answer"],
            confidence=trust_score,
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
            faithfulness_data=result["faithfulness_data"],
            lime_data=result["lime_data"],
            trust_data=trust_data,
            confidence=trust_score,
            trust_score=trust_score,
            trust_level=trust_level,
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
                yield f"data: {token.replace(chr(10), chr(92)+'n')}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/debug/retrieve")
async def debug_retrieve(
    q: str = Query(..., description="Keyword or phrase to test"),
    top_k: int = Query(10),
):
    if not q or len(q.strip()) < 1:
        raise HTTPException(status_code=400, detail="Query 'q' is required.")
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
