"""
Regulation Analyser & Assistant - Main FastAPI Application
Multi-Agent System with RAG, SHAP Explainability, and Document References
"""
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio

from agents.orchestrator import OrchestratorAgent
from rag.document_store import DocumentStore
from explainability.shap_explainer import SHAPExplainer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Regulation Analyser API",
    description="On-premise multi-agent AI for regulatory document analysis with explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────
doc_store = DocumentStore()
orchestrator = OrchestratorAgent(doc_store)
shap_explainer = SHAPExplainer()

# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: Optional[str] = "full"          # full | quick | compare
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    agent_trace: list                      # which agents ran & why
    references: list                       # source documents with chunks
    shap_analysis: dict                    # SHAP feature importance
    confidence: float
    mode: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check Ollama + ChromaDB connectivity"""
    status = {
        "api": "ok",
        "chroma": doc_store.health_check(),
        "ollama": orchestrator.llm_health_check(),
        "documents_indexed": doc_store.count_documents(),
    }
    return status


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and index a regulatory document into ChromaDB"""
    allowed = {".pdf", ".txt", ".docx", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    contents = await file.read()
    doc_id = str(uuid.uuid4())

    # Index in background so upload returns fast
    background_tasks.add_task(
        doc_store.ingest_document,
        doc_id=doc_id,
        filename=file.filename,
        content=contents,
        ext=ext,
    )
    logger.info(f"Queued document for indexing: {file.filename} | id={doc_id}")
    return {"doc_id": doc_id, "filename": file.filename, "status": "indexing"}


@app.get("/documents")
async def list_documents():
    """List all indexed documents"""
    return doc_store.list_documents()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the index"""
    doc_store.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Main query endpoint.
    Orchestrates multiple agents, retrieves context, generates answer,
    and returns SHAP explainability + document references.
    """
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"Query [{session_id}]: {req.query[:80]}...")

    try:
        result = await orchestrator.run(
            query=req.query,
            session_id=session_id,
            mode=req.mode,
            top_k=req.top_k,
        )

        # Attach SHAP analysis to the retrieved context
        shap_data = shap_explainer.analyse(
            query=req.query,
            context_chunks=[r["text"] for r in result["references"]],
            answer=result["answer"],
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
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query processing failed: {str(e)}")


@app.get("/query/stream")
async def query_stream(query: str, session_id: Optional[str] = None):
    """Streaming version of query for real-time token output"""
    session_id = session_id or str(uuid.uuid4())

    async def event_generator():
        async for token in orchestrator.stream(query=query, session_id=session_id):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
