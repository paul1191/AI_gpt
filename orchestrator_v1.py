"""
Orchestrator Agent — coordinates the full multi-agent pipeline.

Agent Pipeline:
  1. QueryAnalyserAgent  — classifies intent, extracts entities, refines search terms
  2. RetrieverAgent      — fetches relevant chunks from ChromaDB using refined terms
  3. SynthesiserAgent    — generates grounded answer with [N] inline citations
  4. CriticAgent         — scores confidence, flags hallucinations / gaps
  5. FormatterAgent      — structures final output with clean reference list

Each agent appends a structured record to `agent_trace` — this is the full
audit trail returned to the frontend for explainability.

Design decisions:
  - OLLAMA_BASE and MODEL are read from env vars with sensible defaults
  - A single httpx.AsyncClient is reused across all Ollama calls (connection pooling)
  - JSON parsing uses regex-based fence stripping (not .strip("```json") which
    strips individual characters, not the full fence string)
  - The QueryAnalyser's suggested_search_terms are used by the Retriever to
    improve semantic recall beyond the raw query
  - Critic agent uses temperature=0.0 for deterministic scoring
"""
import os
import re
import json
import logging
import httpx
from typing import AsyncGenerator, Dict, Any, List

from rag.document_store import DocumentStore

logger = logging.getLogger(__name__)

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "llama3.1:latest")


def _strip_json_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM JSON output.

    BUG FIX: str.strip("```json") strips individual *characters* from both
    ends — it does NOT strip the substring. Use regex instead.
    """
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class OrchestratorAgent:
    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store
        # Reuse a single client for all Ollama calls — avoids connection setup overhead
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    # ── Health ────────────────────────────────────────────────────────────────

    def llm_health_check(self) -> str:
        """Synchronous health check (called from FastAPI startup)."""
        try:
            # Use a short-lived sync client here — this is only called rarely
            r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if any(MODEL in m for m in models):
                return f"ok (model={MODEL})"
            return f"warning: model '{MODEL}' not found — run: ollama pull {MODEL}"
        except httpx.ConnectError:
            return "error: Ollama not running — run: ollama serve"
        except Exception as e:
            return f"error: {e}"

    # ── Main orchestration ────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        session_id: str,
        mode: str = "full",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        agent_trace: List[Dict] = []

        # Agent 1: Analyse query intent and extract search hints
        analysis = await self._query_analyser(query, agent_trace)

        # Agent 2: Retrieve — use suggested_search_terms if available
        search_terms = analysis.get("suggested_search_terms") or [query]
        # Combine original query + first suggested term for richer retrieval
        combined_query = query
        if search_terms and search_terms[0] != query:
            combined_query = f"{query} {search_terms[0]}"

        references = self._retriever(combined_query, top_k, agent_trace)

        if not references:
            agent_trace.append({
                "agent": "Orchestrator",
                "purpose": "Early exit — no documents in index",
                "input": query,
                "output": "No documents found",
                "model": "rule-based",
            })
            return {
                "answer": (
                    "⚠️ No relevant regulatory documents found in the index.\n\n"
                    "Please upload at least one document via the sidebar before querying."
                ),
                "agent_trace": agent_trace,
                "references": [],
                "confidence": 0.0,
            }

        # Agent 3: Synthesise a grounded answer
        answer = await self._synthesiser(query, analysis, references, agent_trace)

        # Agent 4: Critic scores confidence and checks groundedness
        confidence = await self._critic(query, answer, references, agent_trace)

        # Agent 5: Format output with clean reference list
        formatted_answer = self._formatter(answer, references, agent_trace)

        return {
            "answer": formatted_answer,
            "agent_trace": agent_trace,
            "references": references,
            "confidence": confidence,
        }

    # ── Streaming (for SSE endpoint) ──────────────────────────────────────────

    async def stream(self, query: str, session_id: str) -> AsyncGenerator[str, None]:
        """Yield tokens from Ollama for real-time streaming UI."""
        references = self.doc_store.retrieve(query, top_k=3)
        if not references:
            yield "No documents found. Please upload regulatory documents first."
            return

        # Build context string and pass to synthesis prompt
        context = "\n\n".join(
            f"[Source {i+1}: {r['filename']}]\n{r['text']}"
            for i, r in enumerate(references)
        )
        prompt = self._build_synthesis_prompt(query, context, references)

        async with self._http.stream("POST", f"{OLLAMA_BASE}/api/generate", json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
        }) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if token := data.get("response"):
                            yield token
                    except json.JSONDecodeError:
                        continue

    # ── Agent 1: Query Analyser ───────────────────────────────────────────────

    async def _query_analyser(self, query: str, trace: List) -> Dict:
        """
        Classifies query intent and extracts regulatory entities.
        Output is used to:
          (a) guide the Retriever with richer search terms
          (b) give the Synthesiser context about what the user really needs
        """
        prompt = f"""You are a regulatory query analyser. Analyse the query below and respond ONLY with a valid JSON object — no preamble, no explanation, no markdown fences.

Query: "{query}"

Return exactly this structure:
{{
  "intent": "regulation_lookup|compliance_check|comparison|summary|general",
  "regulation_references": ["list of any regulation names mentioned, e.g. GDPR, Basel III"],
  "jurisdiction": "jurisdiction if mentioned, else null",
  "article_references": ["any article or section numbers mentioned"],
  "key_concepts": ["main regulatory concepts in the query"],
  "suggested_search_terms": ["2-3 alternative search phrases that would find relevant chunks"]
}}"""

        raw = await self._call_ollama(prompt, temperature=0.1)
        try:
            analysis = json.loads(_strip_json_fences(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"QueryAnalyser JSON parse failed, using defaults. Raw: {raw[:200]}")
            analysis = {
                "intent": "general",
                "key_concepts": [query],
                "suggested_search_terms": [query],
                "regulation_references": [],
                "jurisdiction": None,
                "article_references": [],
            }

        trace.append({
            "agent": "QueryAnalyser",
            "purpose": "Classifies query intent and extracts regulatory entities to improve retrieval",
            "input": query,
            "output": analysis,
            "model": MODEL,
        })
        return analysis

    # ── Agent 2: Retriever ────────────────────────────────────────────────────

    def _retriever(self, query: str, top_k: int, trace: List) -> List[Dict]:
        """
        Semantic search against ChromaDB.
        Uses the combined query (original + suggested terms) for better recall.
        """
        results = self.doc_store.retrieve(query, top_k=top_k)

        trace.append({
            "agent": "Retriever",
            "purpose": "Semantic search in ChromaDB using cosine similarity on all-MiniLM-L6-v2 embeddings",
            "input": {"query": query, "top_k": top_k},
            "output": {
                "chunks_retrieved": len(results),
                "unique_sources": list({r["filename"] for r in results}),
                "top_similarity_score": results[0]["similarity"] if results else 0,
                "similarity_scores": [r["similarity"] for r in results],
            },
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        })
        return results

    # ── Agent 3: Synthesiser ──────────────────────────────────────────────────

    async def _synthesiser(
        self, query: str, analysis: Dict, references: List[Dict], trace: List
    ) -> str:
        """
        Generates a grounded, cited answer from the retrieved context only.
        Uses [1], [2] inline citations keyed to the reference list.
        """
        context = "\n\n".join(
            f"[Source {i+1}: {r['filename']} | Chunk {r['chunk_index']} | Relevance: {r['relevance_pct']}%]\n{r['text']}"
            for i, r in enumerate(references)
        )
        prompt = self._build_synthesis_prompt(query, context, references)
        answer = await self._call_ollama(prompt, temperature=0.2, max_tokens=1200)

        trace.append({
            "agent": "Synthesiser",
            "purpose": "Generates a grounded answer using only the retrieved regulatory context, with inline [N] citations",
            "input": {
                "query": query,
                "detected_intent": analysis.get("intent", "unknown"),
                "context_chunks_used": len(references),
            },
            "output": {
                "answer_char_length": len(answer),
                "preview": answer[:300] + ("..." if len(answer) > 300 else ""),
            },
            "model": MODEL,
        })
        return answer

    # ── Agent 4: Critic ───────────────────────────────────────────────────────

    async def _critic(
        self, query: str, answer: str, references: List[Dict], trace: List
    ) -> float:
        """
        Validates the answer against the source context.
        Flags unsupported claims, scores confidence 0.0–1.0.
        Temperature=0.0 for deterministic, consistent scoring.
        """
        context_preview = "\n---\n".join(
            f"[Source {i+1}] {r['text'][:300]}" for i, r in enumerate(references[:4])
        )
        prompt = f"""You are a strict regulatory compliance critic. Evaluate the answer below against the provided source context.

QUERY: {query}

ANSWER TO EVALUATE:
{answer}

SOURCE CONTEXT:
{context_preview}

Respond ONLY with a valid JSON object — no preamble, no markdown:
{{
  "confidence": <float 0.0 to 1.0, where 1.0 = fully grounded>,
  "grounded": <true if all claims are supported by context, false otherwise>,
  "issues": ["list any specific unsupported claims, hallucinations, or gaps — empty list if none"],
  "completeness": "complete|partial|insufficient"
}}"""

        raw = await self._call_ollama(prompt, temperature=0.0)
        try:
            critic_result = json.loads(_strip_json_fences(raw))
            confidence = float(critic_result.get("confidence", 0.7))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning(f"Critic JSON parse failed. Raw: {raw[:200]}")
            confidence = 0.7
            critic_result = {
                "confidence": confidence,
                "grounded": True,
                "issues": [],
                "completeness": "partial",
            }

        trace.append({
            "agent": "Critic",
            "purpose": "Validates answer groundedness against source context and scores confidence",
            "input": {"answer_length": len(answer), "sources_checked": len(references[:4])},
            "output": critic_result,
            "model": MODEL,
        })
        return min(max(confidence, 0.0), 1.0)

    # ── Agent 5: Formatter ────────────────────────────────────────────────────

    def _formatter(self, answer: str, references: List[Dict], trace: List) -> str:
        """
        Appends a structured reference list to the answer.
        Deduplicates sources by filename while preserving chunk detail.
        Rule-based — no LLM call needed.
        """
        ref_lines = ["\n\n---\n**References Used:**"]
        for i, ref in enumerate(references):
            # Build location string
            location = ""
            if ref.get("page_number") and ref["page_number"] > 0:
                page_label = ref.get("page_label") or f"Page {ref['page_number']}"
                location += f" | 📄 {page_label}"
            if ref.get("clause_refs"):
                location += f" | 📌 {ref['clause_refs']}"

            ref_lines.append(
                f"\n[{i+1}] **{ref['filename']}**"
                f"{location}"
                f" | Chunk {ref['chunk_index']}/{ref['total_chunks']}"
                f" (Relevance: {ref['relevance_pct']}%)"
            )

        formatted = answer + "".join(ref_lines)

        trace.append({
            "agent": "Formatter",
            "purpose": "Appends structured reference list with chunk-level detail to the answer",
            "input": {"references_count": len(references)},
            "output": {"final_char_length": len(formatted)},
            "model": "rule-based",
        })
        return formatted

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_synthesis_prompt(
        self, query: str, context: str, references: List[Dict]
    ) -> str:
        source_list = "\n".join(
            f"  [{i+1}] {r['filename']}" for i, r in enumerate(references)
        )
        return f"""You are an expert regulatory compliance assistant. Your task is to answer the query accurately using ONLY the provided regulatory context.

AVAILABLE SOURCES:
{source_list}

REGULATORY CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Answer concisely and accurately using only the context above.
2. After each factual claim, cite the source using [1], [2] etc.
3. Use formal, precise regulatory language appropriate for compliance professionals.
4. If the context is insufficient to fully answer the query, explicitly state what information is missing.
5. Do NOT invent, assume, or infer rules or facts not present in the provided context.

ANSWER:"""

    async def _call_ollama(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        """Call Ollama /api/generate and return the full response string."""
        try:
            resp = await self._http.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            raise RuntimeError(
                "Ollama request timed out. Is the model loaded? "
                f"Try: ollama run {MODEL}"
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama")
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE}. "
                "Run: ollama serve"
            )
