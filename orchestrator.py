"""
Orchestrator Agent — coordinates the full multi-agent pipeline.

Agent Pipeline:
  1. QueryAnalyserAgent   — classifies intent, extracts entities, decides mode
  2. RetrieverAgent       — fetches relevant chunks from ChromaDB
  3. SynthesiserAgent     — generates grounded answer with citations
  4. CriticAgent          — scores confidence, flags gaps or contradictions
  5. FormatterAgent       — structures final output with references

Each agent logs its reasoning to `agent_trace` for full explainability.
"""
import logging
import httpx
import json
from typing import AsyncGenerator, Dict, Any, List

from rag.document_store import DocumentStore

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
MODEL = "llama3"    # must be pulled: ollama pull llama3


class OrchestratorAgent:
    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store

    # ── Health ────────────────────────────────────────────────────────────────

    def llm_health_check(self) -> str:
        try:
            r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            if any(MODEL in m for m in models):
                return f"ok (model={MODEL})"
            return f"model '{MODEL}' not pulled — run: ollama pull {MODEL}"
        except Exception as e:
            return f"error: {e}"

    # ── Main orchestration ────────────────────────────────────────────────────

    async def run(self, query: str, session_id: str, mode: str = "full", top_k: int = 5) -> Dict[str, Any]:
        agent_trace = []

        # ── Agent 1: Query Analyser ───────────────────────────────────────────
        analysis = await self._query_analyser(query, agent_trace)

        # ── Agent 2: Retriever ────────────────────────────────────────────────
        references = self._retriever(query, top_k, agent_trace)

        if not references:
            return {
                "answer": "No relevant regulatory documents found. Please upload documents first.",
                "agent_trace": agent_trace,
                "references": [],
                "confidence": 0.0,
            }

        # ── Agent 3: Synthesiser ──────────────────────────────────────────────
        answer = await self._synthesiser(query, analysis, references, agent_trace)

        # ── Agent 4: Critic ───────────────────────────────────────────────────
        confidence = await self._critic(query, answer, references, agent_trace)

        # ── Agent 5: Formatter ────────────────────────────────────────────────
        formatted_answer = self._formatter(answer, references, agent_trace)

        return {
            "answer": formatted_answer,
            "agent_trace": agent_trace,
            "references": references,
            "confidence": confidence,
        }

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream(self, query: str, session_id: str) -> AsyncGenerator[str, None]:
        """Yield tokens directly from Ollama for streaming UI."""
        references = self.doc_store.retrieve(query, top_k=3)
        context = "\n\n".join(r["text"] for r in references[:3])
        prompt = self._build_synthesis_prompt(query, context, references)

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{OLLAMA_BASE}/api/generate", json={
                "model": MODEL,
                "prompt": prompt,
                "stream": True,
            }) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

    # ── Agent implementations ─────────────────────────────────────────────────

    async def _query_analyser(self, query: str, trace: list) -> Dict:
        """
        Agent 1 — Query Analyser
        Classifies: regulation_lookup | compliance_check | comparison | summary | general
        Extracts: regulation names, article references, jurisdictions
        """
        prompt = f"""You are a regulatory query analyser. Analyse this query and respond ONLY with valid JSON.

Query: "{query}"

Respond with:
{{
  "intent": "regulation_lookup|compliance_check|comparison|summary|general",
  "regulation_references": ["list of any regulation names mentioned"],
  "jurisdiction": "jurisdiction if mentioned or null",
  "article_references": ["any article/section numbers"],
  "key_concepts": ["main regulatory concepts"],
  "suggested_search_terms": ["optimised search terms for retrieval"]
}}"""

        response = await self._call_ollama(prompt, temperature=0.1)
        try:
            # Strip markdown fences if present
            clean = response.strip().strip("```json").strip("```").strip()
            analysis = json.loads(clean)
        except Exception:
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
            "purpose": "Classifies query intent and extracts regulatory entities",
            "input": query,
            "output": analysis,
            "model": MODEL,
        })
        return analysis

    def _retriever(self, query: str, top_k: int, trace: list) -> List[Dict]:
        """
        Agent 2 — Retriever
        Semantic search in ChromaDB, returns ranked chunks with similarity scores.
        """
        results = self.doc_store.retrieve(query, top_k=top_k)

        trace.append({
            "agent": "Retriever",
            "purpose": "Semantic search in ChromaDB for relevant regulatory chunks",
            "input": {"query": query, "top_k": top_k},
            "output": {
                "chunks_retrieved": len(results),
                "sources": list({r["filename"] for r in results}),
                "top_similarity": results[0]["similarity"] if results else 0,
            },
            "model": "SentenceTransformers/all-MiniLM-L6-v2",
        })
        return results

    async def _synthesiser(self, query: str, analysis: Dict, references: List[Dict], trace: list) -> str:
        """
        Agent 3 — Synthesiser
        Generates a grounded answer using ONLY the retrieved context.
        Each claim is attributed to a source.
        """
        context = "\n\n".join(
            f"[Source {i+1}: {r['filename']} | Chunk {r['chunk_index']} | Similarity: {r['relevance_pct']}%]\n{r['text']}"
            for i, r in enumerate(references)
        )
        prompt = self._build_synthesis_prompt(query, context, references)
        answer = await self._call_ollama(prompt, temperature=0.2, max_tokens=1200)

        trace.append({
            "agent": "Synthesiser",
            "purpose": "Generates grounded answer with in-line source citations",
            "input": {"query": query, "context_chunks": len(references)},
            "output": {"answer_length": len(answer), "preview": answer[:200] + "..."},
            "model": MODEL,
        })
        return answer

    async def _critic(self, query: str, answer: str, references: List[Dict], trace: list) -> float:
        """
        Agent 4 — Critic
        Scores confidence: checks if answer is grounded, flags hallucinations.
        Returns 0.0-1.0 confidence score.
        """
        context_preview = "\n".join(r["text"][:200] for r in references[:3])
        prompt = f"""You are a regulatory compliance critic. Score the following answer for accuracy and groundedness.

QUERY: {query}

ANSWER: {answer}

SOURCE CONTEXT (first 200 chars each):
{context_preview}

Respond ONLY with a JSON object:
{{
  "confidence": <float 0.0 to 1.0>,
  "grounded": <true/false>,
  "issues": ["list any hallucinations, unsupported claims, or gaps"],
  "completeness": <"complete"|"partial"|"insufficient">
}}"""

        response = await self._call_ollama(prompt, temperature=0.0)
        try:
            clean = response.strip().strip("```json").strip("```").strip()
            critic_result = json.loads(clean)
            confidence = float(critic_result.get("confidence", 0.7))
        except Exception:
            confidence = 0.7
            critic_result = {"confidence": confidence,
                             "issues": [], "grounded": True}

        trace.append({
            "agent": "Critic",
            "purpose": "Validates answer groundedness and scores confidence",
            "input": {"answer_length": len(answer)},
            "output": critic_result,
            "model": MODEL,
        })
        return min(max(confidence, 0.0), 1.0)

    def _formatter(self, answer: str, references: List[Dict], trace: list) -> str:
        """
        Agent 5 — Formatter
        Ensures consistent output structure with clear reference markers.
        """
        ref_section = "\n\n---\n**References Used:**\n"
        for i, ref in enumerate(references):
            location = ""
            if ref.get("page_number") and ref["page_number"] > 0:
                location += f" | 📄 Page {ref['page_number']}"
            if ref.get("clause_refs"):
                location += f" | 📌 {ref['clause_refs']}"
            ref_section += (
                f"\n[{i+1}] **{ref['filename']}**"
                f"{location}"
                f" | Chunk {ref['chunk_index']}/{ref['total_chunks']}"
                f" (Relevance: {ref['relevance_pct']}%)"
            )

        formatted = answer + ref_section

        trace.append({
            "agent": "Formatter",
            "purpose": "Appends structured reference list to the answer",
            "input": {"references": len(references)},
            "output": {"final_length": len(formatted)},
            "model": "rule-based",
        })
        return formatted

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_synthesis_prompt(self, query: str, context: str, references: List[Dict]) -> str:
        source_list = "\n".join(
            f"- [{i+1}] {r['filename']}" for i, r in enumerate(references))
        return f"""You are an expert regulatory compliance assistant. Answer the query using ONLY the provided regulatory context.

AVAILABLE SOURCES:
{source_list}

REGULATORY CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Answer concisely and accurately using the context above.
2. Cite sources using [1], [2] etc. after each claim.
3. If the context does not contain enough information, clearly state what is missing.
4. Use formal, precise regulatory language.
5. Do NOT invent rules or references not present in the context.

ANSWER:"""

    async def _call_ollama(self, prompt: str, temperature: float = 0.2, max_tokens: int = 800) -> str:
        """Call Ollama REST API (non-streaming)."""
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/generate", json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            })
            resp.raise_for_status()
            return resp.json().get("response", "")
