"""
Orchestrator Agent — coordinates the full multi-agent pipeline.

Agent Pipeline:
  1. QueryAnalyserAgent  — classifies intent, extracts entities, refines search terms
  2. RetrieverAgent      — fetches relevant chunks from ChromaDB using refined terms
  3. SynthesiserAgent    — generates grounded answer WITH conversation history context
  4. CriticAgent         — scores confidence, flags hallucinations / gaps
  5. FormatterAgent      — structures final output with clean reference list

Chat mode (v1.3):
  - run() now accepts `conversation_history: str` — the last N turns formatted
    as "User: ...\nAssistant: ..." pairs
  - History is injected into the Synthesiser prompt BEFORE the current query
  - This lets the LLM resolve pronouns, follow-ups and implicit references
    (e.g. "what about liquidity?" after asking about capital requirements)
  - QueryAnalyser also receives history so it can extract better search terms
    from follow-up questions that lack explicit regulatory keywords
  - History is NOT passed to the Critic — it scores the current answer only
"""
import os
import re
import json
import logging
import httpx
from typing import AsyncGenerator, Dict, Any, List

from rag.document_store import DocumentStore

logger = logging.getLogger(__name__)

OLLAMA_BASE  = os.getenv("OLLAMA_BASE",        "http://localhost:11434")
MODEL        = os.getenv("OLLAMA_MODEL",        "llama3.1")
CRITIC_MODEL = os.getenv("OLLAMA_CRITIC_MODEL", MODEL)   # override to use different model for critic


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from LLM JSON output using regex (not str.strip)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class OrchestratorAgent:
    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    # ── Health ────────────────────────────────────────────────────────────────

    def llm_health_check(self) -> str:
        try:
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
        conversation_history: str = "",   # NEW: last N turns as formatted string
    ) -> Dict[str, Any]:
        """
        Full 5-agent pipeline.

        conversation_history: pre-formatted string of prior turns:
          "User: <q1>\nAssistant: <a1>\n\nUser: <q2>\nAssistant: <a2>"
        Empty string = first message in session (no history).
        """
        agent_trace: List[Dict] = []
        has_history = bool(conversation_history.strip())

        # Agent 1 — QueryAnalyser (pass history so it can handle follow-up questions)
        analysis = await self._query_analyser(query, agent_trace, conversation_history)

        # Agent 2 — Retriever (use suggested_search_terms for better recall)
        search_terms  = analysis.get("suggested_search_terms") or [query]
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

        # Agent 3 — Synthesiser (receives conversation history for context)
        answer = await self._synthesiser(
            query, analysis, references, agent_trace, conversation_history
        )

        # Agent 4 — Critic (scores current answer only, no history needed)
        confidence = await self._critic(query, answer, references, agent_trace)

        # Agent 5 — Formatter
        formatted_answer = self._formatter(answer, references, agent_trace)

        return {
            "answer":      formatted_answer,
            "agent_trace": agent_trace,
            "references":  references,
            "confidence":  confidence,
        }

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream(self, query: str, session_id: str) -> AsyncGenerator[str, None]:
        references = self.doc_store.retrieve(query, top_k=3)
        if not references:
            yield "No documents found. Please upload regulatory documents first."
            return
        context = "\n\n".join(
            f"[Source {i+1}: {r['filename']}]\n{r['text']}"
            for i, r in enumerate(references)
        )
        prompt = self._build_synthesis_prompt(query, context, references, "")
        async with self._http.stream("POST", f"{OLLAMA_BASE}/api/generate", json={
            "model": MODEL, "prompt": prompt, "stream": True,
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

    async def _query_analyser(
        self, query: str, trace: List, conversation_history: str = ""
    ) -> Dict:
        """
        Classifies intent and extracts regulatory entities.

        Chat mode: if conversation_history is provided, it is shown to the LLM
        so it can resolve follow-up questions that lack explicit keywords.
        e.g. "what about liquidity?" → infers "Basel III liquidity requirements"
        from the history context.
        """
        history_block = ""
        if conversation_history.strip():
            history_block = f"""
CONVERSATION SO FAR (for context only — use this to understand what the user is referring to):
{conversation_history}

"""

        prompt = f"""You are a regulatory query analyser. Analyse the CURRENT QUERY below and respond ONLY with a valid JSON object — no preamble, no explanation, no markdown fences.
{history_block}
CURRENT QUERY: "{query}"

Return exactly this structure:
{{
  "intent": "regulation_lookup|compliance_check|comparison|summary|general",
  "regulation_references": ["regulation names mentioned or implied from history"],
  "jurisdiction": "jurisdiction if mentioned or implied, else null",
  "article_references": ["any article or section numbers"],
  "key_concepts": ["main regulatory concepts"],
  "suggested_search_terms": ["2-3 search phrases that would find relevant document chunks"]
}}"""

        raw = await self._call_ollama(prompt, temperature=0.1)
        try:
            analysis = json.loads(_strip_json_fences(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"QueryAnalyser JSON parse failed. Raw: {raw[:200]}")
            analysis = {
                "intent": "general",
                "key_concepts": [query],
                "suggested_search_terms": [query],
                "regulation_references": [],
                "jurisdiction": None,
                "article_references": [],
            }

        trace.append({
            "agent":   "QueryAnalyser",
            "purpose": "Classifies intent and extracts regulatory entities (history-aware for follow-ups)",
            "input":   {"query": query, "has_history": bool(conversation_history.strip())},
            "output":  analysis,
            "model":   MODEL,
        })
        return analysis

    # ── Agent 2: Retriever ────────────────────────────────────────────────────

    def _retriever(self, query: str, top_k: int, trace: List) -> List[Dict]:
        results = self.doc_store.retrieve(query, top_k=top_k)
        trace.append({
            "agent":   "Retriever",
            "purpose": "Semantic search in ChromaDB using cosine similarity",
            "input":   {"query": query, "top_k": top_k},
            "output":  {
                "chunks_retrieved":    len(results),
                "unique_sources":      list({r["filename"] for r in results}),
                "top_similarity_score": results[0]["similarity"] if results else 0,
                "similarity_scores":   [r["similarity"] for r in results],
            },
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        })
        return results

    # ── Agent 3: Synthesiser ──────────────────────────────────────────────────

    async def _synthesiser(
        self,
        query: str,
        analysis: Dict,
        references: List[Dict],
        trace: List,
        conversation_history: str = "",
    ) -> str:
        """
        Generates a grounded, cited answer.

        Chat mode: conversation_history is injected into the prompt so the LLM
        can resolve references to prior answers (e.g. "those requirements",
        "the regulation we discussed", "how does that compare to...").
        """
        context = "\n\n".join(
            f"[Source {i+1}: {r['filename']} | Chunk {r['chunk_index']} | Relevance: {r['relevance_pct']}%]\n{r['text']}"
            for i, r in enumerate(references)
        )
        prompt = self._build_synthesis_prompt(
            query, context, references, conversation_history
        )
        answer = await self._call_ollama(prompt, temperature=0.2, max_tokens=1200)

        trace.append({
            "agent":   "Synthesiser",
            "purpose": "Generates grounded answer with [N] citations, conversation-history-aware",
            "input":   {
                "query":               query,
                "detected_intent":     analysis.get("intent", "unknown"),
                "context_chunks_used": len(references),
                "history_turns_used":  conversation_history.count("User:"),
            },
            "output":  {
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
  "issues": ["list any unsupported claims, hallucinations, or gaps — empty list if none"],
  "completeness": "complete|partial|insufficient"
}}"""

        raw = await self._call_ollama(prompt, temperature=0.0, model=CRITIC_MODEL)
        try:
            critic_result = json.loads(_strip_json_fences(raw))
            confidence = float(critic_result.get("confidence", 0.7))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning(f"Critic JSON parse failed. Raw: {raw[:200]}")
            confidence = 0.7
            critic_result = {
                "confidence": confidence, "grounded": True,
                "issues": [], "completeness": "partial",
            }

        trace.append({
            "agent":   "Critic",
            "purpose": "Validates answer groundedness and scores confidence",
            "input":   {"answer_length": len(answer), "sources_checked": len(references[:4])},
            "output":  critic_result,
            "model":   CRITIC_MODEL,
        })
        return min(max(confidence, 0.0), 1.0)

    # ── Agent 5: Formatter ────────────────────────────────────────────────────

    def _formatter(self, answer: str, references: List[Dict], trace: List) -> str:
        ref_section = "\n\n---\n**References Used:**\n"
        for i, ref in enumerate(references):
            location = ""
            if ref.get("page_number") and ref["page_number"] > 0:
                page_label = ref.get("page_label") or f"Page {ref['page_number']}"
                location += f" | 📄 {page_label}"
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
            "agent":   "Formatter",
            "purpose": "Appends structured reference list with page and clause detail",
            "input":   {"references_count": len(references)},
            "output":  {"final_char_length": len(formatted)},
            "model":   "rule-based",
        })
        return formatted

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_synthesis_prompt(
        self,
        query: str,
        context: str,
        references: List[Dict],
        conversation_history: str,
    ) -> str:
        source_list = "\n".join(
            f"  [{i+1}] {r['filename']}" for i, r in enumerate(references)
        )

        # Chat history block — only included when history exists
        history_block = ""
        if conversation_history.strip():
            history_block = f"""
CONVERSATION HISTORY (previous turns in this session — use this to understand follow-up questions and resolve references like "those requirements", "the regulation above", "how does that compare"):
{conversation_history}

"""

        return f"""You are an expert regulatory compliance assistant. Answer the CURRENT QUERY accurately using ONLY the provided regulatory context.
{history_block}
AVAILABLE SOURCES:
{source_list}

REGULATORY CONTEXT:
{context}

CURRENT QUERY: {query}

INSTRUCTIONS:
1. Answer concisely and accurately using only the regulatory context above.
2. After each factual claim, cite the source using [1], [2] etc.
3. Use formal, precise regulatory language appropriate for compliance professionals.
4. If this is a follow-up question, connect your answer to the conversation history naturally.
5. If the context is insufficient to fully answer, explicitly state what is missing.
6. Do NOT invent, assume, or infer rules or facts not present in the provided context.

ANSWER:"""

    async def _call_ollama(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        model: str = None,
    ) -> str:
        """Call Ollama /api/generate and return the full response string."""
        use_model = model or MODEL
        try:
            resp = await self._http.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": use_model,
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
                f"Ollama timed out. Is the model loaded? Try: ollama run {use_model}"
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama")
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE}. Run: ollama serve"
            )
