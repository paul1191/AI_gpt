"""
Orchestrator Agent — coordinates the full multi-agent pipeline.

Agent Pipeline v1.5:
  1. QueryAnalyserAgent  — classifies intent, extracts entities, refines search terms
  2. RetrieverAgent      — hybrid BM25 + semantic search via DocumentStore
  3. SynthesiserAgent    — grounded answer with [N] inline citations (LLM)
  4. CriticAgent         — RAG Faithfulness scorer (algorithmic, no LLM call)
                           replaces previous LLM-based confidence scoring
  5. FormatterAgent      — structures final output with clean reference list

The Critic no longer calls Ollama. It delegates to FaithfulnessExplainer,
LIMEExplainer, and TrustScoreBuilder — all algorithmic, deterministic, and fast.
The composite trust_score replaces the old LLM confidence float.

Trust score is built from 5 signals:
  grounding_ratio      (RAG Faithfulness — 30%)
  avg_max_similarity   (RAG Faithfulness — 25%)
  source_coverage      (RAG Faithfulness — 15%)
  shap_query_alignment (SHAP             — 15%)
  lime_stability       (LIME             — 15%)

Chat mode (v1.3+):
  conversation_history is injected into QueryAnalyser + Synthesiser prompts.
"""
import os
import re
import json
import logging
import httpx
from typing import AsyncGenerator, Dict, Any, List

from rag.document_store import DocumentStore
from explainability.faithfulness_explainer import FaithfulnessExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.trust_score import build_trust_score

logger = logging.getLogger(__name__)

OLLAMA_BASE  = os.getenv("OLLAMA_BASE",         "http://localhost:11434")
MODEL        = os.getenv("OLLAMA_MODEL",         "llama3.1")
CRITIC_MODEL = os.getenv("OLLAMA_CRITIC_MODEL",  MODEL)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class OrchestratorAgent:
    def __init__(self, doc_store: DocumentStore):
        self.doc_store   = doc_store
        self._http       = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self._faithfulness = FaithfulnessExplainer()
        self._lime         = LIMEExplainer()

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
        conversation_history: str = "",
        shap_data: Dict = None,    # passed from main.py after SHAP analysis
    ) -> Dict[str, Any]:
        agent_trace: List[Dict] = []

        # Agent 1 — QueryAnalyser
        analysis = await self._query_analyser(query, agent_trace, conversation_history)

        # Agent 2 — Retriever
        search_terms   = analysis.get("suggested_search_terms") or [query]
        combined_query = query
        if search_terms and search_terms[0] != query:
            combined_query = f"{query} {search_terms[0]}"
        references = self._retriever(combined_query, top_k, agent_trace)

        if not references:
            agent_trace.append({
                "agent":   "Orchestrator",
                "purpose": "Early exit — no documents in index",
                "input":   query,
                "output":  "No documents found",
                "model":   "rule-based",
            })
            return {
                "answer": (
                    "⚠️ No relevant regulatory documents found in the index.\n\n"
                    "Please upload at least one document via the sidebar before querying."
                ),
                "agent_trace": agent_trace,
                "references":  [],
                "confidence":  0.0,
                "trust_data":  {},
                "faithfulness_data": {},
                "lime_data":   {},
            }

        # Agent 3 — Synthesiser
        answer = await self._synthesiser(
            query, analysis, references, agent_trace, conversation_history
        )

        # Agent 4 — Critic (RAG Faithfulness — algorithmic, no LLM)
        faithfulness_data, lime_data, trust_data = self._critic(
            query=query,
            answer=answer,
            references=references,
            agent_trace=agent_trace,
            shap_data=shap_data or {},
        )

        # Use trust_score as the confidence value (replaces old LLM float)
        confidence = trust_data.get("trust_score", 0.5)

        # Agent 5 — Formatter
        formatted_answer = self._formatter(answer, references, agent_trace)

        return {
            "answer":            formatted_answer,
            "agent_trace":       agent_trace,
            "references":        references,
            "confidence":        confidence,
            "trust_data":        trust_data,
            "faithfulness_data": faithfulness_data,
            "lime_data":         lime_data,
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
        history_block = ""
        if conversation_history.strip():
            history_block = (
                f"\nCONVERSATION SO FAR (use this to understand follow-up questions):\n"
                f"{conversation_history}\n"
            )

        prompt = f"""You are a regulatory query analyser. Analyse the CURRENT QUERY and respond ONLY with a valid JSON object — no preamble, no markdown fences.
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
            analysis = {
                "intent": "general", "key_concepts": [query],
                "suggested_search_terms": [query],
                "regulation_references": [], "jurisdiction": None, "article_references": [],
            }

        trace.append({
            "agent":   "QueryAnalyser",
            "purpose": "Classifies intent and extracts regulatory entities (history-aware)",
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
            "purpose": "Hybrid BM25 + semantic search with RRF re-ranking",
            "input":   {"query": query, "top_k": top_k},
            "output":  {
                "chunks_retrieved":     len(results),
                "unique_sources":       list({r["filename"] for r in results}),
                "top_similarity_score": results[0]["similarity"] if results else 0,
                "bm25_hits":            sum(1 for r in results if r.get("bm25_hit")),
            },
            "model": "BM25 + sentence-transformers/all-MiniLM-L6-v2 + RRF",
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
            "purpose": "Generates grounded answer with [N] citations",
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

    # ── Agent 4: Critic (RAG Faithfulness — no LLM) ───────────────────────────

    def _critic(
        self,
        query: str,
        answer: str,
        references: List[Dict],
        agent_trace: List,
        shap_data: Dict,
    ):
        """
        Algorithmic critic — no LLM call.

        1. FaithfulnessExplainer: sentence-level grounding vs retrieved chunks
        2. LIMEExplainer: retrieval stability under query perturbation
        3. TrustScoreBuilder: combines faithfulness + LIME + SHAP signals

        Returns (faithfulness_data, lime_data, trust_data).
        """
        # Get the shared embedder from DocumentStore (no second model load)
        embedder = getattr(self.doc_store, "embedder", None)

        # Step 1 — Faithfulness
        faithfulness_data = self._faithfulness.analyse(
            answer=answer,
            chunks=references,
            embedder=embedder,
        )

        # Step 2 — LIME
        lime_data = self._lime.analyse(
            query=query,
            original_chunks=references,
            doc_store=self.doc_store,
            top_k=len(references),
        )

        # Step 3 — Trust Score (uses signals from all three)
        trust_data = build_trust_score(
            faithfulness=faithfulness_data,
            shap_data=shap_data,
            lime_data=lime_data,
            answer=answer,
        )

        agent_trace.append({
            "agent":   "Critic",
            "purpose": "RAG Faithfulness + LIME stability + SHAP alignment → composite trust score",
            "input":   {
                "answer_sentences":  faithfulness_data.get("total_sentences", 0),
                "perturbations_run": lime_data.get("n_perturbations", 0),
                "shap_tokens_used":  len(shap_data.get("query_alignment_tokens", [])),
            },
            "output":  {
                "trust_score":    trust_data.get("trust_score"),
                "trust_level":    trust_data.get("trust_level"),
                "grounding_ratio": faithfulness_data.get("grounding_ratio"),
                "lime_stability":  lime_data.get("stability_score"),
                "warnings":        trust_data.get("warnings", []),
            },
            "model": "algorithmic (FaithfulnessExplainer + LIMEExplainer + TrustScoreBuilder)",
        })

        return faithfulness_data, lime_data, trust_data

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
        self, query: str, context: str, references: List[Dict], conversation_history: str
    ) -> str:
        source_list = "\n".join(
            f"  [{i+1}] {r['filename']}" for i, r in enumerate(references)
        )
        history_block = ""
        if conversation_history.strip():
            history_block = (
                f"\nCONVERSATION HISTORY (use this to understand follow-up questions):\n"
                f"{conversation_history}\n"
            )
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
3. Use formal, precise regulatory language.
4. If this is a follow-up question, connect your answer to the conversation history naturally.
5. If the context is insufficient, explicitly state what is missing.
6. Do NOT invent or infer rules not present in the provided context.

ANSWER:"""

    async def _call_ollama(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        model: str = None,
    ) -> str:
        use_model = model or MODEL
        try:
            resp = await self._http.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model":   use_model,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except httpx.TimeoutException:
            raise RuntimeError(f"Ollama timed out. Try: ollama run {use_model}")
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_BASE}. Run: ollama serve")
