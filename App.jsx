import { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000";

// ── Colour palette ────────────────────────────────────────────────────────────
// Deep navy / amber / slate — authoritative, legal, precise
const c = {
  bg:       "#0B0F1A",
  surface:  "#111827",
  surface2: "#1a2235",
  border:   "#1e2d45",
  accent:   "#F59E0B",
  accent2:  "#3B82F6",
  success:  "#10B981",
  danger:   "#EF4444",
  text:     "#E2E8F0",
  muted:    "#64748B",
  highlight:"#FDE68A",
};

// ── Tiny styled helpers ────────────────────────────────────────────────────────
const pill = (color, text) => (
  <span style={{
    background: color + "22", color, border: `1px solid ${color}44`,
    borderRadius: 20, padding: "2px 10px", fontSize: 11, fontWeight: 700,
    letterSpacing: 0.5, textTransform: "uppercase",
  }}>{text}</span>
);

function ConfidenceMeter({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? c.success : pct >= 50 ? c.accent : c.danger;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div style={{ flex: 1, height: 6, background: "#1e2d45", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4,
          transition: "width 0.8s ease", boxShadow: `0 0 8px ${color}88` }} />
      </div>
      <span style={{ color, fontWeight: 700, fontSize: 13, minWidth: 40 }}>{pct}%</span>
    </div>
  );
}

function AgentStep({ step, index }) {
  const [open, setOpen] = useState(false);
  const colors = [c.accent2, c.accent, c.success, "#A78BFA", "#F472B6"];
  const col = colors[index % colors.length];
  return (
    <div style={{ border: `1px solid ${col}33`, borderRadius: 8, overflow: "hidden", marginBottom: 8 }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", background: `${col}11`, padding: "10px 14px",
        border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 10,
      }}>
        <span style={{ width: 22, height: 22, borderRadius: "50%", background: col,
          color: "#000", fontSize: 11, fontWeight: 800, display: "flex", alignItems: "center", justifyContent: "center" }}>
          {index + 1}
        </span>
        <span style={{ color: col, fontWeight: 700, fontSize: 13, flex: 1, textAlign: "left" }}>
          {step.agent}
        </span>
        <span style={{ color: c.muted, fontSize: 11 }}>{step.model}</span>
        <span style={{ color: c.muted, fontSize: 16 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{ padding: "12px 14px", background: c.surface }}>
          <p style={{ color: c.muted, fontSize: 12, marginBottom: 8 }}>{step.purpose}</p>
          <pre style={{ color: c.text, fontSize: 11, background: c.bg, padding: 10, borderRadius: 6,
            overflow: "auto", maxHeight: 200, margin: 0 }}>
            {JSON.stringify(step.output, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function ShapChart({ data }) {
  if (!data?.top_features?.length) return null;
  const max = Math.max(...data.top_features.map(f => f.importance));
  return (
    <div>
      <p style={{ color: c.muted, fontSize: 11, marginBottom: 12 }}>{data.explanation}</p>
      {data.top_features.slice(0, 10).map((f, i) => (
        <div key={i} style={{ marginBottom: 6, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: c.text, fontSize: 11, width: 140, textAlign: "right", flexShrink: 0 }}>
            {f.feature}
          </span>
          <div style={{ flex: 1, height: 18, background: "#1e2d45", borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              width: `${(f.importance / max) * 100}%`, height: "100%",
              background: `linear-gradient(90deg, ${c.accent2}, ${c.accent})`,
              borderRadius: 3, minWidth: 2,
            }} />
          </div>
          <span style={{ color: c.muted, fontSize: 10, width: 45, textAlign: "right" }}>
            {f.importance.toFixed(3)}
          </span>
        </div>
      ))}
      <p style={{ color: c.muted, fontSize: 10, marginTop: 10, fontStyle: "italic" }}>
        Method: {data.method}
      </p>
    </div>
  );
}

function ReferenceCard({ ref: r, index }) {
  const [open, setOpen] = useState(false);
  const relColor = r.relevance_pct >= 80 ? c.success : r.relevance_pct >= 60 ? c.accent : c.muted;
  return (
    <div style={{ border: `1px solid ${c.border}`, borderRadius: 8, marginBottom: 8, overflow: "hidden" }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", padding: "10px 14px", background: c.surface2, border: "none",
        cursor: "pointer", display: "flex", alignItems: "center", gap: 10,
      }}>
        <span style={{ color: c.accent, fontWeight: 800, fontSize: 13 }}>[{index + 1}]</span>
        <span style={{ color: c.text, fontSize: 12, flex: 1, textAlign: "left" }}>{r.filename}</span>
        <span style={{ color: c.muted, fontSize: 11 }}>Chunk {r.chunk_index}</span>
        <span style={{ color: relColor, fontWeight: 700, fontSize: 12 }}>{r.relevance_pct}%</span>
        <span style={{ color: c.muted, fontSize: 14 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{ padding: "12px 14px", background: c.bg }}>
          <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
            {pill(relColor, `${r.relevance_pct}% relevant`)}
            {pill(c.muted, `chunk ${r.chunk_index} / ${r.total_chunks}`)}
          </div>
          <p style={{ color: c.text, fontSize: 12, lineHeight: 1.7, margin: 0 }}>{r.text}</p>
        </div>
      )}
    </div>
  );
}

function DocumentManager({ onRefresh }) {
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileRef = useRef();

  const loadDocs = async () => {
    try {
      const r = await fetch(`${API}/documents`);
      setDocs(await r.json());
    } catch { setDocs([]); }
  };

  useEffect(() => { loadDocs(); }, []);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch(`${API}/documents/upload`, { method: "POST", body: fd });
      const data = await r.json();
      setUploadStatus({ ok: true, msg: `"${data.filename}" queued for indexing` });
      setTimeout(() => { loadDocs(); onRefresh?.(); }, 3000);
    } catch (err) {
      setUploadStatus({ ok: false, msg: "Upload failed" });
    }
    setUploading(false);
    fileRef.current.value = "";
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this document?")) return;
    await fetch(`${API}/documents/${id}`, { method: "DELETE" });
    loadDocs();
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <h3 style={{ color: c.text, margin: 0, fontSize: 14, fontWeight: 700 }}>
          📂 Indexed Documents ({docs.length})
        </h3>
        <button onClick={loadDocs} style={{
          background: "none", border: `1px solid ${c.border}`, color: c.muted,
          borderRadius: 6, padding: "3px 10px", cursor: "pointer", fontSize: 11,
        }}>↻ Refresh</button>
      </div>

      <label style={{
        display: "flex", alignItems: "center", gap: 8, padding: "10px 14px",
        border: `2px dashed ${c.accent}55`, borderRadius: 8, cursor: "pointer",
        background: `${c.accent}08`, marginBottom: 12, transition: "all 0.2s",
      }}>
        <span style={{ fontSize: 20 }}>📎</span>
        <span style={{ color: c.accent, fontSize: 12, fontWeight: 600 }}>
          {uploading ? "Uploading…" : "Upload Regulatory Document"}
        </span>
        <span style={{ color: c.muted, fontSize: 11 }}>PDF, DOCX, TXT, MD</span>
        <input ref={fileRef} type="file" accept=".pdf,.docx,.txt,.md"
          onChange={handleUpload} style={{ display: "none" }} disabled={uploading} />
      </label>

      {uploadStatus && (
        <div style={{
          padding: "8px 12px", borderRadius: 6, marginBottom: 10, fontSize: 12,
          background: uploadStatus.ok ? `${c.success}22` : `${c.danger}22`,
          color: uploadStatus.ok ? c.success : c.danger,
          border: `1px solid ${uploadStatus.ok ? c.success : c.danger}44`,
        }}>{uploadStatus.msg}</div>
      )}

      {docs.length === 0 ? (
        <p style={{ color: c.muted, fontSize: 12, textAlign: "center", padding: "20px 0" }}>
          No documents indexed yet. Upload a regulatory document to begin.
        </p>
      ) : (
        docs.map(doc => (
          <div key={doc.doc_id} style={{
            display: "flex", alignItems: "center", gap: 8, padding: "8px 12px",
            background: c.surface2, borderRadius: 6, marginBottom: 6, border: `1px solid ${c.border}`,
          }}>
            <span style={{ fontSize: 16 }}>📄</span>
            <div style={{ flex: 1 }}>
              <p style={{ color: c.text, fontSize: 12, margin: 0, fontWeight: 600 }}>{doc.filename}</p>
              <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>{doc.chunks} chunks</p>
            </div>
            <button onClick={() => handleDelete(doc.doc_id)} style={{
              background: `${c.danger}22`, border: `1px solid ${c.danger}44`,
              color: c.danger, borderRadius: 4, padding: "2px 8px", cursor: "pointer", fontSize: 10,
            }}>✕</button>
          </div>
        ))
      )}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery]   = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError]   = useState(null);
  const [health, setHealth] = useState(null);
  const [activeTab, setTab] = useState("answer");  // answer | trace | shap | refs
  const [topK, setTopK]     = useState(5);
  const [mode, setMode]     = useState("full");
  const [history, setHistory] = useState([]);
  const textRef = useRef();

  // Check API health on load
  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(setHealth).catch(() =>
      setHealth({ api: "unreachable" })
    );
  }, []);

  const handleQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK, mode }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setResult(data);
      setTab("answer");
      setHistory(h => [{ query, ts: new Date().toLocaleTimeString() }, ...h.slice(0, 9)]);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleQuery();
  };

  const statusColor = (s) => s === "ok" || s?.startsWith("ok") ? c.success : c.danger;

  const tabs = [
    { id: "answer", label: "📋 Answer" },
    { id: "trace",  label: `🤖 Agent Trace (${result?.agent_trace?.length || 0})` },
    { id: "shap",   label: "📊 SHAP Analysis" },
    { id: "refs",   label: `📎 References (${result?.references?.length || 0})` },
  ];

  return (
    <div style={{
      minHeight: "100vh", background: c.bg,
      fontFamily: "'IBM Plex Mono', 'Fira Code', 'Courier New', monospace",
      color: c.text,
    }}>
      {/* ── Header ── */}
      <div style={{
        background: c.surface, borderBottom: `1px solid ${c.border}`,
        padding: "0 24px", display: "flex", alignItems: "center", gap: 16, height: 56,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>⚖️</span>
          <div>
            <span style={{ color: c.accent, fontWeight: 800, fontSize: 15, letterSpacing: 1 }}>
              REG·ANALYST
            </span>
            <span style={{ color: c.muted, fontSize: 10, display: "block", letterSpacing: 2 }}>
              ON-PREMISE AI · MULTI-AGENT
            </span>
          </div>
        </div>
        <div style={{ flex: 1 }} />
        {health && (
          <div style={{ display: "flex", gap: 12 }}>
            {[
              ["API", health.api],
              ["Ollama", health.ollama],
              ["ChromaDB", health.chroma],
            ].map(([label, val]) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: statusColor(val),
                  boxShadow: `0 0 6px ${statusColor(val)}` }} />
                <span style={{ color: c.muted, fontSize: 10 }}>{label}</span>
              </div>
            ))}
            <span style={{ color: c.muted, fontSize: 10 }}>
              {health.documents_indexed ?? 0} docs
            </span>
          </div>
        )}
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 56px)" }}>

        {/* ── Sidebar ── */}
        <div style={{
          width: 300, borderRight: `1px solid ${c.border}`,
          background: c.surface, padding: 16, overflowY: "auto", flexShrink: 0,
        }}>
          <DocumentManager />

          {history.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <h3 style={{ color: c.text, fontSize: 12, fontWeight: 700, marginBottom: 10 }}>
                🕘 Recent Queries
              </h3>
              {history.map((h, i) => (
                <button key={i} onClick={() => setQuery(h.query)} style={{
                  width: "100%", textAlign: "left", background: "none", border: "none",
                  cursor: "pointer", padding: "6px 0", borderBottom: `1px solid ${c.border}`,
                }}>
                  <p style={{ color: c.text, fontSize: 11, margin: 0, whiteSpace: "nowrap",
                    overflow: "hidden", textOverflow: "ellipsis" }}>{h.query}</p>
                  <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>{h.ts}</p>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ── Main ── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Query bar */}
          <div style={{ padding: "16px 20px", borderBottom: `1px solid ${c.border}`, background: c.surface }}>
            <textarea
              ref={textRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask a regulatory question… e.g. 'What are the capital requirements under Basel III for Tier 1 capital?'"
              rows={3}
              style={{
                width: "100%", background: c.bg, border: `1px solid ${c.border}`,
                borderRadius: 8, padding: "12px 14px", color: c.text,
                fontSize: 13, resize: "none", outline: "none", fontFamily: "inherit",
                boxSizing: "border-box",
                transition: "border-color 0.2s",
              }}
              onFocus={e => e.target.style.borderColor = c.accent}
              onBlur={e => e.target.style.borderColor = c.border}
            />
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10 }}>
              <select value={mode} onChange={e => setMode(e.target.value)} style={{
                background: c.surface2, border: `1px solid ${c.border}`, color: c.text,
                borderRadius: 6, padding: "6px 10px", fontSize: 11, cursor: "pointer",
              }}>
                <option value="full">Full Analysis</option>
                <option value="quick">Quick Answer</option>
                <option value="compare">Compare</option>
              </select>

              <select value={topK} onChange={e => setTopK(+e.target.value)} style={{
                background: c.surface2, border: `1px solid ${c.border}`, color: c.text,
                borderRadius: 6, padding: "6px 10px", fontSize: 11, cursor: "pointer",
              }}>
                {[3, 5, 8, 10].map(n => (
                  <option key={n} value={n}>Top {n} chunks</option>
                ))}
              </select>

              <span style={{ color: c.muted, fontSize: 10, flex: 1 }}>Ctrl+Enter to submit</span>

              <button onClick={handleQuery} disabled={loading || !query.trim()} style={{
                background: loading ? c.muted : c.accent,
                color: "#000", border: "none", borderRadius: 6,
                padding: "8px 20px", fontSize: 13, fontWeight: 800,
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.2s",
                boxShadow: loading ? "none" : `0 0 16px ${c.accent}55`,
              }}>
                {loading ? "⏳ Analysing…" : "Analyse ▶"}
              </button>
            </div>
          </div>

          {/* Results area */}
          <div style={{ flex: 1, overflowY: "auto", padding: "0 20px 20px" }}>

            {error && (
              <div style={{ margin: "16px 0", padding: 14, background: `${c.danger}22`,
                border: `1px solid ${c.danger}44`, borderRadius: 8, color: c.danger, fontSize: 13 }}>
                ⚠️ {error}
              </div>
            )}

            {!result && !loading && !error && (
              <div style={{ textAlign: "center", padding: "60px 0", color: c.muted }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>⚖️</div>
                <p style={{ fontSize: 14 }}>Upload regulatory documents, then ask questions.</p>
                <p style={{ fontSize: 11 }}>
                  Powered by Llama 3 + ChromaDB + SHAP — fully on-premise
                </p>
              </div>
            )}

            {result && (
              <div style={{ marginTop: 16 }}>
                {/* Result header */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
                  {pill(c.accent, result.mode)}
                  {pill(c.accent2, `${result.references.length} sources`)}
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ color: c.muted, fontSize: 11 }}>Confidence:</span>
                    <div style={{ width: 120 }}>
                      <ConfidenceMeter value={result.confidence} />
                    </div>
                  </div>
                  <span style={{ color: c.muted, fontSize: 10 }}>Session: {result.session_id?.slice(0, 8)}</span>
                </div>

                {/* Tabs */}
                <div style={{ display: "flex", gap: 2, marginBottom: 16,
                  borderBottom: `1px solid ${c.border}`, paddingBottom: 0 }}>
                  {tabs.map(t => (
                    <button key={t.id} onClick={() => setTab(t.id)} style={{
                      background: activeTab === t.id ? c.accent : "none",
                      color: activeTab === t.id ? "#000" : c.muted,
                      border: "none", borderRadius: "6px 6px 0 0",
                      padding: "8px 14px", fontSize: 11, fontWeight: activeTab === t.id ? 700 : 400,
                      cursor: "pointer", transition: "all 0.2s",
                    }}>{t.label}</button>
                  ))}
                </div>

                {/* Tab content */}
                <div style={{ background: c.surface, borderRadius: 8, padding: 18,
                  border: `1px solid ${c.border}` }}>

                  {activeTab === "answer" && (
                    <div>
                      <pre style={{
                        color: c.text, fontSize: 13, lineHeight: 1.8,
                        whiteSpace: "pre-wrap", wordBreak: "break-word", margin: 0,
                        fontFamily: "'IBM Plex Serif', Georgia, serif",
                      }}>{result.answer}</pre>
                    </div>
                  )}

                  {activeTab === "trace" && (
                    <div>
                      <p style={{ color: c.muted, fontSize: 11, marginBottom: 14 }}>
                        Each agent's reasoning step is shown below. Expand to see inputs and outputs.
                      </p>
                      {result.agent_trace.map((step, i) => (
                        <AgentStep key={i} step={step} index={i} />
                      ))}
                    </div>
                  )}

                  {activeTab === "shap" && (
                    <div>
                      <h4 style={{ color: c.accent, margin: "0 0 12px", fontSize: 13 }}>
                        📊 SHAP Feature Importance
                      </h4>
                      <ShapChart data={result.shap_analysis} />
                      {result.shap_analysis?.chunk_importances?.length > 0 && (
                        <div style={{ marginTop: 20 }}>
                          <h4 style={{ color: c.text, margin: "0 0 10px", fontSize: 12 }}>
                            Per-Chunk Attribution
                          </h4>
                          {result.shap_analysis.chunk_importances.map((ci, i) => (
                            <div key={i} style={{ padding: "8px 12px", background: c.surface2,
                              borderRadius: 6, marginBottom: 6, border: `1px solid ${c.border}` }}>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                <span style={{ color: c.text, fontSize: 11 }}>Chunk {ci.chunk_index}</span>
                                <div style={{ display: "flex", gap: 8 }}>
                                  <span style={{ color: c.accent2, fontSize: 10 }}>
                                    sim: {ci.similarity.toFixed(3)}
                                  </span>
                                  <span style={{ color: c.accent, fontSize: 10 }}>
                                    SHAP: {ci.shap_importance.toFixed(3)}
                                  </span>
                                </div>
                              </div>
                              <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 4 }}>
                                {ci.top_contributing_words?.map((w, j) => (
                                  <span key={j} style={{
                                    background: w.shap > 0 ? `${c.success}22` : `${c.danger}22`,
                                    color: w.shap > 0 ? c.success : c.danger,
                                    borderRadius: 4, padding: "1px 6px", fontSize: 10,
                                    border: `1px solid ${w.shap > 0 ? c.success : c.danger}33`,
                                  }}>{w.word} ({w.shap > 0 ? "+" : ""}{w.shap.toFixed(3)})</span>
                                ))}
                              </div>
                              <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>{ci.preview}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {activeTab === "refs" && (
                    <div>
                      <p style={{ color: c.muted, fontSize: 11, marginBottom: 12 }}>
                        Chunks retrieved from ChromaDB, ranked by semantic similarity.
                      </p>
                      {result.references.map((r, i) => (
                        <ReferenceCard key={i} ref={r} index={i} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
