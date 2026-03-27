import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8000";
const MAX_FILE_MB = 50;

// ── Theme ─────────────────────────────────────────────────────────────────────
const c = {
  bg:       "#0B0F1A",
  surface:  "#111827",
  surface2: "#1a2235",
  border:   "#1e2d45",
  accent:   "#F59E0B",
  accent2:  "#3B82F6",
  success:  "#10B981",
  danger:   "#EF4444",
  warn:     "#F97316",
  text:     "#E2E8F0",
  muted:    "#64748B",
  purple:   "#A78BFA",
  pink:     "#F472B6",
};

// ── Fetch with timeout ────────────────────────────────────────────────────────
async function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

// ── Reusable UI primitives ────────────────────────────────────────────────────
function Pill({ color, text }) {
  return (
    <span style={{
      background: color + "22", color, border: `1px solid ${color}44`,
      borderRadius: 20, padding: "2px 10px", fontSize: 11, fontWeight: 700,
      letterSpacing: 0.5, textTransform: "uppercase", whiteSpace: "nowrap",
    }}>{text}</span>
  );
}

function ConfidenceMeter({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? c.success : pct >= 50 ? c.warn : c.danger;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div style={{ flex: 1, height: 6, background: "#1e2d45", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%", background: color, borderRadius: 4,
          transition: "width 0.8s ease", boxShadow: `0 0 8px ${color}88`,
        }} />
      </div>
      <span style={{ color, fontWeight: 700, fontSize: 13, minWidth: 40 }}>{pct}%</span>
    </div>
  );
}

function Skeleton({ lines = 4 }) {
  return (
    <div style={{ padding: 20 }}>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} style={{
          height: 14, background: c.border, borderRadius: 4,
          marginBottom: 12, width: i === lines - 1 ? "60%" : "100%",
          animation: "pulse 1.4s ease-in-out infinite",
          animationDelay: `${i * 0.1}s`,
        }} />
      ))}
      <style>{`@keyframes pulse { 0%,100%{opacity:0.4} 50%{opacity:1} }`}</style>
    </div>
  );
}

// ── Agent Trace Step ──────────────────────────────────────────────────────────
function AgentStep({ step, index }) {
  const [open, setOpen] = useState(false);
  const colors = [c.accent2, c.accent, c.success, c.purple, c.pink];
  const col = colors[index % colors.length];
  return (
    <div style={{ border: `1px solid ${col}33`, borderRadius: 8, overflow: "hidden", marginBottom: 8 }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", background: open ? `${col}18` : `${col}0a`, padding: "10px 14px",
        border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 10,
      }}>
        <span style={{
          width: 22, height: 22, borderRadius: "50%", background: col,
          color: "#000", fontSize: 11, fontWeight: 800,
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}>{index + 1}</span>
        <span style={{ color: col, fontWeight: 700, fontSize: 13, flex: 1, textAlign: "left" }}>
          {step.agent}
        </span>
        <span style={{ color: c.muted, fontSize: 10 }}>{step.model}</span>
        <span style={{ color: c.muted, fontSize: 16 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{ padding: "12px 14px", background: c.surface }}>
          <p style={{ color: c.muted, fontSize: 12, marginBottom: 8, fontStyle: "italic" }}>
            {step.purpose}
          </p>
          <pre style={{
            color: c.text, fontSize: 11, background: c.bg, padding: 12, borderRadius: 6,
            overflow: "auto", maxHeight: 240, margin: 0, whiteSpace: "pre-wrap",
          }}>
            {JSON.stringify(step.output, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ── SHAP Chart ────────────────────────────────────────────────────────────────
function ShapChart({ data }) {
  if (!data?.top_features?.length) {
    return <p style={{ color: c.muted, fontSize: 12 }}>No SHAP data available.</p>;
  }
  const max = Math.max(...data.top_features.map(f => f.importance), 0.001);
  return (
    <div>
      <p style={{ color: c.muted, fontSize: 11, marginBottom: 12, fontStyle: "italic" }}>
        {data.explanation}
      </p>
      <h5 style={{ color: c.text, fontSize: 12, marginBottom: 10, marginTop: 0 }}>
        Top Influential Tokens (mean |SHAP|)
      </h5>
      {data.top_features.slice(0, 12).map((f, i) => (
        <div key={i} style={{ marginBottom: 6, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: c.text, fontSize: 11, width: 150, textAlign: "right", flexShrink: 0 }}>
            {f.feature}
          </span>
          <div style={{ flex: 1, height: 18, background: "#1e2d45", borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              width: `${(f.importance / max) * 100}%`, height: "100%",
              background: `linear-gradient(90deg, ${c.accent2}, ${c.accent})`,
              borderRadius: 3, minWidth: 2,
            }} />
          </div>
          <span style={{ color: c.muted, fontSize: 10, width: 48, textAlign: "right" }}>
            {f.importance.toFixed(3)}
          </span>
        </div>
      ))}
      <p style={{ color: c.muted, fontSize: 10, marginTop: 14, fontStyle: "italic" }}>
        Method: {data.method}
      </p>
      {data.chunk_importances?.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h5 style={{ color: c.text, fontSize: 12, marginBottom: 10, marginTop: 0 }}>
            Per-Chunk Attribution
          </h5>
          {data.chunk_importances.map((ci, i) => (
            <div key={i} style={{
              padding: "10px 12px", background: c.surface2, borderRadius: 6,
              marginBottom: 8, border: `1px solid ${c.border}`,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ color: c.text, fontSize: 11, fontWeight: 600 }}>Chunk {ci.chunk_index}</span>
                <div style={{ display: "flex", gap: 10 }}>
                  <span style={{ color: c.accent2, fontSize: 10 }}>cosine: {ci.similarity.toFixed(3)}</span>
                  <span style={{ color: c.accent, fontSize: 10 }}>SHAP: {ci.shap_importance.toFixed(3)}</span>
                </div>
              </div>
              {ci.top_contributing_words?.length > 0 && (
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 6 }}>
                  {ci.top_contributing_words.map((w, j) => (
                    <span key={j} style={{
                      background: w.shap > 0 ? `${c.success}22` : `${c.danger}22`,
                      color: w.shap > 0 ? c.success : c.danger,
                      borderRadius: 4, padding: "1px 7px", fontSize: 10,
                      border: `1px solid ${w.shap > 0 ? c.success : c.danger}33`,
                    }}>
                      {w.word} ({w.shap > 0 ? "+" : ""}{w.shap.toFixed(3)})
                    </span>
                  ))}
                </div>
              )}
              <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>{ci.preview}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Reference Card ────────────────────────────────────────────────────────────
function ReferenceCard({ refData, index }) {
  const [open, setOpen] = useState(false);
  const relColor = refData.relevance_pct >= 80 ? c.success
    : refData.relevance_pct >= 60 ? c.warn : c.muted;
  return (
    <div style={{ border: `1px solid ${c.border}`, borderRadius: 8, marginBottom: 8, overflow: "hidden" }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", padding: "10px 14px", background: c.surface2, border: "none",
        cursor: "pointer", display: "flex", alignItems: "center", gap: 10,
      }}>
        <span style={{ color: c.accent, fontWeight: 800, fontSize: 14, flexShrink: 0 }}>[{index + 1}]</span>
        <span style={{ color: c.text, fontSize: 12, flex: 1, textAlign: "left" }}>{refData.filename}</span>
        <span style={{ color: c.muted, fontSize: 11 }}>chunk {refData.chunk_index}/{refData.total_chunks}</span>
        {refData.page_number > 0 && (
          <span style={{ color: c.purple, fontSize: 10 }}>p.{refData.page_number}</span>
        )}
        <span style={{ color: relColor, fontWeight: 700, fontSize: 12, minWidth: 40 }}>{refData.relevance_pct}%</span>
        <span style={{ color: c.muted, fontSize: 14 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{ padding: "14px", background: c.bg }}>
          <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
            <Pill color={relColor} text={`${refData.relevance_pct}% relevant`} />
            <Pill color={c.muted} text={`chunk ${refData.chunk_index} of ${refData.total_chunks}`} />
            <Pill color={c.accent2} text={`similarity ${refData.similarity}`} />
            {refData.page_number > 0 && (
              <Pill color={c.purple} text={`📄 ${refData.page_label || "Page " + refData.page_number}`} />
            )}
            {refData.clause_refs && (
              <Pill color={c.pink} text={`📌 ${refData.clause_refs}`} />
            )}
          </div>
          {(refData.page_number > 0 || refData.clause_refs) && (
            <div style={{
              padding: "6px 10px", background: c.surface2, borderRadius: 6,
              marginBottom: 10, fontSize: 11, display: "flex", gap: 20, flexWrap: "wrap",
            }}>
              {refData.page_number > 0 && (
                <span style={{ color: c.purple }}>
                  📄 <strong>Page:</strong> {refData.page_number}
                </span>
              )}
              {refData.clause_refs && (
                <span style={{ color: c.pink }}>
                  📌 <strong>Clauses:</strong> {refData.clause_refs}
                </span>
              )}
            </div>
          )}
          <p style={{ color: c.text, fontSize: 12, lineHeight: 1.8, margin: 0, whiteSpace: "pre-wrap" }}>
            {refData.text}
          </p>
        </div>
      )}
    </div>
  );
}

// ── Document Manager ──────────────────────────────────────────────────────────
function DocumentManager() {
  const [docs, setDocs]           = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null);
  const [polling, setPolling]     = useState(null);
  const fileRef = useRef();

  const loadDocs = useCallback(async () => {
    try {
      const r = await fetchWithTimeout(`${API}/documents`, {}, 5000);
      if (r.ok) setDocs(await r.json());
    } catch {}
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  // Poll indexing status until ready or error
  useEffect(() => {
    if (!polling) return;
    const interval = setInterval(async () => {
      try {
        const r = await fetchWithTimeout(`${API}/documents/${polling}/status`, {}, 5000);
        const { status } = await r.json();
        if (status === "ready") {
          clearInterval(interval);
          setPolling(null);
          setUploadMsg({ ok: true, text: "Document indexed successfully ✓" });
          loadDocs();
        } else if (status?.startsWith("error:")) {
          clearInterval(interval);
          setPolling(null);
          setUploadMsg({ ok: false, text: `Indexing failed: ${status.slice(6)}` });
        }
      } catch { clearInterval(interval); setPolling(null); }
    }, 2000);
    return () => clearInterval(interval);
  }, [polling, loadDocs]);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setUploadMsg({ ok: false, text: `File too large (max ${MAX_FILE_MB}MB)` });
      return;
    }
    setUploading(true);
    setUploadMsg({ ok: true, text: `Uploading "${file.name}"…` });
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetchWithTimeout(`${API}/documents/upload`, { method: "POST", body: fd }, 60000);
      if (!r.ok) { const err = await r.json(); throw new Error(err.detail || r.statusText); }
      const data = await r.json();
      setUploadMsg({ ok: true, text: `"${data.filename}" uploaded — indexing…` });
      setPolling(data.doc_id);
    } catch (err) {
      setUploadMsg({ ok: false, text: `Upload failed: ${err.message}` });
    }
    setUploading(false);
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleDelete = async (doc_id, filename) => {
    if (!window.confirm(`Delete "${filename}" from the index?`)) return;
    try {
      await fetchWithTimeout(`${API}/documents/${doc_id}`, { method: "DELETE" }, 10000);
      loadDocs();
    } catch (err) { alert(`Delete failed: ${err.message}`); }
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
        <h3 style={{ color: c.text, margin: 0, fontSize: 13, fontWeight: 700 }}>
          📂 Documents ({docs.length})
        </h3>
        <button onClick={loadDocs} style={{
          background: "none", border: `1px solid ${c.border}`, color: c.muted,
          borderRadius: 5, padding: "2px 8px", cursor: "pointer", fontSize: 10,
        }}>↻</button>
      </div>

      <label style={{
        display: "flex", alignItems: "center", gap: 8, padding: "10px 12px",
        border: `2px dashed ${uploading ? c.muted : c.accent}55`,
        borderRadius: 8, cursor: uploading ? "not-allowed" : "pointer",
        background: `${c.accent}08`, marginBottom: 10,
      }}>
        <span style={{ fontSize: 18 }}>📎</span>
        <div>
          <p style={{ color: c.accent, fontSize: 12, fontWeight: 600, margin: 0 }}>
            {uploading ? "Uploading…" : "Upload Document"}
          </p>
          <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>
            PDF · DOCX · TXT · MD · max {MAX_FILE_MB}MB
          </p>
        </div>
        <input ref={fileRef} type="file" accept=".pdf,.docx,.txt,.md"
          onChange={handleUpload} disabled={uploading} style={{ display: "none" }} />
      </label>

      {uploadMsg && (
        <div style={{
          padding: "7px 10px", borderRadius: 6, marginBottom: 10, fontSize: 11,
          background: uploadMsg.ok ? `${c.success}22` : `${c.danger}22`,
          color: uploadMsg.ok ? c.success : c.danger,
          border: `1px solid ${uploadMsg.ok ? c.success : c.danger}44`,
          display: "flex", alignItems: "center", gap: 6,
        }}>
          {polling && <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⏳</span>}
          {uploadMsg.text}
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </div>
      )}

      {docs.length === 0 ? (
        <p style={{ color: c.muted, fontSize: 11, textAlign: "center", padding: "16px 0" }}>
          No documents indexed. Upload one to begin.
        </p>
      ) : (
        docs.map(doc => (
          <div key={doc.doc_id} style={{
            display: "flex", alignItems: "center", gap: 8, padding: "8px 10px",
            background: c.surface2, borderRadius: 6, marginBottom: 5,
            border: `1px solid ${doc.index_status === "indexing" ? c.warn : c.border}`,
          }}>
            <span style={{ fontSize: 15 }}>📄</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{
                color: c.text, fontSize: 11, margin: 0, fontWeight: 600,
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
              }}>{doc.filename}</p>
              <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>
                {doc.chunks} chunks
                {doc.index_status === "indexing" && <span style={{ color: c.warn }}> · indexing…</span>}
              </p>
            </div>
            <button onClick={() => handleDelete(doc.doc_id, doc.filename)} style={{
              background: `${c.danger}22`, border: `1px solid ${c.danger}44`,
              color: c.danger, borderRadius: 4, padding: "2px 7px",
              cursor: "pointer", fontSize: 10, flexShrink: 0,
            }}>✕</button>
          </div>
        ))
      )}
    </div>
  );
}

// ── History Panel ─────────────────────────────────────────────────────────────
function HistoryPanel({ onRestore }) {
  const [msgs, setMsgs]         = useState([]);
  const [search, setSearch]     = useState("");
  const [open, setOpen]         = useState(false);
  const [selected, setSelected] = useState(null);
  const [stats, setStats]       = useState(null);

  const loadHistory = useCallback(async () => {
    try {
      const r = await fetchWithTimeout(`${API}/history?limit=30`, {}, 5000);
      if (r.ok) setMsgs(await r.json());
    } catch {}
  }, []);

  const loadStats = useCallback(async () => {
    try {
      const r = await fetchWithTimeout(`${API}/history/stats`, {}, 5000);
      if (r.ok) setStats(await r.json());
    } catch {}
  }, []);

  useEffect(() => {
    if (open) { loadHistory(); loadStats(); }
  }, [open, loadHistory, loadStats]);

  const handleSearch = async (val) => {
    setSearch(val);
    if (val.length < 2) { loadHistory(); return; }
    try {
      const r = await fetchWithTimeout(
        `${API}/history/search?q=${encodeURIComponent(val)}`, {}, 5000
      );
      if (r.ok) setMsgs(await r.json());
    } catch {}
  };

  const handleDeleteSession = async (session_id, e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this session from history?")) return;
    try {
      await fetchWithTimeout(
        `${API}/history/sessions/${session_id}`,
        { method: "DELETE" }, 5000
      );
      loadHistory();
      loadStats();
      if (selected?.session_id === session_id) setSelected(null);
    } catch {}
  };

  const confColor = (conf) =>
    conf >= 0.8 ? c.success : conf >= 0.5 ? c.warn : c.danger;

  return (
    <div style={{ marginTop: 16 }}>
      {/* Toggle button */}
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", background: open ? `${c.accent2}22` : "none",
        border: `1px solid ${c.border}`, color: c.text,
        borderRadius: 6, padding: "8px 12px", cursor: "pointer",
        display: "flex", alignItems: "center", gap: 8, fontSize: 12,
      }}>
        <span>🗂️</span>
        <span style={{ flex: 1, textAlign: "left", fontWeight: 600 }}>Query History</span>
        {stats && (
          <span style={{ color: c.muted, fontSize: 10 }}>
            {stats.total_messages ?? 0} saved
          </span>
        )}
        <span style={{ color: c.muted }}>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div style={{ marginTop: 8 }}>
          {/* Stats bar */}
          {stats && stats.total_messages > 0 && (
            <div style={{
              display: "flex", gap: 12, padding: "6px 10px",
              background: c.surface2, borderRadius: 6, marginBottom: 8,
              flexWrap: "wrap",
            }}>
              <span style={{ color: c.muted, fontSize: 10 }}>
                💬 <strong style={{ color: c.text }}>{stats.total_messages}</strong> queries
              </span>
              <span style={{ color: c.muted, fontSize: 10 }}>
                🗂️ <strong style={{ color: c.text }}>{stats.total_sessions}</strong> sessions
              </span>
              {stats.avg_confidence && (
                <span style={{ color: c.muted, fontSize: 10 }}>
                  ⭐ <strong style={{ color: confColor(stats.avg_confidence) }}>
                    {Math.round(stats.avg_confidence * 100)}%
                  </strong> avg conf
                </span>
              )}
            </div>
          )}

          {/* Search */}
          <input
            value={search}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search past queries…"
            style={{
              width: "100%", background: c.bg, border: `1px solid ${c.border}`,
              borderRadius: 6, padding: "6px 10px", color: c.text, fontSize: 11,
              marginBottom: 8, boxSizing: "border-box", outline: "none",
            }}
            onFocus={e => (e.target.style.borderColor = c.accent2)}
            onBlur={e => (e.target.style.borderColor = c.border)}
          />

          {msgs.length === 0 ? (
            <p style={{ color: c.muted, fontSize: 11, textAlign: "center", padding: "12px 0" }}>
              {search ? "No results found" : "No history yet"}
            </p>
          ) : (
            <div style={{ maxHeight: 380, overflowY: "auto" }}>
              {msgs.map((msg) => (
                <div
                  key={msg.id}
                  onClick={() => setSelected(selected?.id === msg.id ? null : msg)}
                  style={{
                    padding: "8px 10px",
                    background: selected?.id === msg.id ? `${c.accent2}22` : c.surface2,
                    borderRadius: 6, marginBottom: 5, cursor: "pointer",
                    border: `1px solid ${selected?.id === msg.id ? c.accent2 : c.border}`,
                    transition: "all 0.15s",
                  }}
                >
                  {/* Message header */}
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{
                        color: c.text, fontSize: 11, margin: 0, fontWeight: 600,
                        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                      }}>{msg.query}</p>
                      <div style={{ display: "flex", gap: 8, marginTop: 3, flexWrap: "wrap" }}>
                        <span style={{ color: c.muted, fontSize: 9 }}>
                          {new Date(msg.timestamp).toLocaleString()}
                        </span>
                        <span style={{
                          color: confColor(msg.confidence), fontSize: 9, fontWeight: 700,
                        }}>
                          {Math.round(msg.confidence * 100)}% conf
                        </span>
                        <span style={{ color: c.muted, fontSize: 9 }}>
                          {msg.references?.length ?? 0} src
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDeleteSession(msg.session_id, e)}
                      style={{
                        background: "none", border: "none", color: c.muted,
                        cursor: "pointer", fontSize: 11, padding: "0 2px", flexShrink: 0,
                      }}
                      title="Delete session"
                    >✕</button>
                  </div>

                  {/* Expanded detail */}
                  {selected?.id === msg.id && (
                    <div style={{
                      marginTop: 10, borderTop: `1px solid ${c.border}`, paddingTop: 10,
                    }}>
                      <p style={{ color: c.muted, fontSize: 10, marginBottom: 4, fontWeight: 700 }}>
                        ANSWER PREVIEW:
                      </p>
                      <p style={{
                        color: c.text, fontSize: 11, lineHeight: 1.6,
                        maxHeight: 160, overflowY: "auto", margin: 0,
                        whiteSpace: "pre-wrap", wordBreak: "break-word",
                      }}>
                        {msg.answer?.slice(0, 600)}{msg.answer?.length > 600 ? "…" : ""}
                      </p>

                      {/* References summary */}
                      {msg.references?.length > 0 && (
                        <div style={{ marginTop: 8 }}>
                          <p style={{ color: c.muted, fontSize: 10, fontWeight: 700, marginBottom: 4 }}>
                            SOURCES:
                          </p>
                          {msg.references.slice(0, 3).map((ref, i) => (
                            <div key={i} style={{ fontSize: 10, color: c.muted, marginBottom: 2 }}>
                              [{i + 1}] {ref.filename}
                              {ref.page_number > 0 && ` · p.${ref.page_number}`}
                              {ref.clause_refs && ` · ${ref.clause_refs.slice(0, 40)}`}
                              <span style={{ color: c.accent2 }}> {ref.relevance_pct}%</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Action buttons */}
                      <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
                        <button
                          onClick={(e) => { e.stopPropagation(); onRestore(msg); }}
                          style={{
                            background: `${c.accent}22`, border: `1px solid ${c.accent}44`,
                            color: c.accent, borderRadius: 4, padding: "4px 10px",
                            cursor: "pointer", fontSize: 10, fontWeight: 600,
                          }}
                        >
                          ↩ Restore to view
                        </button>
                        <span style={{ color: c.muted, fontSize: 9, alignSelf: "center" }}>
                          ID: {msg.id}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Answer Renderer ───────────────────────────────────────────────────────────
function AnswerRenderer({ answer }) {
  return (
    <div style={{ color: c.text, fontSize: 13, lineHeight: 1.85,
      fontFamily: "'IBM Plex Serif', Georgia, serif" }}>
      {answer.split("\n").map((line, i) => {
        if (line.trim() === "---") return (
          <hr key={i} style={{ border: "none", borderTop: `1px solid ${c.border}`, margin: "16px 0" }} />
        );
        if (line.startsWith("**References Used:**")) return (
          <p key={i} style={{ color: c.accent, fontWeight: 700, fontSize: 12,
            letterSpacing: 1, marginBottom: 8, marginTop: 4, fontFamily: "monospace" }}>
            📎 REFERENCES USED
          </p>
        );
        if (line.match(/^\[\d+\]\s\*\*/)) {
          const match = line.match(/^\[(\d+)\]\s\*\*(.+?)\*\*(.+)?/);
          if (match) {
            const [, num, filename, rest = ""] = match;
            const relMatch = rest.match(/Relevance:\s([\d.]+)%/);
            const pageMatch = rest.match(/📄\s(Page\s\d+)/);
            const clauseMatch = rest.match(/📌\s([^|]+)/);
            const rel = relMatch ? parseFloat(relMatch[1]) : 0;
            const relColor = rel >= 70 ? c.success : rel >= 50 ? c.warn : c.muted;
            return (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 8, padding: "6px 10px",
                background: c.surface2, borderRadius: 6, marginBottom: 5,
                border: `1px solid ${c.border}`, flexWrap: "wrap",
              }}>
                <span style={{ color: c.accent, fontWeight: 800, fontSize: 12 }}>[{num}]</span>
                <span style={{ color: c.text, fontSize: 11, fontWeight: 600 }}>{filename}</span>
                {pageMatch && <span style={{ color: c.purple, fontSize: 10 }}>📄 {pageMatch[1]}</span>}
                {clauseMatch && <span style={{ color: c.pink, fontSize: 10 }}>📌 {clauseMatch[1].trim()}</span>}
                {rel > 0 && <span style={{ color: relColor, fontWeight: 700, fontSize: 11, marginLeft: "auto" }}>{rel}%</span>}
              </div>
            );
          }
        }
        if (line.includes("**")) {
          const parts = line.split(/\*\*(.+?)\*\*/g);
          return (
            <p key={i} style={{ marginBottom: 6, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
              {parts.map((part, j) =>
                j % 2 === 1
                  ? <strong key={j} style={{ color: c.accent2 }}>{part}</strong>
                  : part
              )}
            </p>
          );
        }
        if (line.trim() === "") return <div key={i} style={{ height: 8 }} />;
        return (
          <p key={i} style={{ marginBottom: 6, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {line}
          </p>
        );
      })}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery]     = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);
  const [health, setHealth]   = useState(null);
  const [activeTab, setTab]   = useState("answer");
  const [topK, setTopK]       = useState(5);
  const [mode, setMode]       = useState("full");
  const textRef = useRef();

  // Health polling every 30 seconds
  const pollHealth = useCallback(async () => {
    try {
      const r = await fetchWithTimeout(`${API}/health`, {}, 5000);
      if (r.ok) setHealth(await r.json());
      else setHealth({ api: "error" });
    } catch { setHealth({ api: "unreachable" }); }
  }, []);

  useEffect(() => {
    pollHealth();
    const interval = setInterval(pollHealth, 30000);
    return () => clearInterval(interval);
  }, [pollHealth]);

  const handleQuery = async () => {
    const q = query.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setTab("answer");
    try {
      const r = await fetchWithTimeout(
        `${API}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, top_k: topK, mode }),
        },
        120000,
      );
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(err.detail || "Query failed");
      }
      setResult(await r.json());
    } catch (e) {
      setError(e.name === "AbortError"
        ? "Request timed out. Try a shorter query or check Ollama is running."
        : e.message);
    }
    setLoading(false);
  };

  // Restore a saved message from history into the result view
  const handleRestore = (msg) => {
    setResult({
      session_id:   msg.session_id,
      query:        msg.query,
      answer:       msg.answer,
      confidence:   msg.confidence,
      mode:         msg.mode,
      references:   msg.references   ?? [],
      agent_trace:  msg.agent_trace  ?? [],
      shap_analysis: msg.shap_analysis ?? {},
      message_id:   msg.id,
    });
    setTab("answer");
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleQuery();
  };

  const statusColor = (s) =>
    !s ? c.muted :
    (s === "ok" || s?.startsWith("ok")) ? c.success :
    s?.startsWith("warning") ? c.warn : c.danger;

  const tabs = [
    { id: "answer", label: "📋 Answer" },
    { id: "trace",  label: `🤖 Agents (${result?.agent_trace?.length ?? 0})` },
    { id: "shap",   label: "📊 SHAP" },
    { id: "refs",   label: `📎 Refs (${result?.references?.length ?? 0})` },
  ];

  return (
    <div style={{
      minHeight: "100vh", background: c.bg,
      fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
      color: c.text,
    }}>
      {/* ── Header ── */}
      <div style={{
        background: c.surface, borderBottom: `1px solid ${c.border}`,
        padding: "0 24px", display: "flex", alignItems: "center", gap: 16, height: 54,
      }}>
        <span style={{ fontSize: 20 }}>⚖️</span>
        <div>
          <span style={{ color: c.accent, fontWeight: 800, fontSize: 14, letterSpacing: 1 }}>
            REG·ANALYST
          </span>
          <span style={{ color: c.muted, fontSize: 9, display: "block", letterSpacing: 2 }}>
            ON-PREMISE · MULTI-AGENT · FULLY LOCAL
          </span>
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
          {[
            ["API",      health?.api],
            ["Ollama",   health?.ollama],
            ["ChromaDB", health?.chroma],
          ].map(([label, val]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%",
                background: statusColor(val),
                boxShadow: `0 0 6px ${statusColor(val)}`,
              }} />
              <span style={{ color: c.muted, fontSize: 10 }}>{label}</span>
            </div>
          ))}
          {health?.saved_queries != null && (
            <span style={{ color: c.muted, fontSize: 10 }}>
              💬 {health.saved_queries} saved
            </span>
          )}
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 54px)" }}>

        {/* ── Sidebar ── */}
        <div style={{
          width: 290, borderRight: `1px solid ${c.border}`,
          background: c.surface, padding: 14, overflowY: "auto", flexShrink: 0,
        }}>
          <DocumentManager />
          <HistoryPanel onRestore={handleRestore} />
        </div>

        {/* ── Main panel ── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Query bar */}
          <div style={{ padding: "14px 20px", borderBottom: `1px solid ${c.border}`, background: c.surface }}>
            <textarea
              ref={textRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKey}
              placeholder={`Ask a regulatory question… e.g. "What are the Tier 1 capital requirements under Basel III?"`}
              rows={3}
              style={{
                width: "100%", background: c.bg, border: `1px solid ${c.border}`,
                borderRadius: 8, padding: "12px 14px", color: c.text,
                fontSize: 13, resize: "none", outline: "none", fontFamily: "inherit",
                boxSizing: "border-box", lineHeight: 1.6,
              }}
              onFocus={e => (e.target.style.borderColor = c.accent)}
              onBlur={e => (e.target.style.borderColor = c.border)}
            />
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10 }}>
              <select value={mode} onChange={e => setMode(e.target.value)} style={{
                background: c.surface2, border: `1px solid ${c.border}`, color: c.text,
                borderRadius: 6, padding: "6px 10px", fontSize: 11,
              }}>
                <option value="full">Full Analysis</option>
                <option value="quick">Quick Answer</option>
                <option value="compare">Compare</option>
              </select>

              <select value={topK} onChange={e => setTopK(+e.target.value)} style={{
                background: c.surface2, border: `1px solid ${c.border}`, color: c.text,
                borderRadius: 6, padding: "6px 10px", fontSize: 11,
              }}>
                {[3, 5, 8, 10].map(n => (
                  <option key={n} value={n}>Top {n} chunks</option>
                ))}
              </select>

              <span style={{ color: c.muted, fontSize: 10, flex: 1 }}>Ctrl+Enter to submit</span>

              <button onClick={handleQuery} disabled={loading || !query.trim()} style={{
                background: loading ? c.muted : c.accent,
                color: "#000", border: "none", borderRadius: 6,
                padding: "8px 22px", fontSize: 13, fontWeight: 800,
                cursor: (loading || !query.trim()) ? "not-allowed" : "pointer",
                transition: "all 0.2s",
                boxShadow: loading ? "none" : `0 0 16px ${c.accent}55`,
              }}>
                {loading ? "⏳ Analysing…" : "Analyse ▶"}
              </button>
            </div>
          </div>

          {/* Results area */}
          <div style={{ flex: 1, overflowY: "auto", padding: "0 20px 24px" }}>

            {error && (
              <div style={{
                margin: "16px 0", padding: 14, background: `${c.danger}22`,
                border: `1px solid ${c.danger}44`, borderRadius: 8, color: c.danger, fontSize: 12,
              }}>⚠️ {error}</div>
            )}

            {loading && !result && (
              <div style={{ marginTop: 16 }}>
                <div style={{ background: c.surface, borderRadius: 8, border: `1px solid ${c.border}` }}>
                  <Skeleton lines={6} />
                </div>
              </div>
            )}

            {!result && !loading && !error && (
              <div style={{ textAlign: "center", padding: "60px 20px", color: c.muted }}>
                <div style={{ fontSize: 48, marginBottom: 14 }}>⚖️</div>
                <p style={{ fontSize: 14, marginBottom: 6 }}>
                  Upload regulatory documents, then ask questions.
                </p>
                <p style={{ fontSize: 11 }}>
                  Llama 3 · ChromaDB · SHAP · SQLite history · 100% on-premise
                </p>
              </div>
            )}

            {result && (
              <div style={{ marginTop: 16 }}>
                {/* Restored from history badge */}
                {result.message_id && (
                  <div style={{
                    padding: "5px 12px", background: `${c.accent2}22`,
                    border: `1px solid ${c.accent2}44`, borderRadius: 6,
                    color: c.accent2, fontSize: 11, marginBottom: 10,
                    display: "inline-flex", alignItems: "center", gap: 6,
                  }}>
                    🗂️ Restored from history · Message ID: {result.message_id}
                  </div>
                )}

                {/* Result meta */}
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
                  <Pill color={c.accent} text={result.mode} />
                  <Pill color={c.accent2} text={`${result.references.length} sources`} />
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ color: c.muted, fontSize: 11 }}>Confidence:</span>
                    <div style={{ width: 130 }}>
                      <ConfidenceMeter value={result.confidence} />
                    </div>
                  </div>
                  <span style={{ color: c.muted, fontSize: 10 }}>
                    session: {result.session_id?.slice(0, 8)}
                  </span>
                </div>

                {/* Tabs */}
                <div style={{ display: "flex", gap: 2, marginBottom: 0, borderBottom: `1px solid ${c.border}` }}>
                  {tabs.map(t => (
                    <button key={t.id} onClick={() => setTab(t.id)} style={{
                      background: activeTab === t.id ? c.accent : "none",
                      color: activeTab === t.id ? "#000" : c.muted,
                      border: "none", borderRadius: "6px 6px 0 0",
                      padding: "8px 14px", fontSize: 11,
                      fontWeight: activeTab === t.id ? 700 : 400,
                      cursor: "pointer",
                    }}>{t.label}</button>
                  ))}
                </div>

                {/* Tab content */}
                <div style={{
                  background: c.surface, borderRadius: "0 8px 8px 8px",
                  padding: 18, border: `1px solid ${c.border}`, borderTop: "none",
                }}>
                  {activeTab === "answer" && <AnswerRenderer answer={result.answer} />}

                  {activeTab === "trace" && (
                    <div>
                      <p style={{ color: c.muted, fontSize: 11, marginBottom: 14, fontStyle: "italic" }}>
                        Full agent audit trail. Expand each step to see inputs, outputs, and model used.
                      </p>
                      {result.agent_trace.map((step, i) => (
                        <AgentStep key={i} step={step} index={i} />
                      ))}
                    </div>
                  )}

                  {activeTab === "shap" && (
                    <div>
                      <h4 style={{ color: c.accent, margin: "0 0 14px", fontSize: 13 }}>
                        📊 SHAP Feature Importance
                      </h4>
                      <ShapChart data={result.shap_analysis} />
                    </div>
                  )}

                  {activeTab === "refs" && (
                    <div>
                      <p style={{ color: c.muted, fontSize: 11, marginBottom: 12, fontStyle: "italic" }}>
                        Source chunks from ChromaDB ranked by cosine similarity.
                        Expand each to read the exact text, page number, and clause references.
                      </p>
                      {result.references.map((r, i) => (
                        <ReferenceCard key={i} refData={r} index={i} />
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
