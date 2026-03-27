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

async function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED PRIMITIVES — used by both Classic and Chat modes
// ═══════════════════════════════════════════════════════════════════════════════

function Pill({ color, text }) {
  return (
    <span style={{
      background: color + "22", color, border: `1px solid ${color}44`,
      borderRadius: 20, padding: "2px 10px", fontSize: 11, fontWeight: 700,
      letterSpacing: 0.5, textTransform: "uppercase", whiteSpace: "nowrap",
    }}>{text}</span>
  );
}

// TrustScoreBadge — replaces ConfidenceMeter entirely
function TrustScoreBadge({ trustData, compact = false }) {
  const score = trustData?.trust_score ?? 0;
  const level = trustData?.trust_level ?? "—";
  const color = trustData?.trust_color ?? c.muted;
  const pct   = Math.round(score * 100);

  if (compact) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{
          width: 80, height: 5, background: "#1e2d45", borderRadius: 4, overflow: "hidden",
        }}>
          <div style={{
            width: `${pct}%`, height: "100%", background: color, borderRadius: 4,
            transition: "width 0.8s ease",
          }} />
        </div>
        <span style={{
          color, fontWeight: 700, fontSize: 10,
          background: color + "22", border: `1px solid ${color}44`,
          borderRadius: 10, padding: "1px 7px",
        }}>{level} {pct}%</span>
      </div>
    );
  }

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "6px 12px", background: color + "18",
      border: `1px solid ${color}44`, borderRadius: 8,
    }}>
      <div style={{ width: 130, height: 7, background: "#1e2d45", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%", background: color, borderRadius: 4,
          transition: "width 0.8s ease", boxShadow: `0 0 10px ${color}88`,
        }} />
      </div>
      <div>
        <span style={{ color, fontWeight: 800, fontSize: 13 }}>{pct}%</span>
        <span style={{
          color, fontSize: 10, fontWeight: 700, letterSpacing: 1,
          background: color + "22", borderRadius: 10,
          padding: "1px 8px", marginLeft: 6,
        }}>{level}</span>
      </div>
    </div>
  );
}

// Keep ConfidenceMeter as alias so HistoryPanel doesn't break
function ConfidenceMeter({ value }) {
  const pct   = Math.round(value * 100);
  const color = pct >= 80 ? c.success : pct >= 50 ? c.warn : c.danger;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ width: 80, height: 5, background: "#1e2d45", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4 }} />
      </div>
      <span style={{ color, fontWeight: 700, fontSize: 10 }}>{pct}%</span>
    </div>
  );
}

function Skeleton({ lines = 4, compact = false }) {
  const pad = compact ? 14 : 20;
  return (
    <div style={{ padding: pad }}>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} style={{
          height: compact ? 11 : 14, background: c.border, borderRadius: 4,
          marginBottom: compact ? 9 : 12, width: i === lines - 1 ? "60%" : "100%",
          animation: "pulse 1.4s ease-in-out infinite",
          animationDelay: `${i * 0.1}s`,
        }} />
      ))}
      <style>{`@keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}`}</style>
    </div>
  );
}

function AgentStep({ step, index }) {
  const [open, setOpen] = useState(false);
  const cols = [c.accent2, c.accent, c.success, c.purple, c.pink];
  const col  = cols[index % cols.length];
  return (
    <div style={{ border: `1px solid ${col}33`, borderRadius: 8, overflow: "hidden", marginBottom: 8 }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", background: open ? `${col}18` : `${col}0a`,
        padding: "10px 14px", border: "none", cursor: "pointer",
        display: "flex", alignItems: "center", gap: 10,
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
          }}>{JSON.stringify(step.output, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

function ShapChart({ data }) {
  if (!data?.top_features?.length)
    return <p style={{ color: c.muted, fontSize: 12 }}>No SHAP data available.</p>;
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
              background: `linear-gradient(90deg,${c.accent2},${c.accent})`,
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
                  <span style={{ color: c.accent,  fontSize: 10 }}>SHAP: {ci.shap_importance.toFixed(3)}</span>
                </div>
              </div>
              {ci.top_contributing_words?.length > 0 && (
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 6 }}>
                  {ci.top_contributing_words.map((w, j) => (
                    <span key={j} style={{
                      background: w.shap > 0 ? `${c.success}22` : `${c.danger}22`,
                      color:      w.shap > 0 ? c.success : c.danger,
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

// ── Faithfulness Panel ────────────────────────────────────────────────────────
function FaithfulnessPanel({ data }) {
  if (!data?.sentences?.length) {
    return (
      <div style={{ color: c.muted, fontSize: 12, padding: 8 }}>
        No faithfulness data available. Run a query to see sentence-level grounding.
      </div>
    );
  }

  const labelColor = { supported: c.success, partial: c.warn, unsupported: c.danger };
  const labelIcon  = { supported: "✅", partial: "⚠️", unsupported: "❌" };

  const n   = data.total_sentences;
  const sup = data.supported_count;
  const par = data.partial_count;
  const uns = data.unsupported_count;

  return (
    <div>
      {/* Summary bar */}
      <div style={{
        display: "flex", gap: 6, marginBottom: 14,
        padding: "8px 12px", background: c.surface2,
        border: `1px solid ${c.border}`, borderRadius: 8,
        flexWrap: "wrap", alignItems: "center",
      }}>
        <span style={{ color: c.muted, fontSize: 11, marginRight: 4 }}>
          {n} sentences analysed:
        </span>
        {[["supported", sup], ["partial", par], ["unsupported", uns]].map(([lbl, cnt]) => (
          <span key={lbl} style={{
            color: labelColor[lbl], fontSize: 11, fontWeight: 700,
            background: labelColor[lbl] + "22", borderRadius: 10,
            padding: "2px 10px", border: `1px solid ${labelColor[lbl]}44`,
          }}>
            {labelIcon[lbl]} {cnt} {lbl}
          </span>
        ))}
        <span style={{ color: c.muted, fontSize: 10, marginLeft: "auto" }}>
          Grounding: <strong style={{ color: data.grounding_ratio >= 0.7 ? c.success : c.warn }}>
            {Math.round(data.grounding_ratio * 100)}%
          </strong>
        </span>
      </div>

      {/* Thresholds legend */}
      <div style={{ display: "flex", gap: 14, marginBottom: 12, fontSize: 10, color: c.muted }}>
        <span>✅ supported: sim ≥ {data.high_threshold}</span>
        <span>⚠️ partial: sim ≥ {data.low_threshold}</span>
        <span>❌ unsupported: sim &lt; {data.low_threshold}</span>
        <span style={{ marginLeft: "auto", fontStyle: "italic" }}>method: {data.method}</span>
      </div>

      {/* Unsupported warning banner */}
      {uns > 0 && (
        <div style={{
          padding: "8px 12px", background: `${c.danger}18`,
          border: `1px solid ${c.danger}44`, borderRadius: 6,
          color: c.danger, fontSize: 11, marginBottom: 12,
        }}>
          ⚠️ {uns} sentence{uns > 1 ? "s" : ""} could not be grounded in the source documents —
          verify these claims independently before citing.
        </div>
      )}

      {/* Sentence list */}
      {data.sentences.map((s, i) => {
        const col = labelColor[s.label];
        return (
          <div key={i} style={{
            padding: "10px 12px", marginBottom: 8,
            background: col + "0d", border: `1px solid ${col}33`,
            borderLeft: `3px solid ${col}`, borderRadius: "0 6px 6px 0",
          }}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
              <span style={{ fontSize: 13, flexShrink: 0, marginTop: 1 }}>{labelIcon[s.label]}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ color: c.text, fontSize: 12, lineHeight: 1.7, margin: "0 0 6px" }}>
                  {s.sentence}
                </p>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                  <span style={{
                    color: col, fontSize: 10, fontWeight: 700,
                    background: col + "22", borderRadius: 10, padding: "1px 8px",
                  }}>
                    {s.label} · sim {s.best_similarity.toFixed(3)}
                  </span>
                  {s.label !== "unsupported" && (
                    <span style={{ color: c.muted, fontSize: 10 }}>
                      best match: <span style={{ color: c.text }}>{s.best_source}</span>
                      {s.best_page > 0 && <span style={{ color: c.purple }}> · p.{s.best_page}</span>}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── LIME Panel ────────────────────────────────────────────────────────────────
function LIMEPanel({ data }) {
  const [showPerturbations, setShowPerturbations] = useState(false);

  if (!data?.word_influence?.length) {
    return (
      <div style={{ color: c.muted, fontSize: 12, padding: 8 }}>
        No LIME data available.
      </div>
    );
  }

  const stability = data.stability_score ?? 1;
  const stabColor = stability >= 0.8 ? c.success : stability >= 0.6 ? c.warn : c.danger;
  const stabLabel = stability >= 0.8 ? "STABLE" : stability >= 0.6 ? "MODERATE" : "FRAGILE";
  const maxInf    = Math.max(...data.word_influence.map(w => w.influence), 0.001);

  return (
    <div>
      {/* Stability summary */}
      <div style={{
        display: "flex", gap: 14, marginBottom: 16,
        padding: "10px 14px", background: stabColor + "18",
        border: `1px solid ${stabColor}44`, borderRadius: 8, flexWrap: "wrap",
      }}>
        <div>
          <p style={{ color: stabColor, fontWeight: 800, fontSize: 16, margin: "0 0 2px" }}>
            {Math.round(stability * 100)}%
            <span style={{
              fontSize: 10, letterSpacing: 1, marginLeft: 8,
              background: stabColor + "22", borderRadius: 10, padding: "2px 8px",
            }}>{stabLabel}</span>
          </p>
          <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>
            Retrieval stability across {data.n_perturbations} query perturbations
          </p>
        </div>
        <div style={{ marginLeft: "auto", textAlign: "right" }}>
          {data.key_terms?.length > 0 && (
            <p style={{ color: c.text, fontSize: 11, margin: "0 0 2px" }}>
              🔑 Key terms: {data.key_terms.map(t => (
                <span key={t} style={{
                  background: c.accent + "22", color: c.accent,
                  borderRadius: 4, padding: "1px 7px", marginLeft: 4, fontSize: 10,
                }}>{t}</span>
              ))}
            </p>
          )}
          {data.stable_terms?.length > 0 && (
            <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>
              stable: {data.stable_terms.join(", ")}
            </p>
          )}
        </div>
      </div>

      {stability < 0.6 && (
        <div style={{
          padding: "8px 12px", background: `${c.danger}18`,
          border: `1px solid ${c.danger}44`, borderRadius: 6,
          color: c.danger, fontSize: 11, marginBottom: 12,
        }}>
          ⚠️ Fragile retrieval — results depend heavily on exact query wording.
          Try rephrasing to verify the answer holds.
        </div>
      )}

      {/* Word influence bars */}
      <h5 style={{ color: c.text, fontSize: 12, margin: "0 0 10px" }}>
        Word Influence on Retrieval
      </h5>
      <p style={{ color: c.muted, fontSize: 11, marginBottom: 12, fontStyle: "italic" }}>
        How much does removing each word change what gets retrieved?
        High influence = that word is critical to the result.
      </p>
      {data.word_influence.map((w, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7 }}>
          <span style={{ color: c.text, fontSize: 12, width: 130, textAlign: "right", flexShrink: 0 }}>
            {w.word}
          </span>
          <div style={{ flex: 1, height: 18, background: "#1e2d45", borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              width: `${(w.influence / maxInf) * 100}%`, height: "100%",
              background: w.influence > 0.3
                ? `linear-gradient(90deg,${c.danger},${c.warn})`
                : `linear-gradient(90deg,${c.accent2},${c.success})`,
              borderRadius: 3, minWidth: w.influence > 0 ? 2 : 0,
              transition: "width 0.5s ease",
            }} />
          </div>
          <span style={{ color: c.muted, fontSize: 10, width: 44, textAlign: "right" }}>
            {w.influence.toFixed(3)}
          </span>
          <span style={{ color: c.muted, fontSize: 9, width: 40 }}>
            ({w.tests}×)
          </span>
        </div>
      ))}

      {/* Perturbation log toggle */}
      <button onClick={() => setShowPerturbations(!showPerturbations)} style={{
        background: "none", border: `1px solid ${c.border}`, color: c.muted,
        borderRadius: 5, padding: "4px 12px", cursor: "pointer", fontSize: 10, marginTop: 12,
      }}>
        {showPerturbations ? "▲ Hide" : "▼ Show"} perturbation log ({data.perturbations?.length ?? 0} variants)
      </button>

      {showPerturbations && data.perturbations?.length > 0 && (
        <div style={{ marginTop: 10, maxHeight: 280, overflowY: "auto" }}>
          {data.perturbations.map((p, i) => (
            <div key={i} style={{
              padding: "7px 10px", background: c.surface2, borderRadius: 5,
              marginBottom: 5, border: `1px solid ${c.border}`,
              display: "flex", alignItems: "center", gap: 10,
            }}>
              <span style={{ color: c.muted, fontSize: 9, flexShrink: 0 }}>#{i + 1}</span>
              <span style={{ color: c.text, fontSize: 10, flex: 1 }}>"{p.perturbed_query}"</span>
              <span style={{ color: c.muted, fontSize: 9 }}>
                removed: <span style={{ color: c.accent }}>{p.masked_words?.join(", ")}</span>
              </span>
              <span style={{
                fontSize: 10, fontWeight: 700,
                color: p.retrieval_changed ? c.danger : c.success,
              }}>
                {Math.round(p.jaccard_similarity * 100)}% match
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Trust Score Panel (full breakdown) ────────────────────────────────────────
function TrustPanel({ data }) {
  if (!data?.components?.length) {
    return <div style={{ color: c.muted, fontSize: 12 }}>No trust data available.</div>;
  }

  const color = data.trust_color ?? c.muted;

  return (
    <div>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", gap: 14, marginBottom: 16,
        padding: "12px 16px", background: color + "18",
        border: `1px solid ${color}44`, borderRadius: 10,
      }}>
        <div>
          <p style={{ color, fontWeight: 800, fontSize: 28, margin: "0 0 2px" }}>
            {Math.round((data.trust_score ?? 0) * 100)}%
          </p>
          <p style={{ color, fontSize: 11, fontWeight: 700, letterSpacing: 1, margin: 0 }}>
            {data.trust_level} TRUST
          </p>
        </div>
        <div style={{ flex: 1, borderLeft: `1px solid ${color}33`, paddingLeft: 14 }}>
          <p style={{ color: c.text, fontSize: 12, margin: "0 0 4px" }}>{data.trust_message}</p>
          {data.warnings?.length > 0 && data.warnings.map((w, i) => (
            <p key={i} style={{ color: c.warn, fontSize: 11, margin: "2px 0" }}>⚠️ {w}</p>
          ))}
        </div>
      </div>

      {/* Component breakdown */}
      <h5 style={{ color: c.text, fontSize: 12, margin: "0 0 10px" }}>Score Components</h5>
      {data.components.map((comp, i) => {
        const contribPct = Math.round(comp.contribution * 100);
        const rawPct     = Math.round(comp.raw_value * 100);
        const srcColor   = comp.source === "SHAP" ? c.accent
          : comp.source === "LIME" ? c.purple : c.accent2;
        return (
          <div key={i} style={{
            padding: "10px 12px", marginBottom: 8,
            background: c.surface2, border: `1px solid ${c.border}`, borderRadius: 7,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <span style={{
                fontSize: 9, fontWeight: 700, letterSpacing: 1,
                background: srcColor + "22", color: srcColor,
                borderRadius: 8, padding: "1px 7px", flexShrink: 0,
              }}>{comp.source}</span>
              <span style={{ color: c.text, fontSize: 12, fontWeight: 600, flex: 1 }}>
                {comp.name}
              </span>
              <span style={{ color: c.muted, fontSize: 10 }}>
                {rawPct}% × {Math.round(comp.weight * 100)}% weight
              </span>
              <span style={{ color, fontWeight: 700, fontSize: 12, minWidth: 34, textAlign: "right" }}>
                +{contribPct}%
              </span>
            </div>
            {/* Contribution bar */}
            <div style={{ height: 5, background: "#1e2d45", borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
              <div style={{
                width: `${rawPct}%`, height: "100%", background: srcColor,
                borderRadius: 3, transition: "width 0.6s ease",
              }} />
            </div>
            <p style={{ color: c.muted, fontSize: 10, margin: 0, lineHeight: 1.5 }}>
              {comp.description}
            </p>
          </div>
        );
      })}

      <p style={{ color: c.muted, fontSize: 9, marginTop: 8, fontStyle: "italic" }}>
        Weights: Grounding 30% · Similarity 25% · Coverage 15% · SHAP Alignment 15% · LIME Stability 15%
      </p>
    </div>
  );
}

function ReferenceCard({ refData, index }) {
  const [open, setOpen] = useState(false);
  const relColor = refData.relevance_pct >= 80 ? c.success
    : refData.relevance_pct >= 60 ? c.warn : c.muted;
  return (
    <div style={{ border: `1px solid ${c.border}`, borderRadius: 8, marginBottom: 8, overflow: "hidden" }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", padding: "10px 14px", background: c.surface2,
        border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 10,
      }}>
        <span style={{ color: c.accent, fontWeight: 800, fontSize: 14, flexShrink: 0 }}>[{index + 1}]</span>
        <span style={{ color: c.text, fontSize: 12, flex: 1, textAlign: "left",
          overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" }}>
          {refData.filename}
        </span>
        <span style={{ color: c.muted, fontSize: 11 }}>chunk {refData.chunk_index}/{refData.total_chunks}</span>
        {refData.page_number > 0 && (
          <span style={{ color: c.purple, fontSize: 10 }}>p.{refData.page_number}</span>
        )}
        <span style={{ color: relColor, fontWeight: 700, fontSize: 12, minWidth: 40 }}>
          {refData.relevance_pct}%
        </span>
        <span style={{ color: c.muted, fontSize: 14 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{ padding: 14, background: c.bg }}>
          <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
            <Pill color={relColor}  text={`${refData.relevance_pct}% relevant`} />
            <Pill color={c.muted}   text={`chunk ${refData.chunk_index} of ${refData.total_chunks}`} />
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
                <span style={{ color: c.purple }}>📄 <strong>Page:</strong> {refData.page_number}</span>
              )}
              {refData.clause_refs && (
                <span style={{ color: c.pink }}>📌 <strong>Clauses:</strong> {refData.clause_refs}</span>
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

function AnswerRenderer({ answer }) {
  return (
    <div style={{ color: c.text, fontSize: 13, lineHeight: 1.85,
      fontFamily: "'IBM Plex Serif',Georgia,serif" }}>
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
            const relMatch   = rest.match(/Relevance:\s([\d.]+)%/);
            const pageMatch  = rest.match(/📄\s([^|]+)/);
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
                {pageMatch  && <span style={{ color: c.purple, fontSize: 10 }}>📄 {pageMatch[1].trim()}</span>}
                {clauseMatch && <span style={{ color: c.pink,   fontSize: 10 }}>📌 {clauseMatch[1].trim()}</span>}
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
                j % 2 === 1 ? <strong key={j} style={{ color: c.accent2 }}>{part}</strong> : part
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

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED SIDEBAR COMPONENTS — DocumentManager + HistoryPanel
// ═══════════════════════════════════════════════════════════════════════════════

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

  useEffect(() => {
    if (!polling) return;
    const iv = setInterval(async () => {
      try {
        const r = await fetchWithTimeout(`${API}/documents/${polling}/status`, {}, 5000);
        const { status } = await r.json();
        if (status === "ready") {
          clearInterval(iv); setPolling(null);
          setUploadMsg({ ok: true, text: "Document indexed successfully ✓" });
          loadDocs();
        } else if (status?.startsWith("error:")) {
          clearInterval(iv); setPolling(null);
          setUploadMsg({ ok: false, text: `Indexing failed: ${status.slice(6)}` });
        }
      } catch { clearInterval(iv); setPolling(null); }
    }, 2000);
    return () => clearInterval(iv);
  }, [polling, loadDocs]);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setUploadMsg({ ok: false, text: `File too large (max ${MAX_FILE_MB}MB)` }); return;
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
    } catch {}
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
          <p style={{ color: c.muted, fontSize: 10, margin: 0 }}>PDF · DOCX · TXT · MD · max {MAX_FILE_MB}MB</p>
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
      ) : docs.map(doc => (
        <div key={doc.doc_id} style={{
          display: "flex", alignItems: "center", gap: 8, padding: "8px 10px",
          background: c.surface2, borderRadius: 6, marginBottom: 5,
          border: `1px solid ${doc.index_status === "indexing" ? c.warn : c.border}`,
        }}>
          <span style={{ fontSize: 15 }}>📄</span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <p style={{ color: c.text, fontSize: 11, margin: 0, fontWeight: 600,
              whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {doc.filename}
            </p>
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
      ))}
    </div>
  );
}

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
      const r = await fetchWithTimeout(`${API}/history/search?q=${encodeURIComponent(val)}`, {}, 5000);
      if (r.ok) setMsgs(await r.json());
    } catch {}
  };

  const handleDeleteSession = async (session_id, e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this session from history?")) return;
    try {
      await fetchWithTimeout(`${API}/history/sessions/${session_id}`, { method: "DELETE" }, 5000);
      loadHistory(); loadStats();
      if (selected?.session_id === session_id) setSelected(null);
    } catch {}
  };

  const confColor = (v) => v >= 0.8 ? c.success : v >= 0.5 ? c.warn : c.danger;

  return (
    <div style={{ marginTop: 16 }}>
      <button onClick={() => setOpen(!open)} style={{
        width: "100%", background: open ? `${c.accent2}22` : "none",
        border: `1px solid ${c.border}`, color: c.text,
        borderRadius: 6, padding: "8px 12px", cursor: "pointer",
        display: "flex", alignItems: "center", gap: 8, fontSize: 12,
      }}>
        <span>🗂️</span>
        <span style={{ flex: 1, textAlign: "left", fontWeight: 600 }}>Query History</span>
        {stats && <span style={{ color: c.muted, fontSize: 10 }}>{stats.total_messages ?? 0} saved</span>}
        <span style={{ color: c.muted }}>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div style={{ marginTop: 8 }}>
          {stats && stats.total_messages > 0 && (
            <div style={{
              display: "flex", gap: 12, padding: "6px 10px", background: c.surface2,
              borderRadius: 6, marginBottom: 8, flexWrap: "wrap",
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
                  </strong> avg
                </span>
              )}
            </div>
          )}

          <input value={search} onChange={e => handleSearch(e.target.value)}
            placeholder="Search past queries…"
            style={{
              width: "100%", background: c.bg, border: `1px solid ${c.border}`,
              borderRadius: 6, padding: "6px 10px", color: c.text, fontSize: 11,
              marginBottom: 8, boxSizing: "border-box", outline: "none",
            }} />

          <div style={{ maxHeight: 380, overflowY: "auto" }}>
            {msgs.length === 0 ? (
              <p style={{ color: c.muted, fontSize: 11, textAlign: "center", padding: "12px 0" }}>
                {search ? "No results found" : "No history yet"}
              </p>
            ) : msgs.map(msg => (
              <div key={msg.id}
                onClick={() => setSelected(selected?.id === msg.id ? null : msg)}
                style={{
                  padding: "8px 10px",
                  background: selected?.id === msg.id ? `${c.accent2}22` : c.surface2,
                  borderRadius: 6, marginBottom: 5, cursor: "pointer",
                  border: `1px solid ${selected?.id === msg.id ? c.accent2 : c.border}`,
                }}>
                <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ color: c.text, fontSize: 11, margin: 0, fontWeight: 600,
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {msg.query}
                    </p>
                    <div style={{ display: "flex", gap: 8, marginTop: 3, flexWrap: "wrap" }}>
                      <span style={{ color: c.muted, fontSize: 9 }}>
                        {new Date(msg.timestamp).toLocaleString()}
                      </span>
                      <span style={{ color: confColor(msg.confidence), fontSize: 9, fontWeight: 700 }}>
                        {Math.round(msg.confidence * 100)}% conf
                      </span>
                      <span style={{ color: c.muted, fontSize: 9 }}>
                        {msg.references?.length ?? 0} src
                      </span>
                    </div>
                  </div>
                  <button onClick={e => handleDeleteSession(msg.session_id, e)} style={{
                    background: "none", border: "none", color: c.muted,
                    cursor: "pointer", fontSize: 11, padding: "0 2px", flexShrink: 0,
                  }} title="Delete session">✕</button>
                </div>

                {selected?.id === msg.id && (
                  <div style={{ marginTop: 10, borderTop: `1px solid ${c.border}`, paddingTop: 10 }}>
                    <p style={{ color: c.muted, fontSize: 10, marginBottom: 4, fontWeight: 700 }}>
                      ANSWER PREVIEW:
                    </p>
                    <p style={{ color: c.text, fontSize: 11, lineHeight: 1.6,
                      maxHeight: 160, overflowY: "auto", margin: "0 0 8px",
                      whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                      {msg.answer?.slice(0, 600)}{msg.answer?.length > 600 ? "…" : ""}
                    </p>
                    {msg.references?.length > 0 && (
                      <div style={{ marginBottom: 8 }}>
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
                    <button onClick={e => { e.stopPropagation(); onRestore(msg); }} style={{
                      background: `${c.accent}22`, border: `1px solid ${c.accent}44`,
                      color: c.accent, borderRadius: 4, padding: "4px 10px",
                      cursor: "pointer", fontSize: 10, fontWeight: 600,
                    }}>↩ Restore to view</button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLASSIC MODE — single query, 4-tab result panel
// ═══════════════════════════════════════════════════════════════════════════════

function ClassicMode({ onRestore }) {
  const [query, setQuery]     = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);
  const [activeTab, setTab]   = useState("answer");
  const [topK, setTopK]       = useState(5);
  const [mode, setMode]       = useState("full");
  const textRef = useRef();

  // Expose restore handler to parent so HistoryPanel can restore into this mode
  useEffect(() => {
    if (onRestore?.pending) {
      setResult(onRestore.pending);
      setTab("answer");
      onRestore.clear();
    }
  }, [onRestore]);

  const handleQuery = async () => {
    const q = query.trim();
    if (!q || loading) return;
    setLoading(true); setError(null); setResult(null); setTab("answer");
    try {
      const r = await fetchWithTimeout(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: topK, mode }),
      }, 120000);
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || "Query failed"); }
      setResult(await r.json());
    } catch (e) {
      setError(e.name === "AbortError"
        ? "Request timed out. Check Ollama is running." : e.message);
    }
    setLoading(false);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleQuery();
  };

  const tabs = [
    { id: "answer",       label: "📋 Answer" },
    { id: "trace",        label: `🤖 Agents (${result?.agent_trace?.length ?? 0})` },
    { id: "faithfulness", label: "🔬 Faithfulness" },
    { id: "shap",         label: "📊 SHAP" },
    { id: "lime",         label: "🧪 LIME" },
    { id: "trust",        label: "🛡 Trust" },
    { id: "refs",         label: `📎 Refs (${result?.references?.length ?? 0})` },
  ];

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Query bar */}
      <div style={{ padding: "14px 20px", borderBottom: `1px solid ${c.border}`, background: c.surface }}>
        <textarea ref={textRef} value={query}
          onChange={e => setQuery(e.target.value)} onKeyDown={handleKey}
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
            {[3, 5, 8, 10].map(n => <option key={n} value={n}>Top {n} chunks</option>)}
          </select>
          <span style={{ color: c.muted, fontSize: 10, flex: 1 }}>Ctrl+Enter to submit</span>
          <button onClick={handleQuery} disabled={loading || !query.trim()} style={{
            background: loading ? c.muted : c.accent,
            color: "#000", border: "none", borderRadius: 6, padding: "8px 22px",
            fontSize: 13, fontWeight: 800,
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
          <div style={{ margin: "16px 0", padding: 14, background: `${c.danger}22`,
            border: `1px solid ${c.danger}44`, borderRadius: 8, color: c.danger, fontSize: 12 }}>
            ⚠️ {error}
          </div>
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
            <p style={{ fontSize: 14, marginBottom: 6 }}>Upload regulatory documents, then ask questions.</p>
            <p style={{ fontSize: 11 }}>Llama 3 · ChromaDB · SHAP · SQLite history · 100% on-premise</p>
          </div>
        )}
        {result && (
          <div style={{ marginTop: 16 }}>
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
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
              <Pill color={c.accent}  text={result.mode} />
              <Pill color={c.accent2} text={`${result.references.length} sources`} />
              <TrustScoreBadge trustData={result.trust_data} />
              <span style={{ color: c.muted, fontSize: 10 }}>session: {result.session_id?.slice(0, 8)}</span>
            </div>
            {/* Tabs */}
            <div style={{ display: "flex", gap: 2, borderBottom: `1px solid ${c.border}` }}>
              {tabs.map(t => (
                <button key={t.id} onClick={() => setTab(t.id)} style={{
                  background: activeTab === t.id ? c.accent : "none",
                  color: activeTab === t.id ? "#000" : c.muted,
                  border: "none", borderRadius: "6px 6px 0 0",
                  padding: "8px 14px", fontSize: 11,
                  fontWeight: activeTab === t.id ? 700 : 400, cursor: "pointer",
                }}>{t.label}</button>
              ))}
            </div>
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
                  {result.agent_trace.map((step, i) => <AgentStep key={i} step={step} index={i} />)}
                </div>
              )}
              {activeTab === "faithfulness" && (
                <div>
                  <h4 style={{ color: c.accent2, margin: "0 0 14px", fontSize: 13 }}>
                    🔬 RAG Faithfulness — Sentence Grounding
                  </h4>
                  <FaithfulnessPanel data={result.faithfulness_data} />
                </div>
              )}
              {activeTab === "shap" && (
                <div>
                  <h4 style={{ color: c.accent, margin: "0 0 14px", fontSize: 13 }}>📊 SHAP Feature Importance</h4>
                  <ShapChart data={result.shap_analysis} />
                </div>
              )}
              {activeTab === "lime" && (
                <div>
                  <h4 style={{ color: c.purple, margin: "0 0 14px", fontSize: 13 }}>
                    🧪 LIME — Retrieval Stability
                  </h4>
                  <LIMEPanel data={result.lime_data} />
                </div>
              )}
              {activeTab === "trust" && (
                <div>
                  <h4 style={{ color: result.trust_data?.trust_color ?? c.accent, margin: "0 0 14px", fontSize: 13 }}>
                    🛡 Trust Score Breakdown
                  </h4>
                  <TrustPanel data={result.trust_data} />
                </div>
              )}
              {activeTab === "refs" && (
                <div>
                  <p style={{ color: c.muted, fontSize: 11, marginBottom: 12, fontStyle: "italic" }}>
                    Source chunks from ChromaDB ranked by cosine similarity.
                  </p>
                  {result.references.map((r, i) => <ReferenceCard key={i} refData={r} index={i} />)}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAT MODE — conversation thread above pinned query bar
// ═══════════════════════════════════════════════════════════════════════════════

function ChatMessageBubble({ msg }) {
  const [detailTab, setDetailTab] = useState(null);
  const toggle = (tab) => setDetailTab(t => t === tab ? null : tab);
  const confColor = msg.confidence >= 0.8 ? c.success
    : msg.confidence >= 0.5 ? c.warn : c.danger;

  return (
    <div style={{ marginBottom: 24 }}>
      {/* User */}
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
        <div style={{
          maxWidth: "78%", padding: "10px 14px",
          background: `${c.accent2}22`, border: `1px solid ${c.accent2}44`,
          borderRadius: "12px 12px 3px 12px", fontSize: 13, color: c.text, lineHeight: 1.5,
        }}>
          {msg.query}
          <div style={{ fontSize: 9, color: c.muted, marginTop: 4, textAlign: "right" }}>
            Turn {msg.turn_number} · {msg.mode}
          </div>
        </div>
      </div>

      {/* Assistant */}
      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
        <div style={{
          width: 28, height: 28, borderRadius: "50%", background: c.accent,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 14, flexShrink: 0, marginTop: 2,
        }}>⚖️</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            background: c.surface, border: `1px solid ${c.border}`,
            borderRadius: "3px 12px 12px 12px", padding: "14px 16px",
          }}>
            {msg.loading ? <Skeleton lines={3} compact /> :
             msg.error   ? <p style={{ color: c.danger, fontSize: 12, margin: 0 }}>⚠️ {msg.error}</p> :
             <AnswerRenderer answer={msg.answer} />}
          </div>

          {!msg.loading && !msg.error && (
            <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              <TrustScoreBadge trustData={msg.trust_data} compact />
              <span style={{ color: c.muted, fontSize: 10 }}>{msg.references?.length ?? 0} sources</span>
              {[
                { id: "trace",        label: `🤖 Agents (${msg.agent_trace?.length ?? 0})` },
                { id: "faithfulness", label: "🔬 Faithful" },
                { id: "shap",         label: "📊 SHAP" },
                { id: "lime",         label: "🧪 LIME" },
                { id: "trust",        label: "🛡 Trust" },
                { id: "refs",         label: `📎 Refs (${msg.references?.length ?? 0})` },
              ].map(btn => (
                <button key={btn.id} onClick={() => toggle(btn.id)} style={{
                  background: detailTab === btn.id ? `${c.accent}22` : "none",
                  border: `1px solid ${detailTab === btn.id ? c.accent : c.border}`,
                  color: detailTab === btn.id ? c.accent : c.muted,
                  borderRadius: 5, padding: "3px 10px", fontSize: 10,
                  cursor: "pointer", transition: "all 0.15s",
                }}>{btn.label}</button>
              ))}
              {msg.message_id && (
                <span style={{ color: c.muted, fontSize: 9, marginLeft: "auto" }}>id:{msg.message_id}</span>
              )}
            </div>
          )}

          {detailTab && !msg.loading && !msg.error && (
            <div style={{ marginTop: 8, background: c.surface, border: `1px solid ${c.border}`, borderRadius: 8, padding: 14 }}>
              {detailTab === "trace" && (
                <div>
                  <p style={{ color: c.muted, fontSize: 11, marginBottom: 10, fontStyle: "italic" }}>
                    Full agent audit trail for this message.
                  </p>
                  {msg.agent_trace?.map((step, i) => <AgentStep key={i} step={step} index={i} />)}
                </div>
              )}
              {detailTab === "faithfulness" && (
                <div>
                  <h4 style={{ color: c.accent2, margin: "0 0 12px", fontSize: 13 }}>🔬 Sentence Grounding</h4>
                  <FaithfulnessPanel data={msg.faithfulness_data} />
                </div>
              )}
              {detailTab === "shap" && <ShapChart data={msg.shap_analysis} />}
              {detailTab === "lime" && (
                <div>
                  <h4 style={{ color: c.purple, margin: "0 0 12px", fontSize: 13 }}>🧪 Retrieval Stability</h4>
                  <LIMEPanel data={msg.lime_data} />
                </div>
              )}
              {detailTab === "trust" && (
                <div>
                  <h4 style={{ color: msg.trust_data?.trust_color ?? c.accent, margin: "0 0 12px", fontSize: 13 }}>
                    🛡 Trust Score
                  </h4>
                  <TrustPanel data={msg.trust_data} />
                </div>
              )}
              {detailTab === "refs" && (
                <div>
                  <p style={{ color: c.muted, fontSize: 11, marginBottom: 10, fontStyle: "italic" }}>
                    Source chunks ranked by cosine similarity.
                  </p>
                  {msg.references?.map((r, i) => <ReferenceCard key={i} refData={r} index={i} />)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ChatMode() {
  const [messages, setMessages]   = useState([]);
  const [query, setQuery]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [topK, setTopK]           = useState(5);
  const [mode, setMode]           = useState("full");
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const threadRef = useRef();

  useEffect(() => {
    if (threadRef.current)
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
  }, [messages]);

  const handleNewChat = () => {
    if (messages.length > 0 &&
      !window.confirm("Start a new chat? This conversation stays in history.")) return;
    setMessages([]);
    setQuery("");
    setSessionId(crypto.randomUUID());
  };

  const handleQuery = async () => {
    const q = query.trim();
    if (!q || loading) return;
    const placeholderId = Date.now();
    setMessages(prev => [...prev, {
      id: placeholderId, query: q, loading: true,
      answer: "", confidence: 0, mode, references: [],
      agent_trace: [], shap_analysis: {},
      turn_number: prev.filter(m => !m.loading).length + 1,
    }]);
    setQuery("");
    setLoading(true);
    try {
      const r = await fetchWithTimeout(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, session_id: sessionId, top_k: topK, mode }),
      }, 120000);
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || "Query failed"); }
      const data = await r.json();
      setMessages(prev => prev.map(m =>
        m.id === placeholderId ? { ...data, id: placeholderId, loading: false, error: null } : m
      ));
    } catch (e) {
      const errMsg = e.name === "AbortError"
        ? "Request timed out. Check Ollama is running." : e.message;
      setMessages(prev => prev.map(m =>
        m.id === placeholderId ? { ...m, loading: false, error: errMsg } : m
      ));
    }
    setLoading(false);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleQuery();
  };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Sub-header for chat controls */}
      <div style={{
        padding: "8px 20px", borderBottom: `1px solid ${c.border}`,
        background: c.surface, display: "flex", alignItems: "center", gap: 10,
      }}>
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "3px 12px", background: `${c.accent2}18`,
          border: `1px solid ${c.accent2}33`, borderRadius: 20,
        }}>
          <span style={{ color: c.accent2, fontSize: 9 }}>SESSION</span>
          <span style={{ color: c.text, fontSize: 9 }}>{sessionId.slice(0, 8)}</span>
          <span style={{ color: c.accent2, fontSize: 9 }}>
            · {messages.filter(m => !m.loading).length} turns
          </span>
        </div>
        {messages.length > 0 && (
          <span style={{
            color: c.success, fontSize: 9, background: `${c.success}18`,
            border: `1px solid ${c.success}33`, borderRadius: 10, padding: "2px 8px",
          }}>
            🧠 Memory active
          </span>
        )}
        <div style={{ flex: 1 }} />
        <button onClick={handleNewChat} style={{
          background: `${c.accent}22`, border: `1px solid ${c.accent}44`,
          color: c.accent, borderRadius: 5, padding: "4px 12px",
          cursor: "pointer", fontSize: 10, fontWeight: 700,
        }}>+ New Chat</button>
      </div>

      {/* Thread */}
      <div ref={threadRef} style={{ flex: 1, overflowY: "auto", padding: "20px 24px" }}>
        {messages.length === 0 ? (
          <div style={{ textAlign: "center", padding: "60px 20px", color: c.muted }}>
            <div style={{ fontSize: 48, marginBottom: 14 }}>💬</div>
            <p style={{ fontSize: 14, marginBottom: 8, color: c.text }}>
              Start a regulatory conversation
            </p>
            <p style={{ fontSize: 11, marginBottom: 4 }}>
              The AI remembers context — ask follow-up questions naturally.
            </p>
            <p style={{ fontSize: 10 }}>
              e.g. "What are Basel III requirements?" then "How does that compare to Basel II?"
            </p>
          </div>
        ) : messages.map(msg => (
          <ChatMessageBubble key={msg.id} msg={msg} />
        ))}
      </div>

      {/* Pinned input */}
      <div style={{ borderTop: `1px solid ${c.border}`, background: c.surface, padding: "12px 20px" }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "center" }}>
          <select value={mode} onChange={e => setMode(e.target.value)} style={{
            background: c.surface2, border: `1px solid ${c.border}`, color: c.text,
            borderRadius: 5, padding: "4px 8px", fontSize: 10,
          }}>
            <option value="full">Full Analysis</option>
            <option value="quick">Quick Answer</option>
            <option value="compare">Compare</option>
          </select>
          <select value={topK} onChange={e => setTopK(+e.target.value)} style={{
            background: c.surface2, border: `1px solid ${c.border}`, color: c.text,
            borderRadius: 5, padding: "4px 8px", fontSize: 10,
          }}>
            {[3, 5, 8, 10].map(n => <option key={n} value={n}>Top {n} chunks</option>)}
          </select>
          <span style={{ color: c.muted, fontSize: 9, flex: 1, textAlign: "right" }}>Ctrl+Enter</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <textarea value={query} onChange={e => setQuery(e.target.value)} onKeyDown={handleKey}
            placeholder={messages.length === 0
              ? `Ask a regulatory question…`
              : `Ask a follow-up… e.g. "What about liquidity requirements?"`}
            rows={2}
            style={{
              flex: 1, background: c.bg, border: `1px solid ${c.border}`,
              borderRadius: 8, padding: "10px 12px", color: c.text,
              fontSize: 13, resize: "none", outline: "none", fontFamily: "inherit", lineHeight: 1.5,
            }}
            onFocus={e => (e.target.style.borderColor = c.accent)}
            onBlur={e => (e.target.style.borderColor = c.border)}
          />
          <button onClick={handleQuery} disabled={loading || !query.trim()} style={{
            background: loading ? c.muted : c.accent,
            color: "#000", border: "none", borderRadius: 8,
            padding: "0 20px", fontSize: 13, fontWeight: 800,
            cursor: (loading || !query.trim()) ? "not-allowed" : "pointer",
            boxShadow: loading ? "none" : `0 0 14px ${c.accent}55`,
            transition: "all 0.2s", minWidth: 80,
          }}>
            {loading ? "⏳" : "Send ▶"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROOT APP — header with mode switcher, shared sidebar
// ═══════════════════════════════════════════════════════════════════════════════

export default function App() {
  const [appMode, setAppMode] = useState("classic");  // "classic" | "chat"
  const [health, setHealth]   = useState(null);

  // Classic mode restore bridge
  const classicRestoreRef = useRef({ pending: null, clear: () => {} });
  const [restoreTick, setRestoreTick] = useState(0);

  const pollHealth = useCallback(async () => {
    try {
      const r = await fetchWithTimeout(`${API}/health`, {}, 5000);
      if (r.ok) setHealth(await r.json());
      else setHealth({ api: "error" });
    } catch { setHealth({ api: "unreachable" }); }
  }, []);

  useEffect(() => {
    pollHealth();
    const iv = setInterval(pollHealth, 30000);
    return () => clearInterval(iv);
  }, [pollHealth]);

  const handleRestore = (msg) => {
    const restored = {
      session_id:    msg.session_id,
      query:         msg.query,
      answer:        msg.answer,
      confidence:    msg.confidence,
      mode:          msg.mode,
      references:    msg.references   ?? [],
      agent_trace:   msg.agent_trace  ?? [],
      shap_analysis: msg.shap_analysis ?? {},
      message_id:    msg.id,
    };
    // Switch to classic and restore
    setAppMode("classic");
    classicRestoreRef.current.pending = restored;
    classicRestoreRef.current.clear   = () => { classicRestoreRef.current.pending = null; };
    setRestoreTick(t => t + 1);
  };

  const statusColor = (s) =>
    !s ? c.muted :
    (s === "ok" || s?.startsWith("ok"))  ? c.success :
    s?.startsWith("warning")              ? c.warn : c.danger;

  return (
    <div style={{
      minHeight: "100vh", background: c.bg,
      fontFamily: "'IBM Plex Mono','Fira Code',monospace", color: c.text,
    }}>
      {/* ── Header ── */}
      <div style={{
        background: c.surface, borderBottom: `1px solid ${c.border}`,
        padding: "0 20px", display: "flex", alignItems: "center", gap: 16, height: 54,
      }}>
        {/* Brand */}
        <span style={{ fontSize: 20 }}>⚖️</span>
        <div>
          <span style={{ color: c.accent, fontWeight: 800, fontSize: 14, letterSpacing: 1 }}>
            REG·ANALYST
          </span>
          <span style={{ color: c.muted, fontSize: 8, display: "block", letterSpacing: 2 }}>
            ON-PREMISE · MULTI-AGENT · FULLY LOCAL
          </span>
        </div>

        {/* ── Mode switcher ── */}
        <div style={{
          display: "flex", alignItems: "center", gap: 0,
          background: c.bg, border: `1px solid ${c.border}`,
          borderRadius: 8, padding: 3, marginLeft: 8,
        }}>
          {[
            { id: "classic", icon: "🔍", label: "Classic" },
            { id: "chat",    icon: "💬", label: "Chat"    },
          ].map(m => (
            <button key={m.id} onClick={() => setAppMode(m.id)} style={{
              background: appMode === m.id ? c.accent : "none",
              color:      appMode === m.id ? "#000"   : c.muted,
              border: "none", borderRadius: 6,
              padding: "5px 14px", fontSize: 11, fontWeight: appMode === m.id ? 800 : 400,
              cursor: "pointer", display: "flex", alignItems: "center", gap: 5,
              transition: "all 0.2s",
            }}>
              <span>{m.icon}</span>
              <span>{m.label}</span>
            </button>
          ))}
        </div>

        {/* Mode description */}
        <span style={{ color: c.muted, fontSize: 10 }}>
          {appMode === "classic"
            ? "Single query → 4-tab result panel"
            : "Conversation thread · AI remembers last 10 turns"}
        </span>

        <div style={{ flex: 1 }} />

        {/* Health */}
        <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
          {[["API", health?.api], ["Ollama", health?.ollama], ["ChromaDB", health?.chroma]].map(([label, val]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%",
                background: statusColor(val), boxShadow: `0 0 6px ${statusColor(val)}`,
              }} />
              <span style={{ color: c.muted, fontSize: 10 }}>{label}</span>
            </div>
          ))}
          {health?.saved_queries != null && (
            <span style={{ color: c.muted, fontSize: 10 }}>💬 {health.saved_queries} saved</span>
          )}
        </div>
      </div>

      {/* ── Body ── */}
      <div style={{ display: "flex", height: "calc(100vh - 54px)" }}>

        {/* Sidebar — always visible, shared by both modes */}
        <div style={{
          width: 290, borderRight: `1px solid ${c.border}`,
          background: c.surface, padding: 14, overflowY: "auto", flexShrink: 0,
        }}>
          <DocumentManager />
          <HistoryPanel onRestore={handleRestore} />
        </div>

        {/* Main content — swap based on mode */}
        {appMode === "classic"
          ? <ClassicMode key={restoreTick} onRestore={classicRestoreRef.current} />
          : <ChatMode />
        }
      </div>
    </div>
  );
}
