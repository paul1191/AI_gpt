"""
Session Store — SQLite-backed conversation history

Stores every query, answer, confidence score, references, agent trace and SHAP data.
Uses Python's built-in sqlite3 — no extra packages needed.

Database file: backend/sessions.db (auto-created on first run)

New in v1.3 (chat mode):
  get_recent_turns_for_prompt() — returns last N turns formatted as a
  conversation string ready to inject into the Synthesiser prompt.
"""
import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SESSION_DB_PATH", "./sessions.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and indexes if they don't exist. Safe to call on every startup."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL UNIQUE,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                query            TEXT NOT NULL,
                answer           TEXT NOT NULL,
                confidence       REAL NOT NULL,
                mode             TEXT NOT NULL,
                references_json  TEXT NOT NULL DEFAULT '[]',
                agent_trace_json TEXT NOT NULL DEFAULT '[]',
                shap_json        TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);

            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp);
        """)
        conn.commit()
        logger.info(f"Session DB initialised at '{DB_PATH}'")
    finally:
        conn.close()


def save_message(
    session_id: str,
    query: str,
    answer: str,
    confidence: float,
    mode: str,
    references: list,
    agent_trace: list,
    shap_analysis: dict,
) -> int:
    """Persist a query + response. Returns the new message id."""
    conn = _get_conn()
    try:
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
            (session_id, now),
        )
        cur = conn.execute(
            """INSERT INTO messages
               (session_id, timestamp, query, answer, confidence, mode,
                references_json, agent_trace_json, shap_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                now,
                query,
                answer,
                float(confidence),
                mode,
                json.dumps(references,    default=str),
                json.dumps(agent_trace,   default=str),
                json.dumps(shap_analysis, default=str),
            ),
        )
        conn.commit()
        msg_id = cur.lastrowid
        logger.info(f"Saved message id={msg_id} session={session_id[:8]}")
        return msg_id
    except Exception as e:
        logger.error(f"Failed to save message: {e}", exc_info=True)
        return -1
    finally:
        conn.close()


def get_recent_turns_for_prompt(session_id: str, n: int = 10) -> str:
    """
    Return the last N turns for a session as a formatted conversation string
    ready to inject into the Synthesiser LLM prompt.

    Format:
      User: <query>
      Assistant: <answer — stripped of the References Used section>

      User: <query>
      Assistant: <answer>
      ...

    The References Used section is stripped from answers because it is
    reconstructed fresh each turn — including it would bloat the prompt
    with redundant file/chunk metadata the LLM doesn't need for context.

    Returns empty string "" if no prior turns exist (first message in session).
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT query, answer
               FROM messages
               WHERE session_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (session_id, n),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return ""

    # Reverse so oldest turn is first (chronological order for LLM)
    turns = list(reversed(rows))
    lines = []
    for row in turns:
        q = row["query"].strip()
        # Strip the "--- References Used:" footer from the answer
        raw_answer = row["answer"] or ""
        answer_clean = raw_answer.split("\n\n---\n")[0].strip()
        # Truncate very long answers to keep prompt manageable (~400 chars each)
        if len(answer_clean) > 400:
            answer_clean = answer_clean[:400] + "…"
        lines.append(f"User: {q}\nAssistant: {answer_clean}")

    return "\n\n".join(lines)


def get_session_history(session_id: str) -> List[Dict]:
    """All messages for a session, oldest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT id, session_id, timestamp, query, answer, confidence, mode,
                      references_json, agent_trace_json, shap_json
               FROM messages WHERE session_id = ?
               ORDER BY timestamp ASC""",
            (session_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_all_sessions() -> List[Dict]:
    """All sessions with message count, avg confidence, last activity."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT
                 s.session_id,
                 s.created_at,
                 COUNT(m.id)       AS message_count,
                 MAX(m.timestamp)  AS last_activity,
                 MIN(m.query)      AS first_query,
                 AVG(m.confidence) AS avg_confidence
               FROM sessions s
               LEFT JOIN messages m ON s.session_id = m.session_id
               GROUP BY s.session_id
               ORDER BY last_activity DESC""",
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search_history(keyword: str, limit: int = 20) -> List[Dict]:
    """Full-text search across all stored queries and answers."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT id, session_id, timestamp, query, answer, confidence, mode,
                      references_json, agent_trace_json, shap_json
               FROM messages
               WHERE query LIKE ? OR answer LIKE ?
               ORDER BY timestamp DESC LIMIT ?""",
            (f"%{keyword}%", f"%{keyword}%", limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_recent_messages(limit: int = 20) -> List[Dict]:
    """Most recent messages across all sessions."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT id, session_id, timestamp, query, answer, confidence, mode,
                      references_json, agent_trace_json, shap_json
               FROM messages ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_message_by_id(message_id: int) -> Optional[Dict]:
    """Single message by integer id."""
    conn = _get_conn()
    try:
        row = conn.execute(
            """SELECT id, session_id, timestamp, query, answer, confidence, mode,
                      references_json, agent_trace_json, shap_json
               FROM messages WHERE id = ?""",
            (message_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_session(session_id: str) -> int:
    """Delete all messages and session row. Returns messages deleted."""
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        deleted = cur.rowcount
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        logger.info(f"Deleted session {session_id[:8]} ({deleted} messages)")
        return deleted
    finally:
        conn.close()


def get_stats() -> Dict:
    """Aggregate statistics about stored history."""
    conn = _get_conn()
    try:
        row = conn.execute(
            """SELECT
                 COUNT(*)                    AS total_messages,
                 COUNT(DISTINCT session_id)  AS total_sessions,
                 AVG(confidence)             AS avg_confidence,
                 MIN(timestamp)              AS earliest,
                 MAX(timestamp)              AS latest
               FROM messages"""
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict:
    """Convert sqlite3.Row to plain dict, deserialising JSON blob fields."""
    d = dict(row)
    for json_field, clean_key in [
        ("references_json",  "references"),
        ("agent_trace_json", "agent_trace"),
        ("shap_json",        "shap_analysis"),
    ]:
        if json_field in d:
            try:
                d[clean_key] = json.loads(d.pop(json_field))
            except (json.JSONDecodeError, TypeError):
                d[clean_key] = []
                d.pop(json_field, None)
    return d
