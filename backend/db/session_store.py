"""
Session Store — SQLite-backed conversation history  v1.5

Changes from v1.3:
  - Added faithfulness_json, lime_json, trust_json columns to messages table
  - _migrate_db() adds missing columns to existing databases without data loss
    (safe to run on an existing sessions.db — ALTER TABLE IF NOT EXISTS equivalent)
  - save_message() accepts faithfulness_data, lime_data, trust_data
  - _row_to_dict() deserialises all 6 JSON blob fields
  - get_recent_turns_for_prompt() unchanged — chat memory unaffected

Tables:
  sessions  — one row per unique session_id
  messages  — one row per query/response pair, linked to sessions
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
    """
    Create tables and indexes if they don't exist, then migrate any
    existing database to add new columns. Safe to call multiple times.
    """
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL UNIQUE,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT NOT NULL,
                timestamp           TEXT NOT NULL,
                query               TEXT NOT NULL,
                answer              TEXT NOT NULL,
                confidence          REAL NOT NULL,
                mode                TEXT NOT NULL,
                references_json     TEXT NOT NULL DEFAULT '[]',
                agent_trace_json    TEXT NOT NULL DEFAULT '[]',
                shap_json           TEXT NOT NULL DEFAULT '{}',
                faithfulness_json   TEXT NOT NULL DEFAULT '{}',
                lime_json           TEXT NOT NULL DEFAULT '{}',
                trust_json          TEXT NOT NULL DEFAULT '{}'
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

    # Migrate existing DB — adds new columns if they don't exist yet
    _migrate_db()


def _migrate_db():
    """
    Add v1.5 columns to an existing database that was created before they existed.
    SQLite doesn't support IF NOT EXISTS on ALTER TABLE, so we check the
    existing column list first.
    """
    conn = _get_conn()
    try:
        # Get current columns
        cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        new_cols = {
            "faithfulness_json": "TEXT NOT NULL DEFAULT '{}'",
            "lime_json":         "TEXT NOT NULL DEFAULT '{}'",
            "trust_json":        "TEXT NOT NULL DEFAULT '{}'",
        }
        for col, definition in new_cols.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {col} {definition}")
                logger.info(f"Migrated DB: added column '{col}'")
        conn.commit()
    except Exception as e:
        logger.warning(f"DB migration warning: {e}")
    finally:
        conn.close()


def save_message(
    session_id:        str,
    query:             str,
    answer:            str,
    confidence:        float,
    mode:              str,
    references:        list,
    agent_trace:       list,
    shap_analysis:     dict,
    faithfulness_data: dict = None,
    lime_data:         dict = None,
    trust_data:        dict = None,
) -> int:
    """
    Persist a query + response to the messages table.
    New fields faithfulness_data, lime_data, trust_data default to {} if not supplied
    (backwards compatible with callers that don't yet pass them).
    Returns the new message id.
    """
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
                references_json, agent_trace_json, shap_json,
                faithfulness_json, lime_json, trust_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                now,
                query,
                answer,
                float(confidence),
                mode,
                json.dumps(references,            default=str),
                json.dumps(agent_trace,           default=str),
                json.dumps(shap_analysis or {},   default=str),
                json.dumps(faithfulness_data or {}, default=str),
                json.dumps(lime_data or {},        default=str),
                json.dumps(trust_data or {},       default=str),
            ),
        )
        conn.commit()
        msg_id = cur.lastrowid
        logger.info(f"Saved message id={msg_id} for session {session_id[:8]}")
        return msg_id
    except Exception as e:
        logger.error(f"Failed to save message: {e}", exc_info=True)
        return -1
    finally:
        conn.close()


def get_session_history(session_id: str) -> List[Dict]:
    """Return all messages for a session, oldest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT id, session_id, timestamp, query, answer, confidence, mode,
                      references_json, agent_trace_json, shap_json,
                      faithfulness_json, lime_json, trust_json
               FROM messages
               WHERE session_id = ?
               ORDER BY timestamp ASC""",
            (session_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_all_sessions() -> List[Dict]:
    """List all sessions with message count, first query, and last activity."""
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
    """Full-text search across stored queries and answers."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT id, session_id, timestamp, query, answer, confidence, mode,
                      references_json, agent_trace_json, shap_json,
                      faithfulness_json, lime_json, trust_json
               FROM messages
               WHERE query LIKE ? OR answer LIKE ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (f"%{keyword}%", f"%{keyword}%", limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_recent_messages(limit: int = 20) -> List[Dict]:
    """Get most recent messages across all sessions."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT id, session_id, timestamp, query, answer, confidence, mode,
                      references_json, agent_trace_json, shap_json,
                      faithfulness_json, lime_json, trust_json
               FROM messages
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_message_by_id(message_id: int) -> Optional[Dict]:
    """Get a single saved message by its integer id."""
    conn = _get_conn()
    try:
        row = conn.execute(
            """SELECT id, session_id, timestamp, query, answer, confidence, mode,
                      references_json, agent_trace_json, shap_json,
                      faithfulness_json, lime_json, trust_json
               FROM messages WHERE id = ?""",
            (message_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_session(session_id: str) -> int:
    """Delete all messages and the session row. Returns number of messages deleted."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
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
                 COUNT(*)                   AS total_messages,
                 COUNT(DISTINCT session_id) AS total_sessions,
                 AVG(confidence)            AS avg_confidence,
                 MIN(timestamp)             AS earliest,
                 MAX(timestamp)             AS latest
               FROM messages"""
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def get_recent_turns_for_prompt(session_id: str, n: int = 10) -> str:
    """
    Fetch last N turns for a session and format as a conversation string
    for injection into the LLM prompt.
    Strips the References Used footer from answers to keep prompts lean.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT query, answer FROM messages
               WHERE session_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (session_id, n),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return ""

    turns = []
    for row in reversed(rows):
        q = row["query"]
        a = row["answer"].split("\n\n---\n")[0].strip()   # strip References footer
        turns.append(f"User: {q}\nAssistant: {a}")

    return "\n\n".join(turns)


# ── Internal helper ───────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> Dict:
    """Convert sqlite3.Row to plain dict, deserialising all JSON blob fields."""
    d = dict(row)
    json_fields = [
        ("references_json",   "references"),
        ("agent_trace_json",  "agent_trace"),
        ("shap_json",         "shap_analysis"),
        ("faithfulness_json", "faithfulness_data"),
        ("lime_json",         "lime_data"),
        ("trust_json",        "trust_data"),
    ]
    for json_field, clean_key in json_fields:
        if json_field in d:
            try:
                d[clean_key] = json.loads(d.pop(json_field))
            except (json.JSONDecodeError, TypeError):
                d[clean_key] = {}
                d.pop(json_field, None)
    return d
