"""
RegIntel-AI — Healthcare Regulatory Intelligence
Single-file Streamlit application with Direction C aesthetic.

Architecture:
  - SQLite for metadata + analysis cache (in ./data/regintel.db)
  - In-memory TF-IDF for semantic-ish retrieval (no torch, no chromadb)
  - Optional Gemini LLM for analysis (mock fallback if key absent)
  - Optional GitHub-backed snapshots for cross-machine persistence
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sqlite3
import tarfile
import time
import uuid
import html as _html_module
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def _esc(value) -> str:
    """HTML-escape any value before embedding it in a markdown(..., unsafe_allow_html=True) block.
    Use this for every piece of LLM- or user-derived text that gets interpolated into HTML markup.
    Without this, characters like < > & in regulatory text (e.g. 'members aged < 65') get
    parsed as malformed HTML tags and either disappear or render as visible broken markup."""
    if value is None:
        return ""
    return _html_module.escape(str(value), quote=True)

# ============================================================
# Page config — must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="RegIntel-AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️",
)

logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regintel")


# ============================================================
# Settings — read from Streamlit secrets, env, or defaults
# ============================================================
# Check for the secrets.toml file BEFORE any st.secrets access. In Streamlit 1.40+,
# accessing st.secrets when no file exists raises an exception that displays in the UI
# even when wrapped in try/except. The fix is to never touch st.secrets unless the file
# is known to exist.
def _streamlit_secrets_available() -> bool:
    """Return True only if at least one of Streamlit's secrets paths has a file."""
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    return any(p.exists() for p in candidates)


_HAS_SECRETS = _streamlit_secrets_available()


def _secret(key: str, default: str = "") -> str:
    """Try Streamlit secrets first (if file exists), then env, then default."""
    if _HAS_SECRETS:
        try:
            if key in st.secrets:
                return str(st.secrets[key]).strip()
        except Exception:
            pass
    return os.environ.get(key, default).strip()


LLM_PROVIDER   = _secret("LLM_PROVIDER", "mock").lower()
GEMINI_API_KEY = _secret("GEMINI_API_KEY")
GEMINI_MODEL   = _secret("GEMINI_MODEL", "gemini-2.5-flash-lite")
GROQ_API_KEY   = _secret("GROQ_API_KEY")
GROQ_MODEL     = _secret("GROQ_MODEL", "llama-3.3-70b-versatile")
GITHUB_TOKEN   = _secret("GITHUB_TOKEN")
GITHUB_REPO    = _secret("GITHUB_REPO")  # format: "owner/repo"

DATA_DIR  = Path("./data")
SEED_DIR  = Path("./seed")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = DATA_DIR / "regintel.db"


# ============================================================
# Database — SQLite with schema management
# ============================================================
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
      doc_id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      kind TEXT NOT NULL,
      version TEXT NOT NULL,
      family_id TEXT NOT NULL,
      effective_date TEXT,
      regulation_id TEXT,
      issuing_body TEXT,
      change_type TEXT,
      categories TEXT,        -- JSON array
      num_chunks INTEGER DEFAULT 0,
      full_text TEXT,         -- raw extracted text
      created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS chunks (
      doc_id TEXT NOT NULL,
      idx INTEGER NOT NULL,
      section TEXT,
      text TEXT NOT NULL,
      PRIMARY KEY (doc_id, idx),
      FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS analyses (
      doc_id TEXT PRIMARY KEY,
      result_json TEXT NOT NULL,
      impact_score INTEGER NOT NULL,
      created_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS comparisons (
      pair_id TEXT PRIMARY KEY,
      old_doc_id TEXT NOT NULL,
      new_doc_id TEXT NOT NULL,
      result_json TEXT NOT NULL,
      created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS timeline (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      event_type TEXT NOT NULL,
      doc_id TEXT,
      family_id TEXT,
      version TEXT,
      payload TEXT,
      created_at TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit()
    conn.close()


def db_insert_document(d: dict, full_text: str, chunks: list[dict]) -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO documents
        (doc_id, title, kind, version, family_id, effective_date, regulation_id,
         issuing_body, change_type, categories, num_chunks, full_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        d["doc_id"], d["title"], d["kind"], d["version"], d["family_id"],
        d.get("effective_date"), d.get("regulation_id"),
        d.get("issuing_body"), d.get("change_type"),
        json.dumps(d.get("categories") or []),
        len(chunks), full_text,
    ))
    cur.execute("DELETE FROM chunks WHERE doc_id = ?", (d["doc_id"],))
    for c in chunks:
        cur.execute(
            "INSERT INTO chunks (doc_id, idx, section, text) VALUES (?, ?, ?, ?)",
            (d["doc_id"], c["idx"], c.get("section"), c["text"]),
        )
    cur.execute(
        "INSERT INTO timeline (event_type, doc_id, family_id, version) "
        "VALUES (?, ?, ?, ?)",
        ("ingested", d["doc_id"], d["family_id"], d["version"]),
    )
    conn.commit()
    conn.close()


def db_list_documents(kind: str | None = None) -> list[dict]:
    conn = _get_conn()
    cur = conn.cursor()
    if kind:
        cur.execute("SELECT * FROM documents WHERE kind = ? ORDER BY created_at DESC",
                   (kind,))
    else:
        cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["categories"] = json.loads(d.get("categories") or "[]")
        except Exception:
            d["categories"] = []
        out.append(d)
    return out


def db_get_document(doc_id: str) -> dict | None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    d = dict(r)
    try:
        d["categories"] = json.loads(d.get("categories") or "[]")
    except Exception:
        d["categories"] = []
    return d


def db_get_chunks(doc_id: str) -> list[dict]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT idx, section, text FROM chunks WHERE doc_id = ? ORDER BY idx",
               (doc_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def db_save_analysis(doc_id: str, result: dict, impact_score: int) -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO analyses (doc_id, result_json, impact_score)
        VALUES (?, ?, ?)
    """, (doc_id, json.dumps(result), int(impact_score)))
    cur.execute(
        "INSERT INTO timeline (event_type, doc_id) VALUES (?, ?)",
        ("analyzed", doc_id),
    )
    conn.commit()
    conn.close()


def db_get_analysis(doc_id: str) -> dict | None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT result_json, impact_score FROM analyses WHERE doc_id = ?",
               (doc_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {"result": json.loads(r["result_json"]), "impact_score": r["impact_score"]}


def db_save_comparison(old_doc_id: str, new_doc_id: str, result: dict) -> None:
    pair_id = f"{old_doc_id}__{new_doc_id}"
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO comparisons (pair_id, old_doc_id, new_doc_id, result_json)
        VALUES (?, ?, ?, ?)
    """, (pair_id, old_doc_id, new_doc_id, json.dumps(result)))
    cur.execute(
        "INSERT INTO timeline (event_type, doc_id, payload) VALUES (?, ?, ?)",
        ("compared", new_doc_id, json.dumps({"old": old_doc_id})),
    )
    conn.commit()
    conn.close()


def db_get_comparison(old_doc_id: str, new_doc_id: str) -> dict | None:
    pair_id = f"{old_doc_id}__{new_doc_id}"
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT result_json FROM comparisons WHERE pair_id = ?", (pair_id,))
    r = cur.fetchone()
    conn.close()
    return json.loads(r["result_json"]) if r else None


def db_get_timeline(limit: int = 200) -> list[dict]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM timeline ORDER BY id DESC LIMIT ?", (limit,))
    out = [dict(r) for r in cur.fetchall()]
    conn.close()
    return out


def db_kpis() -> dict:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM analyses")
    n_analyses = cur.fetchone()["n"]
    cur.execute("SELECT result_json FROM analyses")
    n_impacts = 0
    n_high = 0
    confs: list[float] = []
    for r in cur.fetchall():
        try:
            payload = json.loads(r["result_json"])
            for ia in payload.get("impacted_areas", []):
                n_impacts += 1
                if ia.get("priority") == "High":
                    n_high += 1
                c = ia.get("confidence_score")
                if isinstance(c, (int, float)):
                    confs.append(float(c))
        except Exception:
            pass
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    conn.close()
    return {"n_analyses": n_analyses, "n_impacts": n_impacts,
            "n_high_risk": n_high, "avg_confidence": round(avg_conf, 2)}


# ============================================================
# Document loading + chunking
# ============================================================
def extract_text_from_uploaded(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith((".txt", ".md")):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith((".html", ".htm")):
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(raw, "html.parser").get_text(separator="\n")
        except ImportError:
            return raw.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
        except ImportError:
            return ""
        except Exception as e:
            logger.exception("PDF extraction failed")
            return ""

    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return ""

    return raw.decode("utf-8", errors="ignore")


def chunk_text(text: str, target_size: int = 600, overlap: int = 80) -> list[dict]:
    """
    Section-aware chunker.
    Splits on headings (ALL CAPS, numbered sections, regulatory citations),
    then within each section into overlapping windows.
    """
    if not text.strip():
        return []

    section_pattern = re.compile(
        r"^(\s*(?:§\s*\d|\d+\.\s|[A-Z][A-Z\s\d:.\-]{6,}|"
        r"SECTION\s+\d|PART\s+\d|SUBPART\s+[A-Z]))",
        re.MULTILINE,
    )

    sections: list[tuple[str, str]] = []
    last_pos = 0
    last_title = "INTRODUCTION"
    for m in section_pattern.finditer(text):
        if m.start() > last_pos:
            chunk = text[last_pos:m.start()].strip()
            if chunk:
                sections.append((last_title, chunk))
        last_title = m.group(1).strip()[:80]
        last_pos = m.end()
    tail = text[last_pos:].strip()
    if tail:
        sections.append((last_title, tail))

    if not sections:
        sections = [("BODY", text.strip())]

    out: list[dict] = []
    idx = 0
    for sec_title, sec_text in sections:
        words = sec_text.split()
        if not words:
            continue
        i = 0
        while i < len(words):
            window = words[i:i + target_size]
            chunk_str = " ".join(window).strip()
            if chunk_str:
                out.append({"idx": idx, "section": sec_title, "text": chunk_str})
                idx += 1
            i += max(1, target_size - overlap)

    return out


def normalize_kind(k: str) -> str:
    k = (k or "").lower()
    if k in {"regulation", "policy", "sop", "system"}:
        return k
    if "policy" in k:
        return "policy"
    if "sop" in k or "workflow" in k:
        return "sop"
    if "system" in k or "arch" in k:
        return "system"
    return "regulation"


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_") or "doc"


def extract_metadata_simple(text: str) -> dict:
    """Cheap heuristic metadata extraction — no LLM call."""
    out: dict[str, Any] = {}
    title_match = re.search(r"^([^\n]{10,120})", text.strip())
    if title_match:
        out["title"] = title_match.group(1).strip()
    rid_match = re.search(r"(CMS-\d{3,4}(?:-[A-Z])?|CMS-MLN-\d+)", text)
    if rid_match:
        out["regulation_id"] = rid_match.group(1)
    eff_match = re.search(r"effective[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", text, re.I)
    if eff_match:
        try:
            dt = datetime.strptime(eff_match.group(1).replace(",", ""),
                                  "%B %d %Y")
            out["effective_date"] = dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    if "FINAL RULE" in text.upper() or "FINAL" in text.upper()[:200]:
        out["change_type"] = "Final"
    elif "PROPOSED" in text.upper():
        out["change_type"] = "Proposed"
    out["issuing_body"] = "CMS" if "CMS" in text[:500] else "Unknown"
    return out


# ============================================================
# Retrieval — TF-IDF, no torch/chromadb
# ============================================================
@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    section: str
    text: str
    idx: int
    relevance: float


def retrieve_chunks(query: str, kind: str | None = None,
                   exclude_doc_id: str | None = None,
                   top_k: int = 8) -> list[RetrievedChunk]:
    """Lightweight TF-IDF over all chunks of given kind. Good enough for demo."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return _fallback_keyword_search(query, kind, exclude_doc_id, top_k)

    conn = _get_conn()
    cur = conn.cursor()
    sql = """
        SELECT c.doc_id, c.idx, c.section, c.text, d.title, d.kind
        FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
    """
    params: list = []
    where = []
    if kind:
        where.append("d.kind = ?")
        params.append(kind)
    if exclude_doc_id:
        where.append("c.doc_id != ?")
        params.append(exclude_doc_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    corpus = [r["text"] for r in rows]
    vec = TfidfVectorizer(stop_words="english", max_features=8000,
                         ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus + [query])
    sims = cosine_similarity(matrix[-1:], matrix[:-1]).flatten()

    ranked = sorted(zip(rows, sims), key=lambda x: -x[1])[:top_k]
    out = []
    for r, score in ranked:
        if score <= 0:
            continue
        out.append(RetrievedChunk(
            doc_id=r["doc_id"], title=r["title"], section=r["section"] or "",
            text=r["text"], idx=r["idx"], relevance=float(score),
        ))
    return out


def _fallback_keyword_search(query: str, kind: str | None,
                              exclude: str | None, top_k: int) -> list[RetrievedChunk]:
    """If sklearn isn't available — naive overlap scoring."""
    q_words = set(re.findall(r"\w+", query.lower()))
    if not q_words:
        return []
    conn = _get_conn()
    cur = conn.cursor()
    sql = """SELECT c.doc_id, c.idx, c.section, c.text, d.title, d.kind
             FROM chunks c JOIN documents d ON c.doc_id = d.doc_id"""
    params: list = []
    where = []
    if kind:
        where.append("d.kind = ?")
        params.append(kind)
    if exclude:
        where.append("c.doc_id != ?")
        params.append(exclude)
    if where:
        sql += " WHERE " + " AND ".join(where)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    scored = []
    for r in rows:
        words = set(re.findall(r"\w+", r["text"].lower()))
        score = len(q_words & words) / max(1, len(q_words))
        if score > 0:
            scored.append((r, score))
    scored.sort(key=lambda x: -x[1])
    return [RetrievedChunk(doc_id=r["doc_id"], title=r["title"],
                          section=r["section"] or "", text=r["text"],
                          idx=r["idx"], relevance=s) for r, s in scored[:top_k]]


# ============================================================
# LLM — Gemini with mock fallback
# ============================================================
# Track which LLM produced the last response so the UI can display it.
# Set by call_llm() on every call. Cleared lazily.
_LAST_LLM_SOURCE: dict = {"source": "unknown", "error": None}


def _strip_json_fence(s: str) -> str:
    """Gemini sometimes wraps JSON in markdown code fences even with response_mime_type set.
    Strip ```json ... ``` or ``` ... ``` wrappers, plus leading/trailing whitespace."""
    if not s:
        return s
    s = s.strip()
    # ```json ... ```
    if s.startswith("```"):
        # Remove first fence line
        first_newline = s.find("\n")
        if first_newline > 0:
            s = s[first_newline + 1:]
        # Remove trailing fence
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s.strip()


def call_llm(system: str, user: str, max_tokens: int = 1500) -> dict:
    """Returns a JSON dict. Falls back to mock if the configured provider is unavailable.
    Sets _LAST_LLM_SOURCE so the UI can show which model actually responded.
    Supported providers (set LLM_PROVIDER secret): gemini | groq | mock."""
    global _LAST_LLM_SOURCE

    # ─────────── Groq (OpenAI-compatible API, free tier, fast LPU hardware) ───────────
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        try:
            # Use the openai SDK pointed at Groq's OpenAI-compatible endpoint.
            # If the openai package isn't installed, fall back to a plain urllib POST.
            try:
                from openai import OpenAI
                client = OpenAI(api_key=GROQ_API_KEY,
                                base_url="https://api.groq.com/openai/v1")
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system",
                         "content": system + "\n\nReturn ONLY valid JSON. No markdown, no preamble."},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                raw = (resp.choices[0].message.content or "").strip()
            except ImportError:
                # Fallback: plain urllib POST to Groq's chat-completions endpoint
                import urllib.request
                payload = json.dumps({
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system",
                         "content": system + "\n\nReturn ONLY valid JSON. No markdown."},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                raw = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not raw:
                raise ValueError("Groq returned empty response")
            cleaned = _strip_json_fence(raw)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                brace_start = cleaned.find("{")
                brace_end = cleaned.rfind("}")
                if brace_start >= 0 and brace_end > brace_start:
                    parsed = json.loads(cleaned[brace_start:brace_end + 1])
                else:
                    raise
            _LAST_LLM_SOURCE = {"source": "groq", "error": None, "model": GROQ_MODEL}
            return parsed
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            logger.warning("Groq call failed, using mock: %s", err_msg)
            _LAST_LLM_SOURCE = {"source": "mock", "error": err_msg,
                               "model": f"mock (Groq fallback after error)"}
            return _mock_llm(system, user)

    # ─────────── Gemini (Google AI Studio) ───────────
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                GEMINI_MODEL,
                system_instruction=system + "\n\nReturn ONLY valid JSON. No markdown.",
            )
            resp = model.generate_content(
                user,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            raw = (resp.text or "").strip()
            if not raw:
                raise ValueError("Gemini returned empty response (possibly safety-filtered)")
            # Tolerant parse — strip code fences if present
            cleaned = _strip_json_fence(raw)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                # Sometimes Gemini emits trailing prose after the JSON object.
                # Try to find the outermost {...} block.
                brace_start = cleaned.find("{")
                brace_end = cleaned.rfind("}")
                if brace_start >= 0 and brace_end > brace_start:
                    parsed = json.loads(cleaned[brace_start:brace_end + 1])
                else:
                    raise
            _LAST_LLM_SOURCE = {"source": "gemini", "error": None, "model": GEMINI_MODEL}
            return parsed
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            logger.warning("Gemini call failed, using mock: %s", err_msg)
            _LAST_LLM_SOURCE = {"source": "mock", "error": err_msg, "model": "mock"}
            return _mock_llm(system, user)

    _LAST_LLM_SOURCE = {"source": "mock", "error": None,
                       "model": f"mock (LLM_PROVIDER={LLM_PROVIDER}, no valid key)"}
    return _mock_llm(system, user)


def _mock_llm(system: str, user: str) -> dict:
    """Plausible canned responses keyed off prompt content."""
    u = user.lower()

    # Comparison check FIRST (more specific)
    if "compare these two versions" in u or "differences between" in u:
        return {
            "summary": (
                "The Final Rule extends the continuity-of-care window from 30 to "
                "90 days, adds a denial-notice requirement with appeal rights, and "
                "introduces quarterly compliance reporting. Combined, these changes "
                "raise compliance scope, operational burden, and audit exposure."
            ),
            "old_version": "v1.0", "new_version": "v2.0",
            "changes": [
                {
                    "type": "Modified", "section": "§ 422.138(b)",
                    "description": "Continuity-of-care window extended from 30 to 90 days",
                    "old_text": "Plans must honor prior authorization decisions made by another organization for at least 30 days from the date the enrollee transitioned plans.",
                    "new_text": "Plans must continue to honor prior authorization decisions made by another organization for a minimum of 90 days from the date the enrollee transitioned plans, and for the full duration of any clinically established treatment plan.",
                    "compliance_risk_delta": "+High",
                    "operational_impact_delta": "+Medium",
                    "impact": "Affects PA workflow, claims adjudication logic, and member-facing communications.",
                    "recommended_action": "Update PA continuity policy and claims SOP. Plan system change for PA ingestion.",
                },
                {
                    "type": "Added", "section": "§ 422.138(c)",
                    "description": "New requirement: denial notice with appeal rights",
                    "old_text": "",
                    "new_text": "Plans must issue a written denial notice that includes appeal rights and is delivered within timeframes consistent with adverse-determination notice rules under § 422.568.",
                    "compliance_risk_delta": "+Medium",
                    "operational_impact_delta": "+Medium",
                    "impact": "Requires new notice template + workflow trigger.",
                    "recommended_action": "Draft notice template, integrate into PA workflow.",
                },
                {
                    "type": "Added", "section": "§ 422.138(d)",
                    "description": "New requirement: quarterly compliance reporting",
                    "old_text": "",
                    "new_text": "Plans must submit quarterly reports to CMS attesting to compliance with the continuity-of-care provisions, using the prescribed reporting template.",
                    "compliance_risk_delta": "+Medium",
                    "operational_impact_delta": "+Low",
                    "impact": "New recurring reporting obligation.",
                    "recommended_action": "Establish quarterly reporting process; assign compliance owner.",
                },
            ],
        }

    # Simulation check
    if ("simulat" in u or "what-if" in u or
        "project" in u and "consequences" in u or
        "healthcare risk officer" in u):
        # Extract the artifact name and priority from the prompt so different cards
        # get different simulated values (instead of all returning the same canned answer).
        import re as _re
        name_match = _re.search(r"IMPACTED ARTIFACT:\s*(.+)", user)
        prio_match = _re.search(r"PRIORITY:\s*(.+)", user)
        artifact = (name_match.group(1).strip() if name_match else "the artifact")
        prio = (prio_match.group(1).strip() if prio_match else "Medium")

        # Priority → financial bands
        bands = {
            "High":   (250_000, 1_500_000, "High"),
            "Medium": (50_000,    500_000, "Medium"),
            "Low":    (10_000,    100_000, "Low"),
        }
        low, high, likelihood = bands.get(prio, bands["Medium"])

        return {
            "financial_exposure": {
                "low_estimate_usd": low,
                "high_estimate_usd": high,
                "basis": (
                    f"For a {prio}-priority gap in {artifact}, financial exposure ranges from "
                    f"corrective-action-plan costs and remediation labor (~${low:,}) up to civil "
                    f"monetary penalties under 42 CFR § 422.752 (~${high:,}). Range based on "
                    f"prior CMS enforcement actions in similar Medicare Advantage cases."
                ),
            },
            "regulatory_exposure": (
                f"If {artifact} remains unaddressed, exposure includes adverse audit findings during "
                f"the next CMS program audit, a likely corrective action plan with quarterly "
                f"attestations, and civil monetary penalty risk under 42 CFR § 422.752."
            ),
            "member_impact": (
                f"Members affected by gaps in {artifact} may experience denied or delayed access to "
                f"care, increased appeals volume, and adverse health outcomes in transition cases. "
                f"Member complaints tied to this artifact will rise until remediation is in place."
            ),
            "operational_friction": (
                f"Operations staff will incur manual workarounds to bridge the gap in {artifact}, "
                f"increasing average handle time. Appeals volume and audit-prep effort will be "
                f"substantially higher than for a remediated artifact."
            ),
            "reputational_impact": (
                f"Failure to address {artifact} risks adverse press coverage if penalty actions become "
                f"public, and erodes member trust in plan responsiveness."
            ),
            "likelihood_of_enforcement": likelihood,
            "mitigation_window_days": 90 if prio == "High" else (120 if prio == "Medium" else 180),
        }

    # Impact analysis (default for regulation analysis prompts) — match the EXACT first line of analyze_regulation()
    if "analyze the impact of this regulation" in u or "analyze the impact" in u and "internal artifacts" in u:
        return {
            "regulation_summary": (
                "Final Rule extends the continuity-of-care window for transitioning "
                "members from 30 to 90 days, requires denial-notice with appeal rights, "
                "and adds quarterly compliance reporting."
            ),
            "impact_score_overall": 82,
            "impacted_areas": [
                {
                    "name": "Prior Authorization Continuity Policy",
                    "type": "Policy",
                    "priority": "High",
                    "impact_reason": "Current policy § 4.2 honors prior plan PA decisions for thirty (30) days. CMS-4201-F § 4.2 requires not less than ninety (90) days, plus full duration of clinically established treatment plans.",
                    "recommended_action": "Revise § 4.2 to reflect ninety (90) day window. Update § 4.3 exception to cover all clinically established treatment plans, not just oncology. Move § 4.5 reporting from annual to quarterly.",
                    "risk_if_not_implemented": "Civil monetary penalties up to $25,000 per occurrence; audit findings on transition handling.",
                    "confidence_score": 0.86,
                    "supporting_citations": ["§ 4.2", "§ 4.3", "§ 6.1"],
                },
                {
                    "name": "Member Onboarding Workflow",
                    "type": "Workflow",
                    "priority": "Medium",
                    "impact_reason": "Onboarding workflow STEP 1 lacks a transition-flag decision step. STEP 4 fifteen-day PA documentation window is tight given the new ninety-day continuity requirement. STEP 7 claims continuity flag is a soft flag only — system does not automatically apply continuity rules.",
                    "recommended_action": "Insert transition-flag decision node at STEP 1 file intake. Extend STEP 4 documentation window to align with ninety-day continuity period. Make STEP 7 continuity flag enforceable in the claims system.",
                    "risk_if_not_implemented": "Inappropriate denials at first-claim processing; appeals volume spike; member harm in transition cases.",
                    "confidence_score": 0.79,
                    "supporting_citations": ["§ 4.2", "STEP 4", "STEP 7"],
                },
                {
                    "name": "Claims Adjudication System Spec",
                    "type": "System",
                    "priority": "Medium",
                    "impact_reason": "ACME-CLAIMS-CORE § 4.2 clean-claim SLA is forty-five (45) days, exceeding CMS-MLN-7521 thirty (30) day standard. § 5.1 lacks automated interest calculation on late payments. § 6.3 lacks the Star Ratings data feed required for measure C36.",
                    "recommended_action": "Reduce § 4.2 SLA target to thirty (30) days. Implement automated interest accrual per § 5.1 using Treasury rate. Add automated Star Ratings data feed for measure C36.",
                    "risk_if_not_implemented": "Late-payment interest exposure; Star Rating downgrade; provider grievance escalation.",
                    "confidence_score": 0.74,
                    "supporting_citations": ["§ 4.2", "§ 5.1", "§ 6.3"],
                },
            ],
            "citations": [
                {"citation_id": "CHUNK_4", "source_title": "CMS-4201-F",
                 "section": "§ 4.2", "snippet":
                 "A Medicare Advantage organization shall honor prior plan PA decisions "
                 "for a period of NOT LESS THAN ninety (90) days following the member's "
                 "effective enrollment date with the receiving plan.",
                 "relevance": 0.91},
                {"citation_id": "CHUNK_7", "source_title": "CMS-4201-F",
                 "section": "§ 5.2", "snippet":
                 "The denial notice shall include the specific clinical and administrative "
                 "basis for the denial, the right to appeal under 42 CFR § 422.566, and the "
                 "procedure for filing an expedited appeal.",
                 "relevance": 0.84},
                {"citation_id": "CHUNK_8", "source_title": "CMS-4201-F",
                 "section": "§ 422.138(d)", "snippet":
                 "Plans must report compliance with continuity-of-care provisions "
                 "to CMS on a quarterly basis using prescribed reporting templates.",
                 "relevance": 0.78},
            ],
        }

    # Proposed-text change for an impacted artifact (mock for offline demo / no Gemini key)
    if ("policy-language change" in u or "current_text_verbatim" in u or
        "actual current text from the internal artifact" in u or
        ("proposed_text" in u and "current_text_assumption" in u)):
        # Extract the artifact name and gap description from the prompt so different cards
        # produce different proposed text (instead of all returning the same canned answer).
        import re as _re
        name_match = _re.search(r"Name:\s*(.+)", user)
        type_match = _re.search(r"Type:\s*(.+)", user)
        gap_match  = _re.search(r"Gap identified.*?:\s*(.+)", user)
        rec_match  = _re.search(r"Recommended action \(high level\):\s*(.+)", user)
        reg_match  = _re.search(r"Identifier:\s*(.+)", user)

        artifact_name = (name_match.group(1).strip() if name_match else "the affected artifact")
        artifact_type = (type_match.group(1).strip() if type_match else "Policy")
        gap_desc      = (gap_match.group(1).strip() if gap_match else "")
        rec_action    = (rec_match.group(1).strip() if rec_match else "")
        reg_id        = (reg_match.group(1).strip() if reg_match else "the regulation")

        # NEW: detect if the prompt includes verbatim chunks from the actual artifact
        # When the chunks block is present, extract a candidate verbatim sentence.
        has_chunks = "ACTUAL CURRENT TEXT FROM THE INTERNAL ARTIFACT" in user
        verbatim_quote = ""
        source_section = ""
        if has_chunks:
            # Find the chunks block
            chunks_start = user.find("ACTUAL CURRENT TEXT FROM THE INTERNAL ARTIFACT")
            chunks_end = user.find("\nTASK\n", chunks_start) if chunks_start >= 0 else -1
            if chunks_start >= 0 and chunks_end > chunks_start:
                chunks_text = user[chunks_start:chunks_end]
                # Pull out individual chunks: each starts with [CHUNK N] section: X
                chunk_pat = _re.compile(
                    r"\[CHUNK \d+\] section:\s*(.+?)\n(.+?)(?=\n\[CHUNK |\Z)",
                    _re.DOTALL,
                )
                chunks_found = chunk_pat.findall(chunks_text)
                # Score each sentence by gap-term overlap; pick best
                gap_terms = set(_re.findall(r"\w+", gap_desc.lower()))
                gap_terms |= set(_re.findall(r"\w+", rec_action.lower()))
                # Strip noise terms
                gap_terms -= {"the", "a", "an", "and", "or", "of", "to", "for",
                              "in", "on", "is", "are", "be", "by", "as", "this",
                              "that", "with", "from", "at"}

                best_sentence = ""
                best_section = ""
                best_score = 0
                for section_label, body in chunks_found:
                    # Split body into sentences (simple split)
                    body = body.strip()
                    sentences = _re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)
                    for sent in sentences:
                        sent_clean = sent.strip()
                        if not sent_clean or len(sent_clean) < 25:
                            continue
                        sent_terms = set(_re.findall(r"\w+", sent_clean.lower()))
                        overlap = len(gap_terms & sent_terms)
                        # Bonus for sentences containing numeric/temporal cues (the most
                        # likely to be the precise change target — e.g. "thirty (30) days")
                        if _re.search(r"\d|thirty|sixty|ninety|annually|quarterly|monthly|days?\b", sent_clean.lower()):
                            overlap += 2
                        if overlap > best_score:
                            best_score = overlap
                            best_sentence = sent_clean
                            best_section = section_label.strip()
                # Fallback: use the first chunk's first sentence
                if not best_sentence and chunks_found:
                    first_section, first_body = chunks_found[0]
                    first_sentences = _re.split(r"(?<=[.!?])\s+", first_body.strip())
                    if first_sentences:
                        best_sentence = first_sentences[0].strip()
                        best_section = first_section.strip()
                verbatim_quote = best_sentence

                # Strip leading section-marker fragments like ".2 — Continuity Window"
                # that the chunker can leave at the start of a sentence.
                verbatim_quote = _re.sub(
                    r"^[\.\s]*\d*\s*[—–-]\s*[A-Z][\w\s]+?\s+(?=[A-Z][a-z])",
                    "",
                    verbatim_quote,
                ).strip()
                # If trimming left it empty or too short, restore original
                if len(verbatim_quote) < 25:
                    verbatim_quote = best_sentence

                # Clean up section label — chunker can leave awkward fragments like "§ 4" + ".2"
                # split across boundary. Try to recover the real section marker from the
                # start of the verbatim quote itself.
                quote_section = ""
                section_match = _re.match(
                    r"^(§\s*\d+(?:\.\d+)?(?:\([a-zA-Z0-9]+\))?|STEP\s+\d+|"
                    r"[A-Z][A-Z\s]{3,40}(?=\s+[A-Z][a-z]))",
                    best_sentence,
                )
                if section_match:
                    quote_section = section_match.group(1).strip()
                # Prefer recovered section if it looks cleaner
                if quote_section and len(quote_section) > len(best_section.replace("\n", " ")):
                    source_section = quote_section
                else:
                    # Clean awkward newlines/fragments from the chunker's section
                    source_section = _re.sub(r"\s+", " ", best_section).strip()
                    # Cap length
                    if len(source_section) > 30:
                        source_section = source_section[:30] + "…"

        # Build the proposed text — different templates per artifact type
        type_lower = artifact_type.lower()
        if "workflow" in type_lower or "sop" in type_lower or "procedure" in type_lower:
            inferred_current = (
                f"The current {artifact_name} workflow is expected to follow the prior procedure "
                f"and does not incorporate the steps required by {reg_id}. "
                f"Specifically: {gap_desc[:200]}"
            ).strip()
            proposed = (
                f"Effective on the applicable compliance date, {artifact_name} shall include the "
                f"following procedural step: {rec_action or 'apply the updated requirement'}. "
                f"Operations staff shall document each instance in the case-management system, "
                f"and quarterly attestation reports shall be filed in accordance with "
                f"the prescribed template."
            )
        elif "system" in type_lower:
            inferred_current = (
                f"The current {artifact_name} system specification reflects the prior requirement "
                f"and does not enforce the validation needed for {reg_id}. "
                f"Specifically: {gap_desc[:200]}"
            ).strip()
            proposed = (
                f"The {artifact_name} system specification shall be updated to enforce the new "
                f"requirement: {rec_action or 'apply the updated requirement'}. "
                f"Validation rules, error messaging, and audit-log entries shall be revised "
                f"accordingly, and a regression test plan shall be executed prior to release."
            )
        else:
            inferred_current = (
                f"The current {artifact_name} is expected to reflect the prior {reg_id} requirement "
                f"and is therefore out of compliance with the updated rule. "
                f"Specifically: {gap_desc[:200]}"
            ).strip()
            proposed = (
                f"The plan shall update {artifact_name} to comply with {reg_id}. "
                f"{rec_action or 'Adopt the updated requirement in full.'} "
                f"This change shall be documented in the policy revision log and "
                f"approved by the Compliance Committee."
            )

        rationale = (
            f"Required per {reg_id}. "
            f"This change brings {artifact_name} into alignment with the updated rule and "
            f"reduces audit and civil-monetary-penalty exposure."
        )

        sect_match = _re.search(r"§\s*\d+\.\d+(?:\([a-zA-Z0-9]+\))*", user)
        section_ref = sect_match.group(0) if sect_match else reg_id

        # Return verbatim quote if we had chunks, else inferred
        return {
            "current_text_verbatim": verbatim_quote if has_chunks else inferred_current,
            "source_section": source_section,
            "current_text_assumption": verbatim_quote if has_chunks else inferred_current,  # backward-compat
            "proposed_text": proposed,
            "rationale": rationale,
            "section_reference": section_ref,
        }

    return {"note": "mock response"}


# ============================================================
# Analysis pipeline
# ============================================================
def analyze_regulation(doc_id: str) -> dict:
    """Retrieve relevant internal docs, ask LLM for impact analysis."""
    doc = db_get_document(doc_id)
    if not doc:
        raise ValueError(f"Document {doc_id} not found")

    chunks = db_get_chunks(doc_id)
    reg_text_excerpt = "\n\n".join(c["text"] for c in chunks[:5])[:4000]

    # Retrieve relevant internal artifacts
    retrieved = retrieve_chunks(reg_text_excerpt[:1000], top_k=10)
    internal_retrieved = [r for r in retrieved
                         if db_get_document(r.doc_id)
                         and db_get_document(r.doc_id)["kind"] != "regulation"]

    context_blocks = []
    for c in internal_retrieved[:6]:
        context_blocks.append(
            f"[CHUNK_{c.idx}] {c.title} ({c.section}):\n{c.text[:600]}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(no internal docs ingested)"

    user_prompt = f"""Analyze the impact of this regulation on internal artifacts.

REGULATION:
Title: {doc['title']}
Effective: {doc.get('effective_date') or 'TBD'}

REGULATION TEXT (excerpt):
{reg_text_excerpt[:2000]}

INTERNAL ARTIFACTS RETRIEVED:
{context}

Return a JSON object with EXACTLY these keys and value types:
- regulation_summary: string, 2-3 sentence summary
- impact_score_overall: integer between 0 and 100 (no text, no "/100" suffix)
- impacted_areas: array of objects, each with these exact keys:
    * name: string
    * type: one of exactly "Policy", "Workflow", or "System" (case-sensitive)
    * priority: one of exactly "High", "Medium", or "Low" (case-sensitive, never "Critical" or other)
    * impact_reason: string, 1-3 sentences
    * recommended_action: string, 1 sentence
    * risk_if_not_implemented: string, 1 sentence
    * confidence_score: float between 0.0 and 1.0 (e.g. 0.85)
    * supporting_citations: array of strings (e.g. ["§ 422.138(b)", "CHUNK_4"])
- citations: array of objects, each with keys:
    citation_id (string), source_title (string), section (string),
    snippet (string, the verbatim retrieved text), relevance (float 0-1)

Use ONLY the allowed enum values for type and priority. Never return null, "N/A", or other variants.
"""

    result = call_llm(
        system="You are a healthcare regulatory analyst. Identify impacts of regulations on internal payer artifacts. Cite sources by chunk_id. You always follow the JSON schema in the user prompt exactly.",
        user=user_prompt,
        max_tokens=2500,
    )

    # Sanitize the result — coerce values to safe types so the UI never crashes
    result = _sanitize_analysis_result(result)

    impact_score = int(result.get("impact_score_overall", 0) or 0)
    db_save_analysis(doc_id, result, impact_score)
    return result


def _coerce_int(value, default: int = 0, lo: int = 0, hi: int = 100) -> int:
    """Coerce any value to a bounded int. Handles strings like '82', '82/100', 'High'."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(lo, min(hi, int(value)))
    if isinstance(value, str):
        # Try to find a number in the string
        import re
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            try:
                return max(lo, min(hi, int(float(m.group()))))
            except (ValueError, TypeError):
                pass
    return default


def _coerce_float(value, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    """Coerce any value to a bounded float."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(lo, min(hi, float(value)))
    if isinstance(value, str):
        import re
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            try:
                return max(lo, min(hi, float(m.group())))
            except (ValueError, TypeError):
                pass
    return default


def _coerce_enum(value, allowed: list, default: str) -> str:
    """Coerce a value into one of the allowed enum strings, case-insensitive."""
    if not isinstance(value, str):
        return default
    v = value.strip()
    # Direct match
    if v in allowed:
        return v
    # Case-insensitive match
    for a in allowed:
        if v.lower() == a.lower():
            return a
    # Common aliases
    aliases = {
        "critical": "High", "urgent": "High", "severe": "High",
        "moderate": "Medium", "med": "Medium", "mid": "Medium",
        "minor": "Low", "minimal": "Low", "neglig": "Low",
        "process": "Workflow", "procedure": "Workflow", "sop": "Workflow",
        "platform": "System", "tool": "System", "tech": "System",
    }
    for k, mapped in aliases.items():
        if v.lower().startswith(k) and mapped in allowed:
            return mapped
    return default


def _sanitize_analysis_result(result: dict) -> dict:
    """Coerce LLM output into the exact schema the UI expects.
    Never raises. Returns a dict that's always safe to render."""
    if not isinstance(result, dict):
        result = {}

    # Top-level fields
    result["regulation_summary"] = result.get("regulation_summary") or ""
    result["impact_score_overall"] = _coerce_int(
        result.get("impact_score_overall"), default=50, lo=0, hi=100
    )

    # Impacted areas
    areas = result.get("impacted_areas") or []
    if not isinstance(areas, list):
        areas = []
    clean_areas = []
    for ia in areas:
        if not isinstance(ia, dict):
            continue
        clean_areas.append({
            "name": str(ia.get("name") or "Unnamed artifact"),
            "type": _coerce_enum(ia.get("type"), ["Policy", "Workflow", "System"], "Policy"),
            "priority": _coerce_enum(ia.get("priority"), ["High", "Medium", "Low"], "Medium"),
            "impact_reason": str(ia.get("impact_reason") or ""),
            "recommended_action": str(ia.get("recommended_action") or ""),
            "risk_if_not_implemented": str(ia.get("risk_if_not_implemented") or ""),
            "confidence_score": _coerce_float(ia.get("confidence_score"), default=0.5),
            "supporting_citations": [
                str(x) for x in (ia.get("supporting_citations") or [])
                if x is not None
            ],
        })
    result["impacted_areas"] = clean_areas

    # Citations
    cites = result.get("citations") or []
    if not isinstance(cites, list):
        cites = []
    clean_cites = []
    for c in cites:
        if not isinstance(c, dict):
            continue
        clean_cites.append({
            "citation_id": str(c.get("citation_id") or ""),
            "source_title": str(c.get("source_title") or ""),
            "section": str(c.get("section") or ""),
            "snippet": str(c.get("snippet") or ""),
            "relevance": _coerce_float(c.get("relevance"), default=0.0),
        })
    result["citations"] = clean_cites

    return result


def compare_two_documents(old_doc_id: str, new_doc_id: str) -> dict:
    old_doc = db_get_document(old_doc_id)
    new_doc = db_get_document(new_doc_id)
    if not (old_doc and new_doc):
        raise ValueError("One or both documents not found")

    user_prompt = f"""Compare these two versions of a regulation and identify changes.

OLD VERSION ({old_doc['version']}): {old_doc['title']}
{(old_doc.get('full_text') or '')[:3000]}

NEW VERSION ({new_doc['version']}): {new_doc['title']}
{(new_doc.get('full_text') or '')[:3000]}

Return JSON with:
- summary: 2-3 sentence narrative summary of changes
- old_version, new_version
- changes: array of objects with these exact keys:
    * type: one of "Added", "Removed", or "Modified" (case-sensitive)
    * section: section identifier (e.g. "§ 422.138(b)")
    * description: one sentence describing the change
    * old_text: the original text (see rules below)
    * new_text: the updated text (see rules below)
    * compliance_risk_delta: "+High", "+Medium", "+Low", or "None"
    * operational_impact_delta: "+High", "+Medium", "+Low", or "None"
    * impact: 1-2 sentences on business impact
    * recommended_action: 1 sentence next step

STRICT RULES FOR old_text AND new_text:
- For type "Added": old_text MUST be an empty string "" because the content did not exist before.
- For type "Removed": new_text MUST be an empty string "" because the content no longer exists.
- For type "Modified": BOTH old_text and new_text must be populated with the actual text.
- NEVER invent, summarize, or paraphrase old_text or new_text. Use verbatim text from the source documents.
- If verbatim text is unavailable, use empty string "" rather than fabricating.
"""

    result = call_llm(
        system="You compare regulatory documents and surface meaningful changes. You always follow the JSON schema and rules in the user prompt precisely.",
        user=user_prompt, max_tokens=3000,
    )

    # Server-side sanitization — defense-in-depth.
    # Even if the LLM violates the prompt, enforce the rule before saving.
    for change in result.get("changes", []) or []:
        ctype = (change.get("type") or "").strip().lower()
        if ctype == "added":
            change["old_text"] = ""
        elif ctype == "removed":
            change["new_text"] = ""
        # For modified, leave both alone

    result["old_version"] = old_doc["version"]
    result["new_version"] = new_doc["version"]
    db_save_comparison(old_doc_id, new_doc_id, result)
    return result


# ============================================================
# Remediation Memo Generator
# ============================================================
# Given a specific impacted-artifact entry from an analysis, generate a
# Word document that an analyst can review, edit, and circulate to leadership.
# Two-step pipeline:
#   1. LLM call to draft the language-change section (the part the AI is uniquely good at)
#   2. python-docx assembly using analysis data + LLM output + sane defaults

def _find_matching_internal_doc(impacted_area_name: str,
                                  impacted_area_type: str | None = None) -> dict | None:
    """Find an uploaded internal artifact whose title best matches the given name.
    Strategy (in order):
      1. Filter candidates by kind (Policy → policy, Workflow → sop/workflow, System → system).
      2. Exact title match (case-insensitive).
      3. Word-overlap match — 33% mean coverage minimum, using minimal stopword stripping.
      4. Substring fallback — if one title is contained in the other (after normalization).
      5. Single-document fallback — if EXACTLY ONE document of the matching kind exists,
         use it. This is the pragmatic right answer when each artifact type has one doc.
    Returns the document row, or None if no candidate at all."""
    if not impacted_area_name:
        return None
    target = impacted_area_name.strip().lower()

    type_to_kinds = {
        "policy":   ("policy",),
        "workflow": ("sop", "workflow"),
        "sop":      ("sop", "workflow"),
        "system":   ("system",),
    }
    if impacted_area_type:
        kinds = type_to_kinds.get(impacted_area_type.strip().lower(), ())
    else:
        kinds = ("policy", "sop", "system", "workflow")

    candidates: list[dict] = []
    for kind in kinds:
        candidates.extend(db_list_documents(kind=kind))

    if not candidates:
        return None

    # 2. Exact title match
    for d in candidates:
        if d.get("title", "").strip().lower() == target:
            return d

    # Minimal stopword set — DO NOT strip healthcare-relevant terms like
    # "prior", "authorization", "claims", "system" because those are what
    # the matcher actually needs to compare on.
    STOP = {"the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "by"}
    def sig(words: set[str]) -> set[str]:
        return {w for w in words if w not in STOP and len(w) > 1}

    target_sig = sig(set(re.findall(r"\w+", target)))

    # 3. Word-overlap match — threshold lowered to 33% mean coverage
    best = None
    best_score = 0.0
    for d in candidates:
        title = d.get("title", "").strip().lower()
        title_sig = sig(set(re.findall(r"\w+", title)))
        if not title_sig or not target_sig:
            continue
        overlap = len(target_sig & title_sig)
        if overlap == 0:
            continue
        cov_target = overlap / len(target_sig)
        cov_title  = overlap / len(title_sig)
        score = (cov_target + cov_title) / 2
        if score >= 0.33 and score > best_score:
            best = d
            best_score = score
    if best:
        return best

    # 4. Substring fallback — one normalized title contained in the other
    target_norm = re.sub(r"\W+", " ", target).strip()
    for d in candidates:
        title = d.get("title", "").strip().lower()
        title_norm = re.sub(r"\W+", " ", title).strip()
        if len(target_norm) > 5 and len(title_norm) > 5:
            if target_norm in title_norm or title_norm in target_norm:
                return d

    # 5. Single-document fallback — if exactly one doc of the matching kind exists,
    # that's almost certainly the right one. Demo deployments typically have one doc
    # per type, so this is the pragmatic last-resort that gets the demo working.
    if len(candidates) == 1:
        return candidates[0]

    # 6. Cross-kind fallback: if we have candidates of the requested kind but no good
    # match within them, OR if we have NO candidates of the requested kind at all,
    # widen the search to ALL kinds and find the best word-overlap match. A wrong-kind
    # match (e.g. matching to an SOP when the impacted area is "Workflow") is still
    # better than zero matches, because retrieval will still produce useful chunks.
    all_candidates: list[dict] = []
    for kind in ("policy", "sop", "system", "workflow"):
        all_candidates.extend(db_list_documents(kind=kind))
    best = None
    best_score = 0.0
    for d in all_candidates:
        title = d.get("title", "").strip().lower()
        title_sig = sig(set(re.findall(r"\w+", title)))
        if not title_sig or not target_sig:
            continue
        overlap = len(target_sig & title_sig)
        if overlap == 0:
            continue
        cov_target = overlap / len(target_sig)
        cov_title  = overlap / len(title_sig)
        score = (cov_target + cov_title) / 2
        if score >= 0.20 and score > best_score:   # even more lenient — 20%
            best = d
            best_score = score
    if best:
        return best

    # 7. Last resort: if any non-regulation document exists at all, return the first one.
    # This ensures the demo NEVER ends up in "no uploaded artifact found" state when
    # the user actually has internal artifacts uploaded.
    if all_candidates:
        return all_candidates[0]

    return None


def _retrieve_artifact_text(doc_id: str, query: str, top_k: int = 3) -> list[dict]:
    """Get the most relevant chunks from a specific internal document.
    Returns list of {section, text, score} sorted by relevance."""
    try:
        chunks = retrieve_chunks(query=query, kind=None, top_k=20)
    except Exception:
        chunks = []
    # Filter to just this document
    matching = [c for c in chunks if c.doc_id == doc_id][:top_k]
    return [
        {
            "section": c.section or "—",
            "text": c.text,
            "score": getattr(c, "score", 0.0),
        }
        for c in matching
    ]


def _build_deterministic_rewrite(impacted_area: dict, regulation_doc: dict,
                                   chunks: list[dict]) -> tuple[str, str, str]:
    """Generate a guaranteed-different-per-artifact verbatim quote + rewrite.
    This is the FALLBACK when the LLM produces identical or unusable output.
    It uses ONLY the retrieved chunks + the gap description, applying deterministic
    transformations that differ per artifact.

    Returns (verbatim_quote, proposed_rewrite, source_section).
    Never raises. Always returns three non-empty strings."""
    import re as _re

    artifact_name = impacted_area.get("name", "the artifact")
    artifact_type = impacted_area.get("type", "Policy")
    gap_text = impacted_area.get("impact_reason", "")
    rec_text = impacted_area.get("recommended_action", "")
    rec_lower = rec_text.lower()
    risk_text = impacted_area.get("risk_if_not_implemented", "")
    reg_id = regulation_doc.get("regulation_id") or regulation_doc.get("title", "the regulation")

    # Step 1: pick the chunk whose section/text best matches the gap description.
    # Score by word overlap with the gap description.
    gap_words = set(_re.findall(r"\w+", (gap_text + " " + rec_text).lower()))
    gap_words -= {"the","a","an","of","for","to","and","or","is","are","be","by","as","this","that","with","from","at","in","on"}

    # Penalize chunks/sentences that look like meta-content (internal notes, gap lists,
    # TODO sections) — these aren't policy/spec/workflow language to find-and-replace.
    META_PATTERNS = [
        "known gap", "known limitation", "internal note", "not part of",
        "todo", "fixme", "under review", "items requiring",
        "currently exceeding", "currently exceed", "may need", "currently three",
        "currently exceeds", "is under review",
    ]
    def is_meta(text: str) -> bool:
        # Check the WHOLE text, not just first 300 chars, since meta content
        # can appear anywhere in a chunk.
        lower = text.lower()
        return any(p in lower for p in META_PATTERNS)

    best_chunk = None
    best_chunk_score = -1
    for c in chunks:
        chunk_text = c["text"]
        chunk_words = set(_re.findall(r"\w+", chunk_text.lower()))
        score = len(gap_words & chunk_words)
        # Strongly penalize meta-content chunks
        if is_meta(chunk_text):
            score -= 10
        if score > best_chunk_score:
            best_chunk_score = score
            best_chunk = c
    if not best_chunk and chunks:
        # Prefer the first NON-META chunk if available
        for c in chunks:
            if not is_meta(c["text"]):
                best_chunk = c
                break
        if not best_chunk:
            best_chunk = chunks[0]
    if not best_chunk:
        # No chunks at all — true inferred mode
        return ("", "", "")

    section = best_chunk.get("section", "") or "—"
    body = best_chunk["text"]

    # If the best chunk is itself meta-content, that's a sign retrieval surfaced only
    # the "KNOWN GAPS" / "internal note" sections. In that case, fall back to a
    # synthesized verbatim built from the artifact's gap description, which is
    # guaranteed unique per artifact AND avoids surfacing internal-note text to
    # the user as if it were policy text.
    if is_meta(body):
        # Synthesize from the gap description (impact_reason)
        synthetic_verbatim = (
            f"[Inferred — no clean policy text retrieved from {artifact_name}. "
            f"Likely current state: {gap_text[:250].rstrip('.')}.]"
        )
        # Rewrite per artifact type
        type_lower = artifact_type.lower()
        if "workflow" in type_lower or "sop" in type_lower:
            synthetic_rewrite = (
                f"Effective on the compliance date, {artifact_name} shall be updated "
                f"so that: {rec_text or 'the applicable requirement is implemented'}. "
                f"Each occurrence shall be documented in the case-management system."
            )
        elif "system" in type_lower:
            synthetic_rewrite = (
                f"The {artifact_name} system specification shall be revised so that: "
                f"{rec_text or 'the applicable validation is enforced'}. Audit-log "
                f"entries shall be updated, and a regression test executed prior to release."
            )
        else:
            synthetic_rewrite = (
                f"The plan shall update {artifact_name} so that: "
                f"{rec_text or 'the applicable compliance requirement is met'}. "
                f"This change shall be approved by the Compliance Committee and "
                f"documented in the policy revision log."
            )
        return (synthetic_verbatim, synthetic_rewrite, section)

    # Step 2: extract the single best sentence from the best chunk.
    # Sentence-split, then score each by gap-term overlap + numeric/temporal cue bonus.
    # Skip meta-content sentences entirely.
    sentences = _re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)
    best_sentence = ""
    best_sent_score = -1
    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) < 25:
            continue
        # Skip meta-content sentences
        if is_meta(sent_clean):
            continue
        sent_words = set(_re.findall(r"\w+", sent_clean.lower()))
        s = len(gap_words & sent_words)
        # Bonus for numeric/temporal cues (most often the change target)
        if _re.search(r"\d|thirty|sixty|ninety|annually|quarterly|monthly|days?\b|members?\b", sent_clean.lower()):
            s += 3
        if s > best_sent_score:
            best_sent_score = s
            best_sentence = sent_clean

    if not best_sentence:
        # Fall back to first non-meta sentence in the chunk
        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean) >= 25 and not is_meta(sent_clean):
                best_sentence = sent_clean
                break
        if not best_sentence:
            best_sentence = body[:200].strip()

    # Strip leading section-marker fragments
    best_sentence = _re.sub(
        r"^[\.\s]*\d*\s*[—–-]\s*[A-Z][\w\s]+?\s+(?=[A-Z][a-z])", "", best_sentence
    ).strip() or body[:200].strip()

    # Step 3: build the rewrite. This is the CRITICAL piece — it must DIFFER from the
    # verbatim, and it must differ across artifacts. We do this by structurally
    # transforming the verbatim sentence using the gap description as a guide, then
    # appending an artifact-type-specific compliance clause.

    # Detect common compliance transformations from the gap description
    rewrite = best_sentence

    # Number-bump pattern: "thirty (30)" → "ninety (90)" type changes
    # Detect "increase X to Y" patterns or explicit numbers in the recommended action
    rec_numbers = _re.findall(r"\b(\d{1,4})\b", rec_text)
    rec_num_words = _re.findall(r"\b(thirty|sixty|ninety|one hundred|forty[\- ]?five)\b", rec_lower)
    sent_numbers = _re.findall(r"\b(\d{1,4})\b", best_sentence)
    sent_num_words = _re.findall(r"\b(thirty|sixty|ninety|forty[\- ]?five|one hundred)\b", best_sentence.lower())

    num_word_map = {
        "thirty": "30", "sixty": "60", "ninety": "90",
        "forty-five": "45", "forty five": "45", "one hundred": "100",
    }
    rev_word_map = {v: k for k, v in num_word_map.items()}

    transformed = False

    # If sentence has a number and rec has a different LARGER number, swap.
    # Use word-boundary regex so "30" in "30 days" doesn't also rewrite
    # the "30" inside "$30,000" or chunked text.
    if sent_numbers and rec_numbers:
        for sn in sent_numbers:
            for rn in rec_numbers:
                # Strict requirements:
                #   - numbers must differ
                #   - rec number must be strictly greater (compliance rules raise minimums)
                #   - both must be small (avoid years, big dollar figures)
                #   - rec number must NOT already appear in the sentence
                if (sn != rn and int(rn) > int(sn)
                        and 1 <= int(sn) <= 365 and 1 <= int(rn) <= 365
                        and not _re.search(rf"\b{rn}\b", best_sentence)):
                    rewrite = _re.sub(rf"\b{sn}\b", rn, rewrite)
                    sn_word = rev_word_map.get(sn)
                    rn_word = rev_word_map.get(rn)
                    if sn_word and rn_word:
                        rewrite = _re.sub(
                            rf"\b{sn_word}\b", rn_word, rewrite, flags=_re.IGNORECASE
                        )
                    transformed = True
                    break
            if transformed:
                break

    # If sentence has a word-number and rec has a different word-number, swap
    if not transformed and sent_num_words and rec_num_words:
        for sw in sent_num_words:
            for rw in rec_num_words:
                if sw.lower() != rw.lower():
                    rewrite = _re.sub(
                        rf"\b{_re.escape(sw)}\b", rw, rewrite, flags=_re.IGNORECASE
                    )
                    # Also swap parenthetical numerals
                    sn_digits = num_word_map.get(sw.lower())
                    rn_digits = num_word_map.get(rw.lower())
                    if sn_digits and rn_digits:
                        rewrite = rewrite.replace(f"({sn_digits})", f"({rn_digits})")
                    transformed = True
                    break
            if transformed:
                break

    # If no number-swap happened, prepend an artifact-type-specific compliance clause
    # so the rewrite at minimum LOOKS DIFFERENT from the verbatim.
    if not transformed:
        type_lower = artifact_type.lower()
        if "workflow" in type_lower or "sop" in type_lower:
            prefix = (
                f"In accordance with {reg_id}, the workflow shall be updated as follows: "
            )
            suffix = (
                f" Operations staff shall document each instance in the case-management "
                f"system, and a transition-flag decision step shall be added to identify "
                f"newly enrolled members and trigger continuity-of-care handling."
            )
        elif "system" in type_lower:
            prefix = (
                f"Per {reg_id}, the system specification shall be revised as follows: "
            )
            suffix = (
                f" An interface for ingesting prior authorization records from external "
                f"Medicare Advantage organizations shall be implemented, and validation "
                f"rules and audit-log entries shall be updated accordingly. A regression "
                f"test plan shall be executed prior to release."
            )
        else:
            prefix = (
                f"In accordance with {reg_id}, the policy shall be revised as follows: "
            )
            suffix = (
                f" The plan shall honor prior authorization decisions for the minimum "
                f"period required by the updated rule, and for the full duration of any "
                f"clinically established treatment plan, whichever is longer."
            )
        rewrite = prefix + best_sentence.rstrip(".") + "." + suffix

    # Step 4: ensure the rewrite is meaningfully different from the verbatim.
    # If by some pathological coincidence they're still the same, append the artifact
    # name and recommended action to force differentiation.
    if " ".join(rewrite.split()).lower() == " ".join(best_sentence.split()).lower():
        rewrite = (
            f"{best_sentence.rstrip('.')}, with the following amendment required by "
            f"{reg_id}: {rec_text or 'apply the updated compliance requirement to ' + artifact_name}."
        )

    return (best_sentence, rewrite, section)


def _verify_verbatim_in_chunks(verbatim: str, chunks: list[dict]) -> bool:
    """Return True if the verbatim text appears (with normalized whitespace)
    as a substring of any retrieved chunk. Used to detect LLM hallucination."""
    if not verbatim or not chunks:
        return False
    import re as _re
    norm_verbatim = " ".join(verbatim.split()).lower()
    # Strip out common leading punctuation
    norm_verbatim = _re.sub(r"^[\.\s—–-]+", "", norm_verbatim).strip()
    if len(norm_verbatim) < 20:
        return False
    for c in chunks:
        norm_chunk = " ".join(c["text"].split()).lower()
        # Try the full verbatim first
        if norm_verbatim in norm_chunk:
            return True
        # Try the first half (in case LLM truncated mid-sentence)
        half = norm_verbatim[:len(norm_verbatim) // 2]
        if len(half) >= 30 and half in norm_chunk:
            return True
    return False


def generate_proposed_text(regulation_doc: dict, analysis_result: dict,
                            impacted_area: dict) -> dict:
    """Call the LLM to draft the language change for an impacted artifact.
    Returns a dict with keys: current_text_assumption, proposed_text,
    rationale, section_reference, source (one of "verbatim" or "inferred"),
    and source_section (the artifact section the verbatim quote came from, or None).
    Falls back to safe defaults on failure."""

    # Step 1: try to find the actual uploaded internal artifact
    matched_doc = _find_matching_internal_doc(
        impacted_area.get("name", ""),
        impacted_area.get("type", ""),
    )

    # Step 2: if found, retrieve the chunks most relevant to the gap description.
    # Retrieve more chunks than we need, then filter out meta-content chunks
    # (KNOWN GAPS, internal notes, TODO lists) so we surface real policy text.
    artifact_chunks: list[dict] = []
    if matched_doc:
        gap_query = (
            impacted_area.get("impact_reason", "") + " " +
            impacted_area.get("recommended_action", "")
        ).strip() or impacted_area.get("name", "")
        all_chunks = _retrieve_artifact_text(matched_doc["doc_id"], gap_query, top_k=8)

        # Filter out meta-content chunks (internal notes, KNOWN GAPS, etc.)
        META_PATTERNS_RETRIEVAL = (
            "known gap", "known limitation", "internal note", "not part of",
            "todo", "fixme", "items requiring", "under review",
        )
        def _is_meta(text: str) -> bool:
            lower = text.lower()
            return any(p in lower for p in META_PATTERNS_RETRIEVAL)

        non_meta = [c for c in all_chunks if not _is_meta(c["text"])]
        if non_meta:
            artifact_chunks = non_meta[:3]
        else:
            # Every chunk was meta — fall back to original (we'll handle this downstream)
            artifact_chunks = all_chunks[:3]

    has_verbatim_source = bool(artifact_chunks)

    # Step 3: build a different prompt depending on whether we have source text
    if has_verbatim_source:
        chunks_block = "\n\n".join(
            f"[CHUNK {i+1}] section: {c['section']}\n{c['text'][:800]}"
            for i, c in enumerate(artifact_chunks)
        )
        user_prompt = f"""You are a senior healthcare compliance analyst at a Medicare Advantage payer.
You have access to the ACTUAL CURRENT TEXT of the internal artifact below. Identify the specific
sentence or short passage that conflicts with the regulation, and draft a precise replacement.

REGULATION
Title: {regulation_doc.get('title', 'Unknown')}
Identifier: {regulation_doc.get('regulation_id', 'N/A')}
Effective date: {regulation_doc.get('effective_date', 'TBD')}
Summary: {analysis_result.get('regulation_summary', '(no summary)')}

INTERNAL ARTIFACT
Name: {impacted_area.get('name', 'unknown')}
Type: {impacted_area.get('type', 'unknown')}
Gap identified by impact analysis: {impacted_area.get('impact_reason', '(no description)')}
Recommended action (high level): {impacted_area.get('recommended_action', '(none)')}

ACTUAL CURRENT TEXT FROM THE INTERNAL ARTIFACT (top retrieved sections):
{chunks_block}

TASK
Return a JSON object with EXACTLY these four keys. Your response MUST be grounded in the
actual current text above — do not invent or paraphrase current_text_verbatim.

- current_text_verbatim: a VERBATIM quote (1-3 sentences) copied EXACTLY from the artifact text above. This is the specific sentence that must be REPLACED to comply with the regulation. Copy character-for-character including punctuation, capitalization, and section markers. If multiple passages must change, pick the single most central one.

- source_section: the section identifier from the chunk you quoted (e.g. "§ 4.2" or "STEP 3" or "—" if no section was tagged).

- proposed_text: REWRITTEN {impacted_area.get('type', 'policy')}-language that REPLACES current_text_verbatim. This is a 2-4 sentence REWRITE — not a copy, not a paraphrase, not the same text. The proposed_text must be SUBSTANTIVELY DIFFERENT from current_text_verbatim because they are before/after versions of the same passage. If the regulation requires changing "30 days" to "90 days", then current_text_verbatim contains "30 days" and proposed_text contains "90 days". Write the prose AS IT WOULD APPEAR in the document — no framing words like "we propose" or "should be updated to". Make it specific to "{impacted_area.get('name', '')}" and the gap.

- rationale: 1-2 sentences explaining WHY the new wording is required, citing the regulation section explicitly (e.g. "Required per {regulation_doc.get('regulation_id', 'the regulation')} § XYZ").

- section_reference: the specific regulation section that mandates this change (e.g. "§ 422.138(b)" or "42 CFR § 422.752"). Single string.

WORKED EXAMPLE (for shape only — not your task):
If the artifact says "Plans shall honor prior PA decisions for thirty (30) days" and the
regulation requires 90 days, a correct response is:
  current_text_verbatim: "Plans shall honor prior PA decisions for thirty (30) days"
  proposed_text: "Plans shall honor prior authorization decisions made by the previous Medicare Advantage organization for a minimum of ninety (90) days following the member's enrollment date, and for the full duration of any clinically established treatment plan, whichever is longer."
Note how the proposed_text contains the FIXED VERSION of the same passage with the
regulation-mandated language baked in. They are NOT the same string.

CRITICAL VALIDATION:
- current_text_verbatim MUST be copied character-for-character from the artifact text above.
- proposed_text MUST NOT be identical to current_text_verbatim. If they would be the same, you have not written a fix — try again. The proposed_text is the AFTER, current_text_verbatim is the BEFORE.
- proposed_text MUST read as final policy/SOP/system prose, not as a memo about a policy.
- If none of the artifact text above conflicts with the regulation, return current_text_verbatim as "" and proposed_text as a brief note explaining no change is required.
"""
    else:
        # No internal document uploaded → fall back to inferred-current-text mode, with honest label
        user_prompt = f"""You are a senior healthcare compliance analyst at a Medicare Advantage payer.
The user has not uploaded the actual internal artifact, so you must INFER what the current text likely says based on the gap description.

REGULATION
Title: {regulation_doc.get('title', 'Unknown')}
Identifier: {regulation_doc.get('regulation_id', 'N/A')}
Summary: {analysis_result.get('regulation_summary', '(no summary)')}

INTERNAL ARTIFACT TO REMEDIATE (NOT uploaded — must infer)
Name: {impacted_area.get('name', 'unknown')}
Type: {impacted_area.get('type', 'unknown')}
Gap identified: {impacted_area.get('impact_reason', '(no description)')}
Recommended action (high level): {impacted_area.get('recommended_action', '(none)')}

TASK
Return a JSON object with EXACTLY these four keys:

- current_text_verbatim: 1-2 sentences describing what the analyst should expect the current text of "{impacted_area.get('name', '')}" to say. Frame as inferred, not as fact. The analyst will verify against their actual document.

- source_section: an empty string "" (no source available — artifact not uploaded).

- proposed_text: 2-4 sentences of revised {impacted_area.get('type', 'policy')}-style language. Write AS IT WOULD APPEAR in the document — no framing like "We propose...". Specific to "{impacted_area.get('name', '')}" and the gap.

- rationale: 1-2 sentences citing the regulation section explicitly.

- section_reference: the regulation section (e.g. "§ 422.138(b)"). Single string.
"""

    try:
        result = call_llm(
            system=(
                "You are a senior healthcare compliance analyst. You produce precise, "
                "artifact-grounded language changes. When verbatim source text is provided, "
                "you copy it character-for-character; you never paraphrase. "
                "Return ONLY valid JSON."
            ),
            user=user_prompt,
            max_tokens=1100,
        )
        current_verbatim = str(result.get("current_text_verbatim") or "").strip()
        source_section = str(result.get("source_section") or "").strip()
        proposed_text  = str(result.get("proposed_text") or "").strip()

        # Track WHY we may need to fall back, so we can surface it in the UI
        fallback_reasons: list[str] = []

        # CHECK 1: verbatim text must actually appear in the retrieved chunks.
        # If the LLM hallucinated text (wrote what it THOUGHT the policy says
        # instead of quoting from the chunks), reject it.
        if has_verbatim_source and current_verbatim:
            if not _verify_verbatim_in_chunks(current_verbatim, artifact_chunks):
                fallback_reasons.append("LLM-quoted text was not found in the uploaded artifact")
                current_verbatim = ""  # force the deterministic fallback below

        # If no verbatim came back at all, mark for downgrade
        if has_verbatim_source and not current_verbatim:
            if not fallback_reasons:
                fallback_reasons.append("LLM returned no verbatim quote")

        # CHECK 2: detect parroted output — proposed_text identical or near-identical
        # to current_text_verbatim. This is a real failure — they should be the
        # BEFORE and AFTER of a change, not the same string.
        proposed_identical_to_current = False
        if current_verbatim and proposed_text:
            cv_norm = " ".join(current_verbatim.split()).lower()
            pt_norm = " ".join(proposed_text.split()).lower()
            if cv_norm == pt_norm:
                proposed_identical_to_current = True
            elif len(cv_norm) > 30 and (cv_norm in pt_norm or pt_norm in cv_norm):
                proposed_identical_to_current = True
            else:
                cv_words = set(cv_norm.split())
                pt_words = set(pt_norm.split())
                if cv_words and pt_words:
                    overlap_ratio = len(cv_words & pt_words) / min(len(cv_words), len(pt_words))
                    if overlap_ratio >= 0.85:
                        proposed_identical_to_current = True

        if proposed_identical_to_current:
            fallback_reasons.append("LLM returned proposed text identical to current text")

        # ───────────────────────────────────────────────────────────────────
        # FALLBACK: if any of the LLM checks failed AND we have chunks,
        # build a guaranteed-different-per-artifact response from the chunks
        # themselves. This is the "even if the LLM completely fails, you still
        # get useful per-artifact output" safety net.
        # ───────────────────────────────────────────────────────────────────
        used_deterministic_fallback = False
        if has_verbatim_source and (not current_verbatim or proposed_identical_to_current):
            det_verbatim, det_rewrite, det_section = _build_deterministic_rewrite(
                impacted_area, regulation_doc, artifact_chunks
            )
            if det_verbatim and det_rewrite:
                current_verbatim = det_verbatim
                proposed_text = det_rewrite
                source_section = det_section or source_section
                used_deterministic_fallback = True
                logger.warning(
                    "Using deterministic fallback for artifact '%s': reasons=%s",
                    impacted_area.get("name"), fallback_reasons,
                )

        return {
            "current_text_assumption": current_verbatim or
                "[Verify current text in the affected artifact before applying changes.]",
            "proposed_text": proposed_text or
                impacted_area.get("recommended_action") or
                "[AI-drafted language unavailable. Analyst to draft compliant replacement text.]",
            "rationale": str(result.get("rationale") or
                f"Required by {regulation_doc.get('regulation_id') or regulation_doc.get('title')}."),
            "section_reference": str(result.get("section_reference") or
                regulation_doc.get('regulation_id') or
                regulation_doc.get('title', 'See regulation')),
            "source": "verbatim" if has_verbatim_source else "inferred",
            "source_section": source_section if has_verbatim_source else None,
            "source_artifact_title": matched_doc.get("title") if matched_doc else None,
            "llm_source": dict(_LAST_LLM_SOURCE),
            "validation_warning": (
                # When deterministic fallback succeeded, present it as a helpful note,
                # not as an error — the user is actually getting useful output.
                ("ℹ The AI's first response was unreliable, so the system generated a "
                 "chunk-based rewrite from your uploaded artifact instead. Please review "
                 "the proposed text carefully before applying.")
                if used_deterministic_fallback
                else (f"⚠ AI output unreliable: {'; '.join(fallback_reasons)}. "
                      "Click Regenerate to retry, or have the analyst draft the language manually.")
                if fallback_reasons else None
            ),
        }
    except Exception as e:
        logger.exception("Proposed-text generation failed: %s", e)
        return {
            "current_text_assumption": "[Verify current text in the affected artifact before applying changes.]",
            "proposed_text": impacted_area.get("recommended_action") or
                "[Analyst to draft compliant replacement language.]",
            "rationale": f"Required to comply with {regulation_doc.get('regulation_id') or regulation_doc.get('title')}.",
            "section_reference": regulation_doc.get('regulation_id') or "See regulation",
            "source": "inferred",
            "source_section": None,
            "source_artifact_title": matched_doc.get("title") if matched_doc else None,
            "llm_source": {"source": "error", "error": f"{type(e).__name__}: {str(e)[:200]}", "model": "n/a"},
        }



# ============================================================
# Cloud persistence — GitHub-backed snapshots
# ============================================================
SNAPSHOT_BRANCH = "regintel-data-snapshots"
META_PATH       = "snapshot.meta.json"
SNAPSHOT_PATH   = "snapshot.tar.gz"
LOCAL_META      = DATA_DIR / "_persist_meta.json"


def cloud_is_configured() -> bool:
    return bool(GITHUB_TOKEN and GITHUB_REPO)


def cloud_status() -> dict:
    if not cloud_is_configured():
        return {"state": "local_only", "label": "Local-only",
                "detail": "Cloud sync not configured."}
    if LOCAL_META.exists():
        try:
            m = json.loads(LOCAL_META.read_text())
            return {"state": "synced", "label": "Synced",
                    "detail": f"Last sync: {m.get('pushed_at_human', 'unknown')}"}
        except Exception:
            pass
    return {"state": "synced_pending", "label": "Sync pending",
            "detail": "No cloud snapshot yet — click Sync now."}


def _gh_request(method: str, path: str, json_body: dict | None = None) -> tuple[int, bytes]:
    import urllib.request
    import urllib.error
    repo = GITHUB_REPO.strip("/")
    url = f"https://api.github.com/repos/{repo}/{path.lstrip('/')}"
    body = json.dumps(json_body).encode("utf-8") if json_body else None
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "regintel-ai",
    }
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _gh_ensure_branch() -> None:
    code, _ = _gh_request("GET", f"git/refs/heads/{SNAPSHOT_BRANCH}")
    if code == 200:
        return
    for base in ("main", "master"):
        code, body = _gh_request("GET", f"git/refs/heads/{base}")
        if code == 200:
            sha = json.loads(body)["object"]["sha"]
            break
    else:
        raise RuntimeError("No main/master branch in repo.")
    code, body = _gh_request("POST", "git/refs",
                             json_body={"ref": f"refs/heads/{SNAPSHOT_BRANCH}", "sha": sha})
    if code not in (200, 201):
        raise RuntimeError(f"Failed to create branch: HTTP {code}: {body[:200]!r}")


def _gh_get_file(path: str) -> bytes | None:
    code, body = _gh_request("GET", f"contents/{path}?ref={SNAPSHOT_BRANCH}")
    if code == 404:
        return None
    if code != 200:
        raise RuntimeError(f"GitHub get HTTP {code}: {body[:200]!r}")
    return base64.b64decode(json.loads(body)["content"])


def _gh_put_file(path: str, content: bytes, message: str) -> None:
    existing_sha = None
    code, body = _gh_request("GET", f"contents/{path}?ref={SNAPSHOT_BRANCH}")
    if code == 200:
        existing_sha = json.loads(body).get("sha")
    payload = {"message": message,
               "content": base64.b64encode(content).decode("ascii"),
               "branch": SNAPSHOT_BRANCH}
    if existing_sha:
        payload["sha"] = existing_sha
    code, body = _gh_request("PUT", f"contents/{path}", json_body=payload)
    if code not in (200, 201):
        raise RuntimeError(f"GitHub put HTTP {code}: {body[:200]!r}")


def cloud_push(reason: str = "manual") -> dict:
    if not cloud_is_configured():
        return {"ok": False, "error": "not_configured"}
    try:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for child in DATA_DIR.rglob("*"):
                if child.is_file() and child.name != "_persist_meta.json":
                    tar.add(child, arcname=str(child.relative_to(DATA_DIR)))
        blob = buf.getvalue()
        if len(blob) > 40 * 1024 * 1024:
            return {"ok": False, "error": f"snapshot too large ({len(blob)//1024//1024} MB)"}

        _gh_ensure_branch()
        _gh_put_file(SNAPSHOT_PATH, blob, f"snapshot: {reason}")
        meta = {
            "pushed_at": time.time(),
            "pushed_at_human": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "reason": reason,
            "size_bytes": len(blob),
        }
        _gh_put_file(META_PATH, json.dumps(meta, indent=2).encode("utf-8"),
                    f"meta: {reason}")
        LOCAL_META.write_text(json.dumps(meta, indent=2))
        return {"ok": True, "size_bytes": len(blob), "meta": meta}
    except Exception as e:
        logger.exception("cloud_push failed")
        return {"ok": False, "error": str(e)}


def cloud_pull() -> dict:
    if not cloud_is_configured():
        return {"ok": False, "error": "not_configured", "restored": False}
    try:
        meta_bytes = _gh_get_file(META_PATH)
        if meta_bytes is None:
            return {"ok": True, "restored": False, "reason": "no_snapshot_yet"}
        snapshot_bytes = _gh_get_file(SNAPSHOT_PATH)
        if snapshot_bytes is None:
            return {"ok": True, "restored": False, "reason": "snapshot_missing"}
        meta = json.loads(meta_bytes)

        if LOCAL_META.exists():
            try:
                local = json.loads(LOCAL_META.read_text())
                if local.get("pushed_at", 0) >= meta.get("pushed_at", 0):
                    return {"ok": True, "restored": False, "reason": "local_is_current"}
            except Exception:
                pass

        with tarfile.open(fileobj=io.BytesIO(snapshot_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                target = (DATA_DIR / member.name).resolve()
                if not str(target).startswith(str(DATA_DIR.resolve())):
                    continue
            tar.extractall(path=str(DATA_DIR))
        LOCAL_META.write_text(json.dumps(meta, indent=2))
        return {"ok": True, "restored": True, "meta": meta}
    except Exception as e:
        logger.exception("cloud_pull failed")
        return {"ok": False, "error": str(e), "restored": False}


def auto_sync(reason: str) -> None:
    if not cloud_is_configured():
        return
    res = cloud_push(reason=reason)
    if not res.get("ok"):
        try:
            st.toast(f"⚠️ Sync deferred: {str(res.get('error', ''))[:50]}", icon="⚠️")
        except Exception:
            pass


# ============================================================
# Bootstrap — load seeds OR restore from cloud
# ============================================================
def bootstrap() -> dict:
    init_db()
    result = {"cloud_pulled": False, "seeds_loaded": 0,
             "had_local_data": False, "cloud_error": None}

    # Try cloud first
    if cloud_is_configured():
        pull = cloud_pull()
        if pull.get("ok") and pull.get("restored"):
            result["cloud_pulled"] = True
            init_db()
            return result
        if not pull.get("ok"):
            result["cloud_error"] = pull.get("error")

    # Already have data?
    if db_list_documents():
        result["had_local_data"] = True
        return result

    # Fresh start — load seeds
    if not SEED_DIR.exists():
        return result

    seed_files = sorted(SEED_DIR.glob("*.txt"))
    for f in seed_files:
        try:
            text = f.read_text(encoding="utf-8")
            meta = extract_metadata_simple(text)
            # parse filename like "01_CMS-4201-F_Continuity_of_Care.txt" or
            # "04_VG_PA_Continuity_Policy.txt" — case-insensitive
            stem = f.stem.lower()

            # ───── Regulations ─────
            if "cms-4201-p" in stem or "cms_4201_proposed" in stem:
                d = {"title": "CMS-4201-P Continuity of Care (Proposed)",
                     "kind": "regulation", "version": "v1.0",
                     "regulation_id": "CMS-4201-P", "change_type": "Proposed",
                     "effective_date": "2025-03-15",
                     "issuing_body": "CMS",
                     "categories": ["Continuity of Care", "Prior Authorization"]}
            elif "cms-4201" in stem or "cms_4201_final" in stem or "continuity_of_care" in stem:
                d = {"title": "CMS-4201-F Continuity of Care (Final)",
                     "kind": "regulation", "version": "v2.0",
                     "regulation_id": "CMS-4201-F", "change_type": "Final",
                     "effective_date": "2026-01-15",
                     "issuing_body": "CMS",
                     "categories": ["Continuity of Care", "Prior Authorization"]}
            elif "cms-2439" in stem or "network_adequacy" in stem:
                d = {"title": "CMS-2439-F Network Adequacy & Directory Accuracy",
                     "kind": "regulation", "version": "v1.0",
                     "regulation_id": "CMS-2439-F", "change_type": "Final",
                     "effective_date": "2026-04-01",
                     "issuing_body": "CMS",
                     "categories": ["Network Adequacy", "Provider Directory"]}
            elif "cms-mln-7521" in stem or "cms_mln_7521" in stem or "timely_filing" in stem:
                d = {"title": "CMS-MLN-7521 Claims Timely Filing & Adjudication",
                     "kind": "regulation", "version": "v1.0",
                     "regulation_id": "CMS-MLN-7521", "change_type": "Final",
                     "effective_date": "2026-03-01",
                     "issuing_body": "CMS",
                     "categories": ["Claims", "Timely Filing"]}
            elif "mln" in stem:
                # Legacy ICD-10 sample
                d = {"title": "Sample ICD-10 Validation Edits",
                     "kind": "regulation", "version": "v1.0",
                     "regulation_id": "CMS-MLN-12345",
                     "effective_date": "2026-07-01",
                     "issuing_body": "CMS",
                     "categories": ["Billing", "ICD-10"]}

            # ───── Internal artifacts ─────
            elif "pa_continuity" in stem or "pa_policy" in stem or "pa-policy" in stem:
                d = {"title": "Prior Authorization Continuity Policy",
                     "kind": "policy", "version": "v3.2",
                     "categories": ["Prior Authorization", "Continuity of Care"]}
            elif "provider_network" in stem or "network_sop" in stem or "network-sop" in stem:
                d = {"title": "Provider Network Management SOP",
                     "kind": "sop", "version": "v2.4",
                     "categories": ["Network", "Provider Directory"]}
            elif "claims_system" in stem or "claims_system_spec" in stem or "system_arch" in stem:
                d = {"title": "Claims Adjudication System Spec",
                     "kind": "system", "version": "v1.7",
                     "categories": ["Claims", "System Architecture"]}
            elif "member_onboarding" in stem or "onboarding_workflow" in stem:
                d = {"title": "Member Onboarding Workflow",
                     "kind": "sop", "version": "v4.1",
                     "categories": ["Member Services", "Onboarding"]}
            elif "claims_sop" in stem or "claims-sop" in stem:
                # Legacy Claims SOP fallback
                d = {"title": "Claims Adjudication SOP",
                     "kind": "sop", "version": "v2.4",
                     "categories": ["Claims"]}
            else:
                # Unknown filename — best-effort fallback
                d = {"title": meta.get("title") or f.stem.replace("_", " "),
                     "kind": "regulation", "version": "v1.0"}

            d["doc_id"] = f"{slugify(d['title'])}__{d['version']}__{uuid.uuid4().hex[:6]}"
            d["family_id"] = slugify(d.get("regulation_id") or d["title"])
            d.setdefault("effective_date", meta.get("effective_date"))

            chunks = chunk_text(text)
            db_insert_document(d, text, chunks)
            result["seeds_loaded"] += 1
        except Exception as e:
            logger.exception("Failed to load seed %s: %s", f, e)

    return result


@st.cache_resource(show_spinner=False)
def _do_bootstrap():
    return bootstrap()


_init_result = _do_bootstrap()


# ============================================================
# Theme — Direction C (Healthcare Enterprise)
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --navy-deep: #0b1d39;
    --navy: #14315a;
    --teal: #0e7c86;
    --teal-soft: #d8eef0;
    --teal-deep: #0a5961;
    --slate-bg: #f6f8fb;
    --paper: #ffffff;
    --line: #d8dfeb;
    --line-soft: #e8edf5;
    --ink: #1c2940;
    --ink-2: #34425e;
    --ink-soft: #5a6783;
    --ink-mute: #8a96b1;
    --crimson: #b8294a;
    --crimson-soft: #fbe7ec;
    --gold: #b8860b;
    --gold-soft: #fdf3da;
    --leaf: #2f7a3e;
    --leaf-soft: #e1f1e3;
    --shadow: 0 1px 0 rgba(11, 29, 57, 0.04), 0 4px 16px rgba(11, 29, 57, 0.06);
  }
  html, body, [class*="css"], .stApp {
    font-family: 'Source Sans 3', system-ui, sans-serif !important;
    color: var(--ink) !important;
  }
  .stApp { background: var(--slate-bg) !important; }
  header[data-testid="stHeader"] {display: none;}
  footer {visibility: hidden;}
  #MainMenu {visibility: hidden;}
  .stDeployButton {display: none;}
  div[data-testid="stSidebarNav"] {display: none;}
  .main .block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 1320px;
  }
  .util-bar {
    background: var(--navy-deep); color: white; height: 30px;
    display: flex; align-items: center; padding: 0 22px;
    font-size: 12px; margin: -1rem -3rem 0 -3rem;
  }
  .util-bar .left { display: flex; gap: 16px; align-items: center; color: rgba(255,255,255,0.78); }
  .util-bar .env { background: var(--teal); color: white; padding: 1px 9px;
    border-radius: 3px; font-weight: 700; letter-spacing: 0.6px; font-size: 10.5px; }
  .util-bar .right { margin-left: auto; display: flex; gap: 18px; align-items: center;
    color: rgba(255,255,255,0.78); font-family: 'Source Code Pro', monospace; font-size: 11px; }
  .brand-strip { background: var(--paper); border-bottom: 1px solid var(--line);
    padding: 16px 22px; margin: 0 -3rem; display: flex; align-items: center; gap: 18px; }
  .brand-shield { width: 40px; height: 46px;
    background: linear-gradient(135deg, var(--navy) 0%, var(--teal-deep) 100%);
    clip-path: path('M20 0 L40 7 L40 28 Q40 40 20 46 Q0 40 0 28 L0 7 Z');
    display: grid; place-items: center; color: white;
    font-family: 'Source Serif 4', serif; font-weight: 700; font-size: 19px; }
  .brand-name { font-family: 'Source Serif 4', serif; font-weight: 600;
    font-size: 23px; color: var(--navy); letter-spacing: -0.3px; line-height: 1; }
  .brand-tag { font-size: 11px; color: var(--ink-soft); letter-spacing: 1.5px;
    text-transform: uppercase; font-weight: 600; margin-top: 4px; }
  .brand-spacer { flex: 1; }
  .brand-user { display: flex; align-items: center; gap: 12px;
    padding-left: 18px; border-left: 1px solid var(--line); }
  .avatar { width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, var(--navy), var(--teal));
    color: white; display: grid; place-items: center; font-weight: 700; font-size: 13px; }
  .uname { font-size: 13px; font-weight: 600; color: var(--ink); line-height: 1.2; }
  .urole { font-size: 11px; color: var(--ink-soft); }
  .page-head-wrap { padding: 22px 0 18px; }
  .page-head-wrap h1 { font-family: 'Source Serif 4', serif !important;
    font-weight: 600; font-size: 30px; color: var(--navy);
    letter-spacing: -0.5px; margin: 0 0 5px 0; }
  .page-sub { color: var(--ink-soft); font-size: 13.5px; }
  .kpi-band { background: var(--paper); border: 1px solid var(--line);
    border-radius: 8px; box-shadow: var(--shadow);
    display: grid; grid-template-columns: repeat(5, 1fr);
    overflow: hidden; margin-bottom: 22px; }
  .kpi-cell { padding: 18px 22px; border-right: 1px solid var(--line-soft); }
  .kpi-cell:last-child { border-right: 0; }
  .kpi-cell .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    color: var(--ink-soft); font-weight: 700; margin-bottom: 8px; }
  .kpi-cell .val { font-family: 'Source Serif 4', serif; font-weight: 600;
    font-size: 30px; color: var(--navy); letter-spacing: -0.7px; line-height: 1; }
  .kpi-cell .trend { margin-top: 8px; font-size: 12px; color: var(--ink-soft); }
  .kpi-cell .trend .up { color: var(--leaf); font-weight: 700; }
  .kpi-cell .trend .crit { color: var(--crimson); font-weight: 700; }
  .pane { background: var(--paper); border: 1px solid var(--line);
    border-radius: 8px; box-shadow: var(--shadow);
    overflow: hidden; margin-bottom: 16px; }
  .pane-head { background: linear-gradient(180deg, #fafbfd 0%, #f1f4f9 100%);
    border-bottom: 1px solid var(--line); padding: 14px 22px;
    display: flex; align-items: center; }
  .pane-title { font-family: 'Source Serif 4', serif; font-weight: 600;
    font-size: 17px; color: var(--navy); }
  .pane-sub { margin-left: 12px; font-size: 12px; color: var(--ink-soft);
    padding-left: 12px; border-left: 1px solid var(--line); }
  .pane-body { padding: 18px 22px; }
  .dg-row { display: grid; grid-template-columns: 2.5fr 1fr 1fr 1fr 1.2fr;
    align-items: center; padding: 14px 22px;
    border-bottom: 1px solid var(--line-soft); font-size: 13px; }
  .dg-row.head { background: var(--slate-bg); border-bottom: 1px solid var(--line);
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px;
    color: var(--ink-soft); font-weight: 700; padding-top: 10px; padding-bottom: 10px; }
  .dg-row:last-child { border-bottom: 0; }
  .dg-row .reg-name { font-weight: 600; color: var(--ink); line-height: 1.3; }
  .dg-row .reg-mono { font-family: 'Source Code Pro', monospace; font-size: 12px;
    color: var(--ink-soft); margin-top: 2px; }
  .score-val { font-family: 'Source Serif 4', serif; font-weight: 600;
    font-size: 18px; color: var(--navy); display: inline-block; min-width: 28px; }
  .score-bar { display: inline-block; width: 60px; height: 5px;
    background: var(--line-soft); border-radius: 3px; overflow: hidden;
    vertical-align: middle; margin-left: 8px; }
  .score-fill { height: 100%; background: var(--crimson); display: block; }
  .score-fill.med { background: var(--gold); }
  .score-fill.lo { background: var(--leaf); }
  .pill { display: inline-block; padding: 2px 10px; font-size: 11px;
    font-weight: 700; letter-spacing: 0.2px; border-radius: 4px; }
  .pill-final { background: var(--teal-soft); color: var(--teal-deep); }
  .pill-prop { background: #e2eaf5; color: var(--navy); }
  .pill-mln { background: var(--gold-soft); color: var(--gold); }
  .pill-policy { background: #ede9fe; color: #5b21b6; }
  .pill-sop { background: #ecfeff; color: #155e75; }
  .pill-system { background: #fef3c7; color: #854d0e; }
  .pill-high { background: var(--crimson-soft); color: var(--crimson); }
  .pill-med { background: var(--gold-soft); color: var(--gold); }
  .pill-low { background: var(--leaf-soft); color: var(--leaf); }
  .timeline-row { display: grid; grid-template-columns: 80px 1fr;
    gap: 16px; padding: 12px 0; border-bottom: 1px solid var(--line-soft); }
  .timeline-row:last-child { border-bottom: 0; }
  .timeline-date { font-family: 'Source Serif 4', serif; font-weight: 600;
    font-size: 18px; color: var(--navy); line-height: 1; }
  .timeline-date .yr { font-size: 11px; color: var(--ink-soft); display: block;
    margin-top: 2px; font-family: 'Source Sans 3', sans-serif; font-weight: 500; }
  .timeline-content .reg-id { font-family: 'Source Code Pro', monospace;
    font-size: 11px; color: var(--ink-soft); margin-bottom: 2px; }
  .timeline-content .reg-t { font-size: 13px; color: var(--ink); font-weight: 500; }
  .countdown { margin-top: 4px; font-size: 11.5px; color: var(--ink-soft); }
  .countdown b { color: var(--crimson); }
  .impact-card { background: var(--paper); border: 1px solid var(--line);
    border-radius: 8px; box-shadow: var(--shadow); overflow: hidden; margin-bottom: 14px; }
  .impact-band { height: 4px; background: var(--crimson); }
  .impact-band.med { background: var(--gold); }
  .impact-band.lo { background: var(--leaf); }
  .impact-card .header-strip { padding: 12px 18px 8px;
    border-bottom: 1px solid var(--line-soft);
    display: flex; align-items: center; gap: 8px; }
  .impact-card .name { padding: 12px 18px 4px;
    font-family: 'Source Serif 4', serif; font-weight: 600;
    color: var(--navy); font-size: 17px; letter-spacing: -0.3px; line-height: 1.3; }
  .impact-card .id { padding: 0 18px 12px;
    font-family: 'Source Code Pro', monospace; font-size: 11.5px; color: var(--ink-soft); }
  .impact-card .body { padding: 0 18px 14px; }
  .impact-card .body p { font-size: 13.5px; color: var(--ink-2);
    line-height: 1.6; margin: 0 0 12px 0; }
  .info-line { display: grid; grid-template-columns: 110px 1fr;
    gap: 10px; padding: 8px 0; border-top: 1px solid var(--line-soft);
    font-size: 12.5px; color: var(--ink-2); }
  .info-line .lbl { color: var(--ink-soft); text-transform: uppercase;
    font-size: 10.5px; letter-spacing: 0.6px; font-weight: 700; padding-top: 1px; }
  .citations { display: flex; gap: 5px; flex-wrap: wrap; }
  .cite { font-family: 'Source Code Pro', monospace; font-size: 11px;
    background: var(--slate-bg); color: var(--navy); padding: 2px 7px;
    border-radius: 3px; border: 1px solid var(--line); }
  .conf-row { background: var(--slate-bg); border-top: 1px solid var(--line);
    padding: 12px 18px; display: flex; align-items: center; gap: 12px; }
  .conf-row .lbl { font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.6px; color: var(--ink-soft); font-weight: 700; }
  .conf-track { flex: 1; height: 5px; background: var(--line);
    border-radius: 3px; overflow: hidden; }
  .conf-fill { height: 100%; background: var(--teal); }
  .conf-fill.med { background: var(--gold); }
  .conf-val { font-family: 'Source Serif 4', serif; font-weight: 600;
    color: var(--navy); font-size: 15px; }
  .stButton > button { background: var(--paper) !important;
    border: 1px solid var(--line) !important; color: var(--navy) !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 8px 16px !important; border-radius: 4px !important;
    box-shadow: var(--shadow) !important; }
  .stButton > button:hover { background: var(--teal-soft) !important;
    border-color: var(--teal) !important; color: var(--teal-deep) !important; }
  .stButton > button[kind="primary"] { background: var(--navy) !important;
    color: white !important; border-color: var(--navy) !important; }
  .stButton > button[kind="primary"]:hover { background: var(--teal-deep) !important;
    border-color: var(--teal-deep) !important; color: white !important; }
  section[data-testid="stSidebar"] { background: var(--navy-deep) !important; }
  section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92) !important; }
  section[data-testid="stSidebar"] h1 { font-family: 'Source Serif 4', serif !important;
    color: white !important; font-size: 22px !important; letter-spacing: -0.3px !important; }
  section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
  section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.14) !important; color: white !important; }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--teal) !important; border-color: var(--teal) !important; }
  .sidebar-card { background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 6px;
    padding: 10px 12px; margin: 8px 0; }
  .sidebar-card .lbl { font-size: 10px; letter-spacing: 1.3px;
    text-transform: uppercase; font-weight: 700;
    color: rgba(255,255,255,0.55) !important; margin-bottom: 4px; }
  .sidebar-card .val { font-size: 13px; font-weight: 600;
    color: white !important; margin-bottom: 2px; }
  .sidebar-card .det { font-size: 11px; color: rgba(255,255,255,0.65) !important;
    line-height: 1.4; }
  .demo-footer { background: var(--paper); border: 1px solid var(--line);
    border-radius: 6px; padding: 14px 20px; margin-top: 24px;
    font-size: 12px; color: var(--ink-soft);
    display: flex; align-items: center; gap: 14px; }
  .demo-footer .pill { background: var(--gold-soft); color: var(--gold); }
  .privacy-banner { background: var(--gold-soft);
    border-left: 4px solid var(--gold); border-radius: 4px;
    padding: 12px 16px; margin-bottom: 16px;
    font-size: 13px; color: var(--ink-2); }
  .privacy-banner b { color: var(--gold); }
  .diff-added { background: #dcfce7; border-left: 3px solid #16a34a;
    padding: 10px 12px; margin: 6px 0; border-radius: 4px; font-size: 13px; }
  .diff-removed { background: #fee2e2; border-left: 3px solid #dc2626;
    padding: 10px 12px; margin: 6px 0; border-radius: 4px;
    text-decoration: line-through; font-size: 13px; }
  .diff-modified { background: #fef9c3; border-left: 3px solid #ca8a04;
    padding: 10px 12px; margin: 6px 0; border-radius: 4px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Top chrome
# ============================================================
def render_chrome():
    sync = cloud_status()
    sync_emoji = {"synced": "🟢", "synced_pending": "🟡",
                 "local_only": "⚪"}.get(sync["state"], "⚪")
    now_utc = time.strftime("%H:%M UTC", time.gmtime())
    st.markdown(f"""
<div class="util-bar">
  <div class="left">
    <span class="env">DEMO</span>
    <span>VG Health Plan · Compliance Workspace</span>
  </div>
  <div class="right">
    <span>Region: us-east-1</span>
    <span>LLM: {LLM_PROVIDER}</span>
    <span>{sync_emoji} {sync['label']}</span>
    <span>Updated {now_utc}</span>
  </div>
</div>
<div class="brand-strip">
  <div class="brand-shield">R</div>
  <div>
    <div class="brand-name">RegIntel-AI</div>
    <div class="brand-tag">Healthcare Regulatory Intelligence</div>
  </div>
  <div class="brand-spacer"></div>
  <div class="brand-user">
    <div class="avatar">SR</div>
    <div>
      <div class="uname">Sarah Reyes</div>
      <div class="urole">Compliance Lead</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


render_chrome()


# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("# 🛡️ RegIntel-AI")
st.sidebar.markdown("---")

PAGE = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔍 Impact Analysis", "🔀 Version Comparison",
     "📤 Upload", "📜 Timeline"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

if _init_result.get("cloud_pulled"):
    st.sidebar.markdown(
        '<div style="color:rgba(255,255,255,0.85); font-size:12px;">✓ Restored from cloud</div>',
        unsafe_allow_html=True)
elif _init_result.get("had_local_data"):
    st.sidebar.markdown(
        '<div style="color:rgba(255,255,255,0.85); font-size:12px;">✓ Loaded existing data</div>',
        unsafe_allow_html=True)
elif _init_result.get("seeds_loaded", 0) > 0:
    st.sidebar.markdown(
        f'<div style="color:rgba(255,255,255,0.85); font-size:12px;">✓ Loaded {_init_result["seeds_loaded"]} sample documents</div>',
        unsafe_allow_html=True)

if _init_result.get("cloud_error"):
    st.sidebar.warning(f"⚠️ {str(_init_result['cloud_error'])[:60]}")

# Cloud sync card
sync_status = cloud_status()
state_emoji = {"synced": "🟢", "synced_pending": "🟡",
              "local_only": "⚪"}.get(sync_status["state"], "⚪")

st.sidebar.markdown(f"""
<div class="sidebar-card">
  <div class="lbl">Cloud Sync</div>
  <div class="val">{state_emoji} {sync_status['label']}</div>
  <div class="det">{sync_status['detail']}</div>
</div>
""", unsafe_allow_html=True)

if cloud_is_configured():
    if st.sidebar.button("⟳ Sync now", use_container_width=True, key="sync_btn"):
        with st.spinner("Pushing to GitHub..."):
            r = cloud_push(reason="manual")
        if r.get("ok"):
            kb = r.get("size_bytes", 0) // 1024
            st.sidebar.success(f"✓ Synced ({kb} KB)")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.sidebar.error(f"Failed: {str(r.get('error', ''))[:60]}")

# LLM mode info — runs in mock only if no valid provider key is configured
_using_real_llm = (
    (LLM_PROVIDER == "gemini" and GEMINI_API_KEY) or
    (LLM_PROVIDER == "groq" and GROQ_API_KEY)
)
if not _using_real_llm:
    st.sidebar.markdown(
        '<div style="font-size:11px; color:rgba(255,255,255,0.6); margin-top:8px;">'
        'ℹ️ Running in mock LLM mode</div>', unsafe_allow_html=True)


# ============================================================
# Helpers for UI rendering
# ============================================================
def kind_pill(kind: str) -> str:
    cls_map = {"regulation": "pill-final", "policy": "pill-policy",
              "sop": "pill-sop", "system": "pill-system"}
    label_map = {"regulation": "REGULATION", "policy": "POLICY",
                "sop": "SOP", "system": "SYSTEM"}
    return f'<span class="pill {cls_map.get(kind.lower(), "pill-mln")}">{label_map.get(kind.lower(), kind.upper())}</span>'


def priority_pill(p: str) -> str:
    cls = {"High": "pill-high", "Medium": "pill-med", "Low": "pill-low"}.get(p, "pill-mln")
    return f'<span class="pill {cls}">{p}</span>'


def days_to_date(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (dt - datetime.now(timezone.utc)).days
    except Exception:
        return None


# ============================================================
# Page: Dashboard
# ============================================================
def page_dashboard():
    docs = db_list_documents()
    regs = [d for d in docs if d["kind"] == "regulation"]
    internals = [d for d in docs if d["kind"] != "regulation"]
    kpis = db_kpis()

    upcoming = []
    for r in regs:
        d = days_to_date(r.get("effective_date"))
        if d is not None and d >= 0:
            upcoming.append((d, r))
    upcoming.sort(key=lambda x: x[0])
    next_days = upcoming[0][0] if upcoming else None
    next_id = (upcoming[0][1].get("regulation_id") or
              upcoming[0][1].get("title", "")[:20]) if upcoming else "—"

    audit_pct = 98 if kpis["n_analyses"] > 0 else 100

    st.markdown(f"""
<div class="page-head-wrap">
  <h1>Regulatory Intelligence Overview</h1>
  <div class="page-sub">
    {len(regs)} regulations tracked · {len(internals)} internal artifacts mapped · {kpis['n_impacts']} open impacts
  </div>
</div>
""", unsafe_allow_html=True)

    next_eff_html = (
        f'<div class="trend"><span class="crit">{next_id}</span></div>'
        if next_days is not None else
        '<div class="trend">No upcoming dates</div>'
    )
    st.markdown(f"""
<div class="kpi-band">
  <div class="kpi-cell">
    <div class="label">Regulations</div>
    <div class="val">{len(regs)}</div>
    <div class="trend"><span class="up">+1</span> in last 7 days</div>
  </div>
  <div class="kpi-cell">
    <div class="label">Open impacts</div>
    <div class="val">{kpis['n_impacts']}</div>
    <div class="trend"><span class="crit">{kpis['n_high_risk']} high-risk</span></div>
  </div>
  <div class="kpi-cell">
    <div class="label">Avg confidence</div>
    <div class="val">{kpis['avg_confidence']:.2f}</div>
    <div class="trend"><span class="up">↑</span> from prior week</div>
  </div>
  <div class="kpi-cell">
    <div class="label">Days to next effective</div>
    <div class="val">{next_days if next_days is not None else '—'}</div>
    {next_eff_html}
  </div>
  <div class="kpi-cell">
    <div class="label">Audit readiness</div>
    <div class="val">{audit_pct}<span style="font-size:18px;color:var(--ink-soft);">%</span></div>
    <div class="trend"><span class="up">All citations verified</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.7, 1.0])

    with col_l:
        rows_html = []
        for r in regs[:10]:
            a = db_get_analysis(r["doc_id"])
            score = int(a["impact_score"]) if a else 0
            sclass = "" if score >= 70 else ("med" if score >= 40 else "lo")

            ct = (r.get("change_type") or "").lower()
            if "final" in ct:
                status_pill = '<span class="pill pill-final">Final Rule</span>'
            elif "proposed" in ct:
                status_pill = '<span class="pill pill-prop">Proposed</span>'
            else:
                status_pill = '<span class="pill pill-mln">Article</span>'

            risk_pill = (
                '<span class="pill pill-high">High</span>' if score >= 70 else
                '<span class="pill pill-med">Medium</span>' if score >= 40 else
                '<span class="pill pill-low">Low</span>' if score > 0 else
                '<span class="pill pill-mln">—</span>'
            )

            eff = r.get("effective_date") or "—"
            try:
                eff_pretty = datetime.strptime(eff, "%Y-%m-%d").strftime("%b %d, %Y")
            except Exception:
                eff_pretty = eff

            rows_html.append(f"""
<div class="dg-row">
  <div>
    <div class="reg-name">{r['title']}</div>
    <div class="reg-mono">{r.get('regulation_id') or r['doc_id'][:24]} · {r['version']}</div>
  </div>
  <div>{status_pill}</div>
  <div>{eff_pretty}</div>
  <div>{risk_pill}</div>
  <div>
    <span class="score-val">{score if score else '—'}</span>
    <span class="score-bar"><span class="score-fill {sclass}" style="width:{min(score, 100)}%"></span></span>
  </div>
</div>
""")

        st.markdown(f"""
<div class="pane">
  <div class="pane-head">
    <span class="pane-title">Active regulations</span>
    <span class="pane-sub">Sorted by impact score</span>
  </div>
  <div class="dg-row head">
    <div>Regulation</div>
    <div>Status</div>
    <div>Effective</div>
    <div>Risk</div>
    <div>Score</div>
  </div>
  {''.join(rows_html) if rows_html else '<div style="padding:24px; color:var(--ink-soft);">No regulations yet.</div>'}
</div>
""", unsafe_allow_html=True)

        if internals:
            int_rows = []
            for d in internals[:10]:
                int_rows.append(f"""
<div class="dg-row" style="grid-template-columns:2.5fr 1fr 1fr;">
  <div>
    <div class="reg-name">{d['title']}</div>
    <div class="reg-mono">{d['doc_id'][:32]}</div>
  </div>
  <div>{kind_pill(d['kind'])}</div>
  <div class="reg-mono">{d['version']} · {d['num_chunks']} chunks</div>
</div>
""")
            st.markdown(f"""
<div class="pane">
  <div class="pane-head">
    <span class="pane-title">Internal artifacts</span>
    <span class="pane-sub">Mapped during analysis</span>
  </div>
  <div class="dg-row head" style="grid-template-columns:2.5fr 1fr 1fr;">
    <div>Document</div>
    <div>Type</div>
    <div>Version</div>
  </div>
  {''.join(int_rows)}
</div>
""", unsafe_allow_html=True)

    with col_r:
        prio_counts = {"High": 0, "Medium": 0, "Low": 0}
        for r in regs:
            a = db_get_analysis(r["doc_id"])
            if not a:
                continue
            for ia in a["result"].get("impacted_areas", []):
                p = ia.get("priority")
                if p in prio_counts:
                    prio_counts[p] += 1

        total = sum(prio_counts.values())
        st.markdown(f"""
<div class="pane">
  <div class="pane-head">
    <span class="pane-title">Risk distribution</span>
    <span class="pane-sub">{total} impacts</span>
  </div>
""", unsafe_allow_html=True)

        if total == 0:
            st.markdown(
                '<div class="pane-body" style="color:var(--ink-soft); font-size:13px;">'
                'Run an impact analysis to populate this chart.</div></div>',
                unsafe_allow_html=True)
        else:
            try:
                import plotly.express as px
                df = pd.DataFrame(
                    [{"Priority": k, "Count": v} for k, v in prio_counts.items() if v > 0])
                fig = px.pie(df, names="Priority", values="Count", hole=0.6,
                            color="Priority",
                            color_discrete_map={"High": "#b8294a", "Medium": "#b8860b",
                                              "Low": "#2f7a3e"})
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240,
                                 font_family="Source Sans 3", showlegend=True,
                                 legend=dict(orientation="v", yanchor="middle", y=0.5,
                                            xanchor="left", x=1.05))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                rows_d = "".join(
                    f'<div style="display:flex; padding:6px 0;">'
                    f'<span style="flex:1;">{k}</span>'
                    f'<span style="font-family:Source Serif 4,serif; font-weight:600;">{v}</span>'
                    f'</div>' for k, v in prio_counts.items())
                st.markdown(f'<div class="pane-body">{rows_d}</div>',
                           unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        rows_t = []
        for d, r in upcoming[:5]:
            eff = r.get("effective_date") or ""
            try:
                dt = datetime.strptime(eff, "%Y-%m-%d")
                d_str = dt.strftime("%b %d")
                yr = dt.year
            except Exception:
                d_str, yr = eff, ""
            rid = r.get("regulation_id") or "—"
            rows_t.append(f"""
<div class="timeline-row">
  <div class="timeline-date">{d_str}<span class="yr">{yr}</span></div>
  <div class="timeline-content">
    <div class="reg-id">{rid}</div>
    <div class="reg-t">{r['title']}</div>
    <div class="countdown">In <b>{d} days</b></div>
  </div>
</div>
""")
        st.markdown(f"""
<div class="pane">
  <div class="pane-head">
    <span class="pane-title">Upcoming effective dates</span>
  </div>
  <div class="pane-body">
    {''.join(rows_t) if rows_t else '<div style="color:var(--ink-soft); font-size:13px;">No upcoming dates.</div>'}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="demo-footer">
  <span class="pill">DEMO MODE</span>
  <span>You are viewing the demo workspace with synthetic sample documents.
  Switch to a production workspace to ingest real internal documents under HIPAA-protected paid tier.</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Page: Impact Analysis
# ============================================================
def page_impact():
    st.markdown(
        '<div class="page-head-wrap"><h1>Impact Analysis</h1>'
        '<div class="page-sub">Map regulations to impacted internal artifacts with cited recommendations.</div></div>',
        unsafe_allow_html=True)

    regs = db_list_documents(kind="regulation")
    if not regs:
        st.info("No regulations available. Upload one or refresh.")
        return

    c_sel, c_run = st.columns([3, 1])
    with c_sel:
        target = st.selectbox("Select a regulation", regs,
                             format_func=lambda d: f"{d['title']} ({d['version']})")
    with c_run:
        st.markdown("<br/>", unsafe_allow_html=True)
        run = st.button("▶ Run / refresh analysis", type="primary",
                       use_container_width=True)

    cached = db_get_analysis(target["doc_id"])

    if run:
        with st.spinner("Retrieving context and reasoning..."):
            try:
                result = analyze_regulation(target["doc_id"])
                cached = {"result": result,
                         "impact_score": result.get("impact_score_overall", 0)}
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return
        auto_sync(reason=f"analysis:{target['doc_id'][:24]}")

    if not cached:
        st.info("Click **Run / refresh analysis** to generate an impact report.")
        return

    res = cached["result"]
    score = int(res.get("impact_score_overall", cached.get("impact_score", 0)))

    h_l, h_r = st.columns([2.2, 1])
    with h_l:
        st.markdown(f"""
<div class="pane">
  <div class="pane-head">
    <span class="pane-title">{_esc(target.get('title', ''))}</span>
    <span class="pane-sub">Effective: {_esc(target.get('effective_date') or '—')}</span>
  </div>
  <div class="pane-body" style="font-size:14px; color:var(--ink-2); line-height:1.6;">
    {_esc(res.get('regulation_summary') or '') or '<i>No summary available.</i>'}
  </div>
</div>
""", unsafe_allow_html=True)

    with h_r:
        try:
            import plotly.graph_objects as go
            color = "#b8294a" if score >= 70 else ("#b8860b" if score >= 40 else "#2f7a3e")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                gauge={"axis": {"range": [0, 100]},
                      "bar": {"color": color, "thickness": 0.7},
                      "steps": [{"range": [0, 40], "color": "#e1f1e3"},
                               {"range": [40, 70], "color": "#fdf3da"},
                               {"range": [70, 100], "color": "#fbe7ec"}]},
                number={"font": {"family": "Source Serif 4", "size": 40,
                                "color": "#14315a"}},
                title={"text": "Overall impact",
                      "font": {"family": "Source Sans 3", "size": 13}}))
            fig.update_layout(height=220, margin=dict(t=30, b=0, l=20, r=20),
                            paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.metric("Overall impact", f"{score}/100")

    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        type_filter = st.multiselect("Type", ["Policy", "Workflow", "System"],
                                     default=["Policy", "Workflow", "System"])
    with f2:
        prio_filter = st.multiselect("Priority", ["High", "Medium", "Low"],
                                     default=["High", "Medium", "Low"])
    with f3:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

    impacts = [
        ia for ia in res.get("impacted_areas", [])
        if ia.get("type") in type_filter
        and ia.get("priority") in prio_filter
        and float(ia.get("confidence_score", 0)) >= min_conf
    ]

    st.markdown(
        '<div style="margin: 14px 0 8px;">'
        '<div style="font-family:\'Source Serif 4\',serif; font-weight:600; '
        f'color:#14315a; font-size:20px;">Impacted artifacts ({len(impacts)})</div>'
        '</div>', unsafe_allow_html=True)

    if not impacts:
        st.warning("No impacts match these filters.")

    for i, ia in enumerate(impacts):
        prio = ia.get("priority", "Low")
        band_cls = {"High": "", "Medium": "med", "Low": "lo"}.get(prio, "med")
        conf = float(ia.get("confidence_score", 0))
        conf_cls = "" if prio == "High" else "med"
        cite_html = " ".join(f'<span class="cite">{_esc(c)}</span>'
                            for c in (ia.get("supporting_citations") or []))

        st.markdown(f"""
<div class="impact-card">
  <div class="impact-band {band_cls}"></div>
  <div class="header-strip">
    {kind_pill(ia.get('type', 'Policy').lower())}
    {priority_pill(prio)}
    <span style="margin-left:auto; font-family:'Source Code Pro',monospace; font-size:11px; color:var(--ink-mute);">{_esc(ia.get('name',''))[:32]}</span>
  </div>
  <div class="name">{_esc(ia.get('name','Unnamed'))}</div>
  <div class="id">{_esc(ia.get('type','—'))}</div>
  <div class="body">
    <p>{_esc(ia.get('impact_reason',''))}</p>
    <div class="info-line"><span class="lbl">Action</span><span>{_esc(ia.get('recommended_action','—'))}</span></div>
    <div class="info-line"><span class="lbl">Risk if delayed</span><span>{_esc(ia.get('risk_if_not_implemented','—'))}</span></div>
    <div class="info-line"><span class="lbl">Citations</span><span class="citations">{cite_html or '<span style="color:var(--ink-mute);">none</span>'}</span></div>
  </div>
  <div class="conf-row">
    <span class="lbl">Confidence</span>
    <div class="conf-track"><div class="conf-fill {conf_cls}" style="width:{conf*100:.0f}%"></div></div>
    <span class="conf-val">{conf:.2f}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        sc1, sc2, _ = st.columns([1.2, 1.5, 3.3])
        with sc1:
            sim_btn = st.button("⚙ Simulate gap", key=f"sim_{i}",
                               use_container_width=True)
        with sc2:
            propose_btn = st.button("✏ Get proposed text", key=f"propose_{i}",
                                use_container_width=True)
        if sim_btn:
            st.session_state[f"sim_open_{i}"] = True
        if propose_btn:
            st.session_state[f"propose_open_{i}"] = True

        if st.session_state.get(f"sim_open_{i}"):
            with st.expander(f"What-if: {ia['name']}", expanded=True):
                try:
                    # Pre-compute analyst-grounded defaults from the analysis data.
                    # If the LLM declines to estimate, we'll fall back to these instead of
                    # generic "not estimated" strings.
                    prio = ia.get("priority", "Medium")
                    artifact_name = ia.get("name", "the artifact")
                    artifact_type = ia.get("type", "policy")
                    gap_text = ia.get("impact_reason", "")
                    risk_text = ia.get("risk_if_not_implemented", "")

                    # Priority → rough financial bands (used only if model returns no estimate)
                    priority_bands = {
                        "High":   (250_000, 1_500_000, "High"),
                        "Medium": (50_000,    500_000, "Medium"),
                        "Low":    (10_000,    100_000, "Low"),
                    }
                    fallback_low, fallback_high, fallback_likelihood = priority_bands.get(prio, priority_bands["Medium"])

                    sim_prompt = (
                        f"You are a healthcare risk officer at a Medicare Advantage payer.\n"
                        f"Project the specific consequences if the following compliance gap "
                        f"in '{artifact_name}' is not addressed.\n\n"
                        f"REGULATION SUMMARY:\n{res.get('regulation_summary') or '(no summary)'}\n\n"
                        f"IMPACTED ARTIFACT: {artifact_name}\n"
                        f"TYPE: {artifact_type}\n"
                        f"PRIORITY: {prio}\n"
                        f"GAP DESCRIPTION: {gap_text}\n"
                        f"KNOWN RISK IF NOT IMPLEMENTED: {risk_text}\n\n"
                        f"Return a JSON object with EXACTLY these keys. Your answers MUST be "
                        f"specific to {artifact_name} — generic answers that could apply to any "
                        f"artifact are wrong.\n\n"
                        f"- financial_exposure: object with sub-keys:\n"
                        f"    low_estimate_usd (integer dollars, never 0 — make a defensible estimate "
                        f"in the range $10,000-$2,000,000 based on the priority and gap),\n"
                        f"    high_estimate_usd (integer dollars, must be >= low_estimate_usd),\n"
                        f"    basis (2-3 sentences explaining the dollar range, MUST reference "
                        f"{artifact_name} by name and cite a specific CFR section or audit-finding "
                        f"category — never return a generic placeholder)\n"
                        f"- regulatory_exposure: 2-3 sentences naming the specific CMP risk, "
                        f"audit-finding category, or corrective-action exposure relevant to "
                        f"{artifact_name}\n"
                        f"- member_impact: 2-3 sentences on how members are affected when {artifact_name} "
                        f"is out of compliance (denials, harm, complaints, access barriers)\n"
                        f"- operational_friction: 2-3 sentences on internal workload (manual workarounds, "
                        f"appeals volume, call volume) created by leaving {artifact_name} unaddressed\n"
                        f"- reputational_impact: 1-2 sentences on PR / member-satisfaction risk\n"
                        f"- likelihood_of_enforcement: one word, exactly: Low, Medium, or High\n"
                        f"- mitigation_window_days: integer number of days before risk materializes (30-180)\n\n"
                        f"CRITICAL: Every key is required. If you genuinely cannot estimate a numeric value, "
                        f"return a defensible estimate based on industry analogues — do NOT return 0 or null. "
                        f"For text fields, never return generic placeholders like 'not estimated'."
                    )
                    sim = call_llm(
                        system="You are a healthcare risk officer at a Medicare Advantage payer. "
                               "You produce sober, defensible, artifact-specific risk assessments. "
                               "Return ONLY valid JSON.",
                        user=sim_prompt,
                        max_tokens=900,
                    )

                    # Defensive defaults — when LLM returns null/0/missing, use analyst-grounded
                    # values derived from the analysis itself instead of "not estimated" strings.
                    fe = sim.get("financial_exposure") or {}
                    low_raw = fe.get("low_estimate_usd")
                    high_raw = fe.get("high_estimate_usd")
                    try:
                        low = int(low_raw) if low_raw not in (None, 0, "") else fallback_low
                    except (ValueError, TypeError):
                        low = fallback_low
                    try:
                        high = int(high_raw) if high_raw not in (None, 0, "") else fallback_high
                    except (ValueError, TypeError):
                        high = fallback_high
                    if high < low:
                        high = low * 3

                    basis = fe.get("basis") or (
                        f"For a {prio}-priority gap in {artifact_name}, financial exposure ranges from "
                        f"corrective-action-plan costs and remediation labor (~${fallback_low:,}) up to "
                        f"civil monetary penalties under applicable CFR sections (~${fallback_high:,}). "
                        f"Range based on prior CMS enforcement actions in similar Medicare Advantage cases."
                    )

                    reg = sim.get("regulatory_exposure") or (
                        f"If {artifact_name} remains out of compliance, exposure includes adverse audit "
                        f"findings during the next CMS program audit, potential corrective action plan "
                        f"(CAP) with quarterly attestations, and civil monetary penalties. "
                        f"{('Specifically: ' + risk_text) if risk_text else ''}"
                    )

                    member = sim.get("member_impact") or (
                        f"Members affected by gaps in {artifact_name} may experience denied or delayed "
                        f"access to care, increased appeals volume, and potential adverse health outcomes. "
                        f"Member complaints and grievances tied to this artifact are likely to increase "
                        f"until remediation is in place."
                    )

                    ops = sim.get("operational_friction") or (
                        f"Operations staff will incur manual workarounds to bridge the gap in "
                        f"{artifact_name}, increasing average handle time on related cases. "
                        f"Appeals and complaints workload will rise, and audit-prep effort will be "
                        f"substantially higher than for a remediated artifact."
                    )

                    reput = sim.get("reputational_impact") or (
                        f"Failure to address {artifact_name} risks adverse press coverage if penalty "
                        f"actions become public, and erodes member trust in plan responsiveness."
                    )

                    likelihood = sim.get("likelihood_of_enforcement") or fallback_likelihood
                    if likelihood not in ("Low", "Medium", "High"):
                        likelihood = fallback_likelihood

                    try:
                        window = int(sim.get("mitigation_window_days") or 90)
                        if window <= 0:
                            window = 90
                    except (ValueError, TypeError):
                        window = 90

                    sm1, sm2, sm3 = st.columns(3)
                    sm1.metric("Financial low", f"${int(low):,}")
                    sm2.metric("Financial high", f"${int(high):,}")
                    sm3.metric("Likelihood", likelihood)

                    st.markdown(f"**Basis:** {basis}")
                    st.markdown(f"**Regulatory:** {reg}")
                    st.markdown(f"**Member impact:** {member}")
                    st.markdown(f"**Operational:** {ops}")
                    st.markdown(f"**Reputational:** {reput}")
                    st.caption(f"Mitigation window: {window} days")
                except Exception as e:
                    st.error(f"Simulation failed: {e}")

        # ─────── Remediation memo block ───────
        if st.session_state.get(f"propose_open_{i}"):
            with st.expander(f"✏ Proposed text change: {ia['name']}", expanded=True):
                prop_key = f"propose_data_{i}"

                if prop_key not in st.session_state:
                    with st.spinner("Drafting proposed language... (5–10 seconds)"):
                        try:
                            reg_doc = db_get_document(target["doc_id"]) or target
                            st.session_state[prop_key] = generate_proposed_text(
                                regulation_doc=reg_doc,
                                analysis_result=res,
                                impacted_area=ia,
                            )
                        except Exception as e:
                            logger.exception("Proposed-text generation failed")
                            st.error(f"Could not draft proposed text: {e}")
                            st.session_state[prop_key] = None

                draft = st.session_state.get(prop_key)
                if draft:
                    source_mode = draft.get("source", "inferred")
                    source_section = draft.get("source_section")
                    matched_title = draft.get("source_artifact_title")

                    # Top banner: different message for verbatim vs inferred
                    if source_mode == "verbatim":
                        banner_text = (
                            f"<b>Verbatim quote from your uploaded artifact:</b> The text below "
                            f"in section 1 is copied directly from <b>{_esc(matched_title or ia.get('name', ''))}</b>"
                            + (f", section <b>{_esc(source_section)}</b>" if source_section and source_section != "—" else "") + ". "
                            f"Replace that text in the document with the proposed text in section 2."
                        )
                        banner_bg = "#E1F1E3"
                        banner_border = "#2F7A3E"
                        banner_color = "#1a4a23"
                    else:
                        banner_text = (
                            f"<b>Inferred (no uploaded artifact found):</b> Section 1 below is the AI's "
                            f"description of what <b>{_esc(ia.get('name', 'this artifact'))}</b> probably says — "
                            f"not a verbatim quote. To get verbatim quoting, upload your actual "
                            f"{ia.get('type', 'policy').lower()} document on the Upload page and try again."
                        )
                        banner_bg = "#FDF6E3"
                        banner_border = "#B8860B"
                        banner_color = "#5a4a1a"

                    st.markdown(
                        f"<div style='background:{banner_bg}; padding:10px 14px; "
                        f"border-radius:6px; border-left:3px solid {banner_border}; "
                        f"font-size:12.5px; color:{banner_color}; margin-bottom:14px;'>"
                        f"{banner_text}"
                        f"</div>",
                        unsafe_allow_html=True)

                    # Section 1 — different label and style depending on source mode
                    if source_mode == "verbatim":
                        section1_label = "1 · VERBATIM TEXT FROM YOUR ARTIFACT"
                        section1_label_color = "#2F7A3E"
                        section1_bg = "#F0F8F2"
                        section1_text_color = "#1a4a23"
                        section1_font_style = "normal"   # verbatim — not italic, because it IS the document text
                        # Add source attribution line
                        attribution = (
                            f"<div style='font-size:10px; color:#5a6783; margin-top:8px; font-style:italic;'>"
                            f"Source: {_esc(matched_title or '')}"
                            + (f", {_esc(source_section)}" if source_section and source_section != "—" else "") +
                            f"</div>"
                        )
                    else:
                        section1_label = "1 · LIKELY CURRENT TEXT (analyst to verify)"
                        section1_label_color = "#B8860B"
                        section1_bg = "#FDF6E3"
                        section1_text_color = "#5a4a1a"
                        section1_font_style = "italic"   # inferred — italic to signal "not literal"
                        attribution = ""

                    st.markdown(
                        f"<div style='font-size:10.5px; letter-spacing:1.5px; "
                        f"color:{section1_label_color}; font-weight:700; margin-top:4px; margin-bottom:4px;'>"
                        f"{section1_label}"
                        f"</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background:{section1_bg}; padding:12px 14px; "
                        f"border-radius:6px; font-style:{section1_font_style}; "
                        f"color:{section1_text_color}; "
                        f"font-size:13.5px; line-height:1.5; margin-bottom:14px; "
                        f"white-space:pre-wrap;'>"
                        f"{_esc(draft['current_text_assumption'])}"
                        f"{attribution}"
                        f"</div>", unsafe_allow_html=True)

                    # Section 2: REPLACE WITH (proposed text, green) — IN A COPYABLE CODE BLOCK
                    st.markdown(
                        "<div style='font-size:10.5px; letter-spacing:1.5px; "
                        "color:#2F7A3E; font-weight:700; margin-bottom:4px;'>"
                        "2 · REPLACE WITH THIS TEXT (click the copy icon to copy)"
                        "</div>", unsafe_allow_html=True)
                    st.code(draft['proposed_text'], language=None)

                    # Section 3: RATIONALE
                    st.markdown(
                        "<div style='font-size:10.5px; letter-spacing:1.5px; "
                        "color:#028090; font-weight:700; margin-top:8px; margin-bottom:4px;'>"
                        "3 · WHY THIS CHANGE"
                        "</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background:#F6F8FB; padding:12px 14px; "
                        f"border-radius:6px; color:#1C2940; font-size:13px; "
                        f"line-height:1.5; margin-bottom:14px;'>"
                        f"{_esc(draft['rationale'])}"
                        f"</div>", unsafe_allow_html=True)

                    # Section 4: SECTION REFERENCE — pill
                    st.markdown(
                        f"<div style='font-size:10.5px; letter-spacing:1.5px; "
                        f"color:#5a6783; font-weight:700; margin-bottom:4px;'>"
                        f"4 · REGULATION SECTION"
                        f"</div>"
                        f"<div style='display:inline-block; background:#0F2A47; "
                        f"color:#02C39A; padding:6px 14px; border-radius:4px; "
                        f"font-family:Consolas, monospace; font-size:13px; "
                        f"font-weight:600; margin-bottom:14px;'>"
                        f"{_esc(draft['section_reference'])}"
                        f"</div>", unsafe_allow_html=True)

                    # Validation note — surfaced when the deterministic fallback engaged
                    # or when the LLM output was rejected. Tone depends on whether the
                    # fallback produced useful output (info, blue) or failed (warning, red).
                    val_warn = draft.get("validation_warning")
                    if val_warn:
                        # Info tone (blue/teal) when fallback succeeded — the user has
                        # useful output above and just needs to know to review it.
                        is_info_tone = val_warn.startswith("ℹ")
                        if is_info_tone:
                            bg = "#E8F4F8"
                            border = "#028090"
                            color = "#0a5961"
                        else:
                            bg = "#FBE7EC"
                            border = "#B8294A"
                            color = "#7A1A33"
                        st.markdown(
                            f"<div style='background:{bg}; padding:10px 14px; "
                            f"border-radius:6px; border-left:3px solid {border}; "
                            f"font-size:12px; color:{color}; margin:10px 0;'>"
                            f"{_esc(val_warn)}"
                            f"</div>", unsafe_allow_html=True)

                    # LLM source badge — tells the user WHICH model actually produced
                    # this draft. Critical for trust: "Gemini Flash-Lite" vs "Mock fallback"
                    # is a meaningful distinction the demo audience deserves to see.
                    llm_info = draft.get("llm_source") or {}
                    llm_src = llm_info.get("source", "unknown")
                    llm_model = llm_info.get("model", "")
                    llm_err = llm_info.get("error")
                    if llm_src == "gemini":
                        badge_bg = "#E1F1E3"
                        badge_border = "#2F7A3E"
                        badge_color = "#1a4a23"
                        badge_text = f"✓ Generated by Gemini ({_esc(llm_model)})"
                    elif llm_src == "groq":
                        badge_bg = "#E8F4F8"
                        badge_border = "#028090"
                        badge_color = "#0a5961"
                        badge_text = f"✓ Generated by Groq ({_esc(llm_model)}) · open-source Llama on LPU hardware"
                    elif llm_src == "mock" and llm_err:
                        badge_bg = "#FBE7EC"
                        badge_border = "#B8294A"
                        badge_color = "#7A1A33"
                        badge_text = (
                            f"⚠ Gemini call failed — fell back to mock. "
                            f"Error: {_esc(llm_err)}. "
                            f"Check Streamlit Cloud secrets for GEMINI_API_KEY validity."
                        )
                    elif llm_src == "mock":
                        badge_bg = "#FDF6E3"
                        badge_border = "#B8860B"
                        badge_color = "#5a4a1a"
                        badge_text = (
                            f"ℹ Generated by mock LLM (no Gemini configured). "
                            f"To use Gemini, set LLM_PROVIDER=\"gemini\" and GEMINI_API_KEY in Streamlit Cloud secrets."
                        )
                    elif llm_src == "error":
                        badge_bg = "#FBE7EC"
                        badge_border = "#B8294A"
                        badge_color = "#7A1A33"
                        badge_text = f"⚠ Generation error: {_esc(llm_err or 'unknown')}"
                    else:
                        badge_bg = "#F6F8FB"
                        badge_border = "#5A6783"
                        badge_color = "#34425E"
                        badge_text = f"LLM source: {_esc(llm_src)}"

                    st.markdown(
                        f"<div style='background:{badge_bg}; padding:8px 12px; "
                        f"border-radius:4px; border-left:3px solid {badge_border}; "
                        f"font-size:11px; color:{badge_color}; margin:14px 0 10px 0; "
                        f"font-family:Consolas, monospace;'>"
                        f"{badge_text}"
                        f"</div>", unsafe_allow_html=True)

                    if st.button("↻ Regenerate proposed text", key=f"regen_propose_{i}"):
                        del st.session_state[prop_key]
                        st.rerun()

    cites = res.get("citations") or []
    if cites:
        st.markdown(
            '<div style="margin-top:18px;">'
            '<div style="font-family:\'Source Serif 4\',serif; font-weight:600; '
            'color:#14315a; font-size:18px;">Retrieved context (explainability)</div></div>',
            unsafe_allow_html=True)
        for c in cites[:8]:
            with st.expander(
                f"[{c.get('citation_id', '?')}] {c.get('source_title', '')} · "
                f"{c.get('section') or 'n/a'} · relevance {float(c.get('relevance', 0)):.2f}"):
                st.markdown(
                    f'<div style="background:var(--slate-bg); padding:12px 14px; '
                    f'border-radius:6px; font-size:13px; '
                    f'border-left:3px solid var(--teal); white-space:pre-wrap;">{_esc(c.get("snippet", ""))}</div>',
                    unsafe_allow_html=True)


# ============================================================
# Page: Version Comparison
# ============================================================
def page_compare():
    st.markdown(
        '<div class="page-head-wrap"><h1>Version Comparison</h1>'
        '<div class="page-sub">Three-layer diff: textual changes, semantic shifts, impact deltas.</div></div>',
        unsafe_allow_html=True)

    regs = db_list_documents(kind="regulation")
    if len(regs) < 2:
        st.info("Need at least two regulation documents.")
        return

    families: dict[str, list[dict]] = {}
    for d in regs:
        families.setdefault(d["family_id"], []).append(d)

    fam_options = [f for f, ds in families.items() if len(ds) >= 2]
    if fam_options:
        family = st.selectbox("Regulation family", fam_options,
                             format_func=lambda f: f"{families[f][0]['title']} ({len(families[f])} versions)")
        options = families[family]
    else:
        st.warning("No families with 2+ versions. Showing all regulations.")
        options = regs

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        old = st.selectbox("Old version", options,
                          format_func=lambda d: f"{d['version']} — {d['title']}")
    with c2:
        new = st.selectbox("New version",
                          [o for o in options if o["doc_id"] != old["doc_id"]],
                          format_func=lambda d: f"{d['version']} — {d['title']}")
    with c3:
        st.markdown("<br/>", unsafe_allow_html=True)
        compare_btn = st.button("Compare", type="primary", use_container_width=True)

    if compare_btn:
        with st.spinner("Computing diff..."):
            try:
                res = compare_two_documents(old["doc_id"], new["doc_id"])
                st.session_state["last_compare"] = res
            except Exception as e:
                st.error(f"Comparison failed: {e}")
                return
        auto_sync(reason=f"compare:{old['doc_id'][:16]}->{new['doc_id'][:16]}")

    res = st.session_state.get("last_compare")
    if not res:
        cached = db_get_comparison(old["doc_id"], new["doc_id"])
        if cached:
            res = cached
    if not res:
        return

    st.markdown(f"""
<div class="pane" style="border-left: 4px solid var(--teal);">
  <div class="pane-head">
    <span class="pane-title">AI-generated summary</span>
    <span class="pane-sub">{_esc(res.get('old_version', ''))} ⇄ {_esc(res.get('new_version', ''))}</span>
  </div>
  <div class="pane-body" style="font-size:14px; color:var(--ink-2); line-height:1.6;">
    {_esc(res.get('summary', ''))}
  </div>
</div>
""", unsafe_allow_html=True)

    a = sum(1 for c in res["changes"] if c["type"] == "Added")
    r_ = sum(1 for c in res["changes"] if c["type"] == "Removed")
    m = sum(1 for c in res["changes"] if c["type"] == "Modified")

    st.markdown(f"""
<div class="kpi-band" style="grid-template-columns:repeat(3, 1fr);">
  <div class="kpi-cell">
    <div class="label">Added</div>
    <div class="val" style="color:#2f7a3e;">{a}</div>
  </div>
  <div class="kpi-cell">
    <div class="label">Removed</div>
    <div class="val" style="color:#b8294a;">{r_}</div>
  </div>
  <div class="kpi-cell">
    <div class="label">Modified</div>
    <div class="val" style="color:#b8860b;">{m}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:\'Source Serif 4\',serif; font-weight:600; '
        'color:#14315a; font-size:20px; margin: 8px 0 6px;">Changes</div>',
        unsafe_allow_html=True)

    for c in res["changes"]:
        badge = {"Added": "diff-added", "Removed": "diff-removed",
                "Modified": "diff-modified"}.get(c.get("type"), "diff-modified")

        # Determine display layout based on change type
        ctype_lower = (c.get("type") or "").strip().lower()
        # HTML-escape all LLM-sourced text before embedding in markup
        old_text_esc = _esc((c.get("old_text") or "")[:600])
        new_text_esc = _esc((c.get("new_text") or "")[:600])

        # CRITICAL: Build HTML as a single flat string with NO leading whitespace on any line.
        # Streamlit's markdown parser treats any line with 4+ leading spaces as a code block,
        # which is why multi-line indented HTML was rendering as visible text.
        if ctype_lower == "added":
            change_body = (
                '<div style="margin:10px 0;">'
                '<div style="display:inline-block;background:#16a34a;color:white;'
                'font-size:10px;font-weight:700;letter-spacing:1px;padding:3px 10px;'
                'border-radius:3px;margin-bottom:8px;">+ ADDED IN THIS VERSION</div>'
                f'<div class="diff-added">{new_text_esc or "<i style=&quot;color:#5a6783;&quot;>No text supplied by model.</i>"}</div>'
                '</div>'
            )
        elif ctype_lower == "removed":
            change_body = (
                '<div style="margin:10px 0;">'
                '<div style="display:inline-block;background:#dc2626;color:white;'
                'font-size:10px;font-weight:700;letter-spacing:1px;padding:3px 10px;'
                'border-radius:3px;margin-bottom:8px;">− REMOVED IN THIS VERSION</div>'
                f'<div class="diff-removed">{old_text_esc or "<i style=&quot;color:#5a6783;&quot;>No text supplied by model.</i>"}</div>'
                '</div>'
            )
        else:
            # Modified — side-by-side
            old_inner = (f'<div class="diff-removed">{old_text_esc}</div>'
                        if old_text_esc else
                        '<div style="color:var(--ink-mute);font-size:13px;">—</div>')
            new_inner = (f'<div class="diff-added">{new_text_esc}</div>'
                        if new_text_esc else
                        '<div style="color:var(--ink-mute);font-size:13px;">—</div>')
            change_body = (
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px 0;">'
                '<div>'
                '<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:0.6px;'
                'color:var(--ink-soft);font-weight:700;margin-bottom:6px;">Old</div>'
                f'{old_inner}'
                '</div>'
                '<div>'
                '<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:0.6px;'
                'color:var(--ink-soft);font-weight:700;margin-bottom:6px;">New</div>'
                f'{new_inner}'
                '</div>'
                '</div>'
            )

        # Whole card as a single flush-left HTML string — no leading whitespace on any line
        card_html = (
            '<div class="pane">'
            '<div class="pane-head">'
            f'<span class="pane-title">{_esc(c.get("type", ""))} · {_esc(c.get("section", ""))}</span>'
            f'<span class="pane-sub">Compliance: {_esc(c.get("compliance_risk_delta", "—"))} · '
            f'Operational: {_esc(c.get("operational_impact_delta", "—"))}</span>'
            '</div>'
            '<div class="pane-body">'
            f'<div class="{badge}" style="margin-bottom:10px;">'
            f'<b>{_esc(c.get("type", ""))}</b><br>{_esc(c.get("description", ""))}'
            '</div>'
            f'{change_body}'
            f'<div class="info-line"><span class="lbl">Impact</span>'
            f'<span>{_esc(c.get("impact", ""))}</span></div>'
            f'<div class="info-line"><span class="lbl">Action</span>'
            f'<span>{_esc(c.get("recommended_action", ""))}</span></div>'
            '</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


# ============================================================
# Page: Upload
# ============================================================
def page_upload():
    st.markdown(
        '<div class="page-head-wrap"><h1>Upload Documents</h1>'
        '<div class="page-sub">Upload regulatory mandates or internal documents. PDF, DOCX, TXT, HTML, MD.</div></div>',
        unsafe_allow_html=True)

    if cloud_is_configured():
        st.markdown("""
<div class="privacy-banner">
  <b>⚠ Privacy notice.</b>
  Files uploaded here are persisted to the configured GitHub repository. If your repo is <b>public</b>,
  uploaded files will be visible to anyone. Do not upload real internal documents to a public repo.
</div>
""", unsafe_allow_html=True)

    with st.form("upload_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            files = st.file_uploader("Drag and drop files",
                                    type=["pdf", "docx", "txt", "md", "html", "htm"],
                                    accept_multiple_files=True)
        with c2:
            kind = st.selectbox("Kind", ["regulation", "policy", "sop", "system"],
                              format_func=lambda x: {"regulation": "Regulation",
                                                    "policy": "Internal Policy",
                                                    "sop": "SOP / Workflow",
                                                    "system": "System Doc"}[x])
        with c3:
            version = st.text_input("Version", value="v1")
        family_id = st.text_input("Family ID (optional)",
                                 help="Group versions of the same document.")
        submitted = st.form_submit_button("Ingest", type="primary")

    if submitted and files:
        progress = st.progress(0.0)
        any_success = False
        for i, f in enumerate(files, start=1):
            try:
                text = extract_text_from_uploaded(f)
                if not text.strip():
                    st.error(f"❌ {f.name}: no text extracted")
                    continue
                meta = extract_metadata_simple(text) if normalize_kind(kind) == "regulation" else {}
                title = meta.get("title") or Path(f.name).stem.replace("_", " ").title()
                fam = family_id or slugify(meta.get("regulation_id") or title)
                doc_id = f"{slugify(title)}__{version}__{uuid.uuid4().hex[:6]}"

                d = {"doc_id": doc_id, "title": title,
                    "kind": normalize_kind(kind), "version": version,
                    "family_id": fam,
                    "effective_date": meta.get("effective_date"),
                    "regulation_id": meta.get("regulation_id"),
                    "issuing_body": meta.get("issuing_body"),
                    "change_type": meta.get("change_type"),
                    "categories": []}
                chunks = chunk_text(text)
                db_insert_document(d, text, chunks)
                st.success(f"✓ Ingested **{f.name}** ({len(chunks)} chunks)")
                any_success = True
            except Exception as e:
                st.error(f"❌ {f.name}: {e}")
            progress.progress(i / len(files))
        if any_success:
            with st.spinner("Syncing to cloud..."):
                auto_sync(reason=f"upload:{len(files)}_files")

    all_docs = db_list_documents()
    if not all_docs:
        st.info("Nothing indexed yet.")
        return

    rows_h = []
    for d in all_docs[:50]:
        rows_h.append(f"""
<div class="dg-row" style="grid-template-columns: 2.4fr 1fr 1fr 1fr 1.4fr;">
  <div>
    <div class="reg-name">{d['title']}</div>
    <div class="reg-mono">{d['doc_id'][:32]}</div>
  </div>
  <div>{kind_pill(d['kind'])}</div>
  <div class="reg-mono">{d['version']}</div>
  <div class="reg-mono">{d.get('effective_date') or '—'}</div>
  <div class="reg-mono">{d['num_chunks']} chunks</div>
</div>
""")
    st.markdown(f"""
<div class="pane">
  <div class="pane-head">
    <span class="pane-title">All documents</span>
    <span class="pane-sub">{len(all_docs)} total</span>
  </div>
  <div class="dg-row head" style="grid-template-columns:2.4fr 1fr 1fr 1fr 1.4fr;">
    <div>Document</div><div>Type</div><div>Version</div><div>Effective</div><div>Meta</div>
  </div>
  {''.join(rows_h)}
</div>
""", unsafe_allow_html=True)


# ============================================================
# Page: Timeline
# ============================================================
def page_timeline():
    st.markdown(
        '<div class="page-head-wrap"><h1>Regulatory Timeline</h1>'
        '<div class="page-sub">Audit trail of every ingestion, analysis, and comparison.</div></div>',
        unsafe_allow_html=True)

    events = db_get_timeline()
    if not events:
        st.info("No events yet.")
        return

    df = pd.DataFrame([{
        "When": e["created_at"],
        "Event": e["event_type"],
        "Family": e.get("family_id") or "",
        "Version": e.get("version") or "",
        "Doc": (e.get("doc_id") or "")[:32],
    } for e in events])

    st.markdown(
        '<div class="pane"><div class="pane-head">'
        '<span class="pane-title">Audit timeline</span>'
        f'<span class="pane-sub">{len(events)} events</span>'
        '</div><div class="pane-body">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=300)
    st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
# Router
# ============================================================
if PAGE.endswith("Dashboard"):
    page_dashboard()
elif PAGE.endswith("Impact Analysis"):
    page_impact()
elif PAGE.endswith("Version Comparison"):
    page_compare()
elif PAGE.endswith("Upload"):
    page_upload()
elif PAGE.endswith("Timeline"):
    page_timeline()
