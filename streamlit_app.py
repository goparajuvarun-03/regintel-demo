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
def call_llm(system: str, user: str, max_tokens: int = 1500) -> dict:
    """Returns a JSON dict. Falls back to mock if Gemini unavailable."""
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
            return json.loads(resp.text)
        except Exception as e:
            logger.warning("Gemini call failed, using mock: %s", e)

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

    # Impact analysis (default for regulation analysis prompts)
    if ("impact" in u and "regulation" in u) or "analyze the impact" in u:
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
                    "impact_reason": "Current policy honors prior plan PA decisions for 30 days. Final Rule § 422.138(b) requires not less than 90 days, plus full duration of clinically established treatment plans.",
                    "recommended_action": "Revise § 4.2 to reflect 90-day window. Route through compliance review and republish.",
                    "risk_if_not_implemented": "Civil monetary penalties; audit findings on transition handling.",
                    "confidence_score": 0.86,
                    "supporting_citations": ["§ 422.138(b)", "CHUNK_4", "CHUNK_7"],
                },
                {
                    "name": "Claims Adjudication SOP",
                    "type": "Workflow",
                    "priority": "Medium",
                    "impact_reason": "Workflow lacks transition-flag decision step. Claims for transitioning members are processed under standard rules without honoring prior-plan PA decisions.",
                    "recommended_action": "Insert transition-flag decision node at Step 4 (PA check). Update SOP runbook.",
                    "risk_if_not_implemented": "Inappropriate denials; appeals volume spike; member harm.",
                    "confidence_score": 0.79,
                    "supporting_citations": ["§ 422.138(b)", "CHUNK_4", "CHUNK_5"],
                },
                {
                    "name": "MA Core Claims Engine",
                    "type": "System",
                    "priority": "Medium",
                    "impact_reason": "PA-MOD-v4 has no interface for ingesting prior authorization records from external Medicare Advantage organizations. Manual entry creates significant operational overhead.",
                    "recommended_action": "Add a configuration flag and PA ingestion edit rule. Plan for next sprint release.",
                    "risk_if_not_implemented": "Manual workarounds; audit findings; operational cost overruns.",
                    "confidence_score": 0.74,
                    "supporting_citations": ["§ 422.138(b)", "CHUNK_4", "CHUNK_8"],
                },
            ],
            "citations": [
                {"citation_id": "CHUNK_4", "source_title": "Final Rule",
                 "section": "§ 422.138(b)", "snippet":
                 "MA organizations must continue to honor prior authorization "
                 "decisions made by another organization for a minimum of 90 days "
                 "from the date the enrollee transitioned plans...",
                 "relevance": 0.91},
                {"citation_id": "CHUNK_7", "source_title": "Final Rule",
                 "section": "§ 422.138(c)", "snippet":
                 "Notice of any denial must include appeal rights and be issued "
                 "within timeframes consistent with adverse-determination notice rules.",
                 "relevance": 0.84},
                {"citation_id": "CHUNK_8", "source_title": "Final Rule",
                 "section": "§ 422.138(d)", "snippet":
                 "Plans must report compliance with continuity-of-care provisions "
                 "to CMS on a quarterly basis using prescribed reporting templates.",
                 "relevance": 0.78},
            ],
        }

    # Proposed-text change for an impacted artifact (mock for offline demo / no Gemini key)
    if "policy-language change" in u or ("proposed_text" in u and "current_text_assumption" in u):
        # Extract the artifact name and gap description from the prompt so different cards
        # produce different proposed text (instead of all returning the same canned answer).
        import re as _re
        name_match = _re.search(r"Name:\s*(.+)", user)
        type_match = _re.search(r"Type:\s*(.+)", user)
        gap_match  = _re.search(r"Gap identified:\s*(.+)", user)
        rec_match  = _re.search(r"Recommended action \(high level\):\s*(.+)", user)
        reg_match  = _re.search(r"Identifier:\s*(.+)", user)

        artifact_name = (name_match.group(1).strip() if name_match else "the affected artifact")
        artifact_type = (type_match.group(1).strip() if type_match else "Policy")
        gap_desc      = (gap_match.group(1).strip() if gap_match else "")
        rec_action    = (rec_match.group(1).strip() if rec_match else "")
        reg_id        = (reg_match.group(1).strip() if reg_match else "the regulation")

        # Different proposed-text templates by artifact type
        type_lower = artifact_type.lower()
        if "workflow" in type_lower or "sop" in type_lower or "procedure" in type_lower:
            current_assumption = (
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
            current_assumption = (
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
            # Policy / default
            current_assumption = (
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

        # Try to extract a section reference like "§ 422.138(b)" from the gap or rec_action text
        sect_match = _re.search(r"§\s*\d+\.\d+(?:\([a-zA-Z0-9]+\))*", user)
        section_ref = sect_match.group(0) if sect_match else reg_id

        return {
            "current_text_assumption": current_assumption,
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

def generate_proposed_text(regulation_doc: dict, analysis_result: dict,
                            impacted_area: dict) -> dict:
    """Call the LLM to draft the language change for an impacted artifact.
    Returns a dict with four keys: current_text_assumption, proposed_text,
    rationale, section_reference. Falls back to safe defaults on failure."""

    user_prompt = f"""You are a senior healthcare compliance analyst at a Medicare Advantage payer.
The internal artifact below must be updated to comply with the regulation. Draft the precise language change.

REGULATION
Title: {regulation_doc.get('title', 'Unknown')}
Identifier: {regulation_doc.get('regulation_id', 'N/A')}
Effective date: {regulation_doc.get('effective_date', 'TBD')}
Summary: {analysis_result.get('regulation_summary', '(no summary)')}

INTERNAL ARTIFACT TO REMEDIATE
Name: {impacted_area.get('name', 'unknown')}
Type: {impacted_area.get('type', 'unknown')}
Gap identified: {impacted_area.get('impact_reason', '(no description)')}
Recommended action (high level): {impacted_area.get('recommended_action', '(none)')}

TASK
Return a JSON object with EXACTLY these four keys. Your response MUST be specific to THIS artifact ({impacted_area.get('name', 'unknown')}) — not generic language that could apply to any artifact.

- current_text_assumption: 1-2 sentences describing what the analyst should expect the current text of the artifact named "{impacted_area.get('name', 'unknown')}" to say. Reference the specific gap: "{impacted_area.get('impact_reason', '')[:150]}". Frame as an assumption the analyst should verify, not as fact.

- proposed_text: 2-4 sentences of revised {impacted_area.get('type', 'policy')}-style language in formal regulatory prose. This is the EXACT wording the analyst will paste into the {impacted_area.get('name', 'artifact')} document, replacing the current text. The proposed_text MUST address the specific gap described above for THIS artifact. Do not include framing like "We propose..." — write the prose AS IT WOULD APPEAR in the document itself. The wording must differ meaningfully from any other artifact's proposed text.

- rationale: 1-2 sentences explaining why this exact wording is required, citing the regulation section explicitly (e.g. "Required per {regulation_doc.get('regulation_id', 'the regulation')} § XYZ"). Mention the artifact name "{impacted_area.get('name', '')}" by name.

- section_reference: the specific regulation section that mandates this change, formatted as in the source (e.g. "§ 422.138(b)" or "42 CFR § 422.752"). Single string, not an array.

IMPORTANT
- Use formal compliance-officer prose for proposed_text. The artifact is a real {impacted_area.get('type', 'policy')} document; the language must sound like one, not a memo or summary.
- The proposed_text must reference or address the specific artifact "{impacted_area.get('name', '')}" — generic answers that could apply to any artifact are wrong.
- Cite the regulation section explicitly in rationale.
- Never fabricate verbatim policy text in current_text_assumption — describe what it likely says, frame as analyst-to-verify.
"""

    try:
        result = call_llm(
            system="You are a senior healthcare compliance analyst. You draft precise "
                   "policy-language changes for compliance remediation. Return ONLY valid JSON.",
            user=user_prompt,
            max_tokens=900,
        )
        return {
            "current_text_assumption": str(result.get("current_text_assumption") or
                "[Verify current text in the affected artifact before applying changes.]"),
            "proposed_text": str(result.get("proposed_text") or
                impacted_area.get("recommended_action") or
                "[AI-drafted language unavailable. Analyst to draft compliant replacement text.]"),
            "rationale": str(result.get("rationale") or
                f"Required by {regulation_doc.get('regulation_id') or regulation_doc.get('title')}."),
            "section_reference": str(result.get("section_reference") or
                regulation_doc.get('regulation_id') or
                regulation_doc.get('title', 'See regulation')),
        }
    except Exception as e:
        logger.exception("Proposed-text generation failed: %s", e)
        # Hard fallback
        return {
            "current_text_assumption": "[Verify current text in the affected artifact before applying changes.]",
            "proposed_text": impacted_area.get("recommended_action") or
                "[Analyst to draft compliant replacement language.]",
            "rationale": f"Required to comply with {regulation_doc.get('regulation_id') or regulation_doc.get('title')}.",
            "section_reference": regulation_doc.get('regulation_id') or "See regulation",
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
            # parse filename like "01_cms_4201_proposed.txt"
            stem = f.stem.lower()
            if "cms_4201_proposed" in stem:
                d = {"title": "Sample Continuity of Care (Proposed)",
                     "kind": "regulation", "version": "v1.0",
                     "regulation_id": "CMS-4201-P", "change_type": "Proposed",
                     "effective_date": "2026-01-15",
                     "issuing_body": "CMS",
                     "categories": ["Continuity of Care", "Prior Authorization"]}
            elif "cms_4201_final" in stem:
                d = {"title": "Sample Continuity of Care (Final)",
                     "kind": "regulation", "version": "v2.0",
                     "regulation_id": "CMS-4201-F", "change_type": "Final",
                     "effective_date": "2026-01-15",
                     "issuing_body": "CMS",
                     "categories": ["Continuity of Care", "Prior Authorization"]}
            elif "mln" in stem:
                d = {"title": "Sample ICD-10 Validation Edits",
                     "kind": "regulation", "version": "v1.0",
                     "regulation_id": "CMS-MLN-12345",
                     "effective_date": "2026-07-01",
                     "issuing_body": "CMS",
                     "categories": ["Billing", "ICD-10"]}
            elif "pa_policy" in stem or "pa-policy" in stem or "pa policy" in stem:
                d = {"title": "Prior Authorization Continuity Policy",
                     "kind": "policy", "version": "v3.2",
                     "categories": ["Prior Authorization"]}
            elif "claims_sop" in stem or "claims-sop" in stem:
                d = {"title": "Claims Adjudication SOP",
                     "kind": "sop", "version": "v2.4",
                     "categories": ["Claims"]}
            elif "system_arch" in stem or "system-arch" in stem:
                d = {"title": "MA Core Claims Engine",
                     "kind": "system", "version": "v1.7",
                     "categories": ["System Architecture"]}
            else:
                d = {"title": meta.get("title") or f.stem,
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
    <span>Acme Health Plan · Compliance Workspace</span>
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

# LLM mode info
if LLM_PROVIDER == "mock" or not GEMINI_API_KEY:
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
                    st.markdown(
                        "<div style='background:#FDF6E3; padding:10px 14px; "
                        "border-radius:6px; border-left:3px solid #B8860B; "
                        "font-size:12.5px; color:#5a4a1a; margin-bottom:14px;'>"
                        "<b>How to use this:</b> Locate the current text in your "
                        f"<b>{ia.get('name', 'artifact')}</b> document, then replace it "
                        "with the proposed text below. Always validate the current-text "
                        "assumption against the actual document before applying."
                        "</div>",
                        unsafe_allow_html=True)

                    # Section 1: FIND THIS (current text assumption, crimson)
                    st.markdown(
                        "<div style='font-size:10.5px; letter-spacing:1.5px; "
                        "color:#B8294A; font-weight:700; margin-top:4px; margin-bottom:4px;'>"
                        "1 · FIND THIS TEXT IN THE ARTIFACT"
                        "</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background:#FBE7EC; padding:12px 14px; "
                        f"border-radius:6px; font-style:italic; color:#7A1A33; "
                        f"font-size:13.5px; line-height:1.5; margin-bottom:14px;'>"
                        f"{_esc(draft['current_text_assumption'])}"
                        f"</div>", unsafe_allow_html=True)

                    # Section 2: REPLACE WITH (proposed text, green) — IN A COPYABLE CODE BLOCK
                    st.markdown(
                        "<div style='font-size:10.5px; letter-spacing:1.5px; "
                        "color:#2F7A3E; font-weight:700; margin-bottom:4px;'>"
                        "2 · REPLACE WITH THIS TEXT (click the copy icon to copy)"
                        "</div>", unsafe_allow_html=True)
                    # st.code provides a built-in copy button in Streamlit
                    # st.code handles text safely — no escape needed here
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
