# RegIntel-AI

**Healthcare Regulatory Intelligence — single-file Streamlit application.**

A demo-ready RAG application that ingests regulatory documents (CMS rules, MLN
articles), maps them against internal payer artifacts (policies, SOPs, system
docs), and produces cited impact analyses with what-if simulations and version
comparisons.

## What's in this package

```
regintel-v2/
├── streamlit_app.py         # Entire application — single file, ~2100 lines
├── requirements.txt         # 9 dependencies, no torch, no chromadb
├── .streamlit/
│   ├── config.toml          # Streamlit theme
│   └── secrets.toml.example # Template for API keys (don't commit secrets.toml)
├── seed/                    # 6 synthetic sample documents
│   ├── 01_cms_4201_proposed.txt
│   ├── 02_cms_4201_final.txt
│   ├── 03_cms_mln_billing.txt
│   ├── 04_pa_policy.txt
│   ├── 05_claims_sop.txt
│   └── 06_system_arch.txt
└── .gitignore
```

## Architecture

- **Storage:** SQLite (`./data/regintel.db`) — metadata, analyses, comparisons, audit timeline
- **Retrieval:** TF-IDF over chunks (scikit-learn). No embeddings model required.
- **LLM:** Google Gemini 2.5 Flash-Lite (paid Tier-1 recommended). Mock fallback if no key.
- **Persistence:** Optional GitHub-backed snapshots for cross-machine demo.
- **UI:** Direction C aesthetic — navy + teal palette, Source Serif 4 / Source Sans 3 typography.

## Quick start (local, 10 minutes)

```powershell
# Windows Command Prompt
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Copy secrets template
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# (edit to your liking — defaults to mock LLM mode)

streamlit run streamlit_app.py
```

Browser opens at `http://localhost:8501`. First load takes ~10 seconds.

## Deploy to Streamlit Cloud (free tier)

See `DEPLOYMENT.md` for the complete walkthrough.

## What's deliberately omitted

This is a demo build, not a production system. Trade-offs:

- **No HIPAA compliance.** Public-data and synthetic-data only.
- **No authentication.** Anyone with the URL has full access.
- **No vector database.** TF-IDF works well at demo scale (≤ 1000 documents).
- **No fine-tuning.** Default Gemini Flash-Lite covers the use case.
- **Single-region, single-tenant.** No HA, no multi-region failover.

These can all be added later when usage justifies the cost — see
`RegIntel-AI_Decision_Matrix.pdf` for the upgrade path.
