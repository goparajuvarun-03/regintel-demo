# Deploying RegIntel-AI to Streamlit Cloud

This is the complete deployment guide for the rebuilt single-file version. The
app runs in mock LLM mode out of the box, so you can deploy first, then add a
real Gemini API key when you're ready.

## Total time: about 60 minutes (most of it waiting)

---

## ⏱️ Phase 1 — Test locally first (10 min)

Before deploying anywhere, confirm the app runs on your laptop.

### Windows Command Prompt:
```
cd C:\regintel-v2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
streamlit run streamlit_app.py
```

### Mac / Linux Terminal:
```
cd ~/regintel-v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run streamlit_app.py
```

**What you should see:**
- Browser opens to `http://localhost:8501`
- Dashboard loads in ~10 seconds with 6 sample documents
- Sidebar shows ⚪ Local-only (correct for first run without GitHub configured)

If this fails — **stop and read the error** before deploying. Common issues:

| Error | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python from python.org with "Add to PATH" ticked |
| `No module named 'streamlit'` | You forgot to activate the venv. Re-run the activate command. |
| Sklearn import error | Run `pip install --upgrade scikit-learn` |

---

## ⏱️ Phase 2 — Set up accounts (15 min)

### GitHub
1. Go to https://github.com/signup
2. Sign up with your work email
3. Verify via the email link

### Streamlit Community Cloud
1. Go to https://streamlit.io/cloud
2. Click **Sign up**
3. Choose **Continue with GitHub** — links the two accounts

### Google AI Studio (for Gemini, optional)
1. Go to https://aistudio.google.com/apikey
2. Sign in with any Google account
3. Click **Create API key** → copy the key (starts with `AIza...`)
4. Save it in Notepad temporarily

> You can skip this step and deploy in mock mode first, then add the key later.

---

## ⏱️ Phase 3 — Generate GitHub Personal Access Token (5 min)

Only needed if you want **cross-machine persistence** (uploads visible from any
device). Skip if you only need the app to work — uploads will still work on the
deployed instance, they just won't persist across redeploys.

1. Go to https://github.com/settings/tokens?type=beta
2. Click **Generate new token**
3. Set:
   - **Token name:** `regintel-snapshots`
   - **Expiration:** 90 days
   - **Repository access:** All repositories *(we'll lock this down later)*
   - **Repository permissions:** Set **Contents** → **Read and write**
4. Click **Generate token**
5. **Copy the token immediately** — starts with `github_pat_`. You won't see it again.
6. Paste it into Notepad next to your Gemini key.

---

## ⏱️ Phase 4 — Create the GitHub repo (10 min)

### 4a. Create repo
1. Go to https://github.com/new
2. **Repository name:** `regintel-ai`
3. **Public** (required for Streamlit Cloud free tier)
4. ✅ Tick "Add a README file"
5. Click **Create repository**

### 4b. Lock down GitHub token (if you generated one in Phase 3)
1. Go back to https://github.com/settings/tokens?type=beta
2. Click your `regintel-snapshots` token
3. **Repository access** → **Only select repositories** → pick `regintel-ai`
4. **Update**

### 4c. Upload code
1. Unzip your `regintel-v2` package somewhere easy
2. On GitHub repo page, click **Add file** → **Upload files**
3. **Select ALL files** inside the `regintel-v2` folder (not the folder itself):
   - `streamlit_app.py`
   - `requirements.txt`
   - `README.md`
   - `DEPLOYMENT.md`
   - `.gitignore` (hidden — see below)
   - The `seed/` folder
   - The `.streamlit/` folder (hidden — see below)
4. Drag them all into the upload area
5. **Important — make sure these aren't there:**
   - ❌ `.streamlit/secrets.toml` — gitignored, but double-check
   - ❌ `data/` folder — gitignored
6. Commit message: *Initial commit*
7. Click **Commit changes**

> ⚠️ **Hidden files alert:** `.gitignore` and `.streamlit` start with a dot and may be hidden by your OS. To show them:
> - **Windows:** File Explorer → View → tick "Hidden items"
> - **Mac:** Finder → Cmd+Shift+. (toggle hidden items)

> ✅ **Verify:** Refresh the repo page. You should see `streamlit_app.py` listed at the top level.

---

## ⏱️ Phase 5 — Deploy to Streamlit Cloud (10 min)

1. Go to https://share.streamlit.io
2. Click **Create app** (top right)
3. Choose **Deploy a public app from GitHub**
4. Fill in:
   - **Repository:** `yourusername/regintel-ai`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** `regintel-ai-yourname` (whatever subdomain you want)
5. **DON'T click Deploy yet** — click **Advanced settings** first (see Phase 6)

---

## ⏱️ Phase 6 — Add secrets (5 min)

In the **Advanced settings** panel that just opened, find the **Secrets** box.

### For mock mode (deploy first, no API costs):
```toml
LLM_PROVIDER = "mock"
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-2.5-flash-lite"
GITHUB_TOKEN = ""
GITHUB_REPO = ""
```

### For Gemini mode (better demo quality):
```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "AIza...your-real-key..."
GEMINI_MODEL = "gemini-2.5-flash-lite"
GITHUB_TOKEN = ""
GITHUB_REPO = ""
```

### For Gemini + cross-machine persistence (full demo setup):
```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "AIza...your-real-key..."
GEMINI_MODEL = "gemini-2.5-flash-lite"
GITHUB_TOKEN = "github_pat_...your-real-token..."
GITHUB_REPO = "yourusername/regintel-ai"
```

Click **Save**.

Now click **Deploy!**

---

## ⏱️ Phase 7 — First deploy (10–15 min)

Streamlit Cloud will:
1. Clone your repo
2. Install dependencies (~3 min — faster than before, no torch!)
3. Start the app
4. Run bootstrap → load 6 sample documents

When you see "You can now view your Streamlit app", refresh and you should land
on the dashboard with all 6 sample documents.

> ⚠️ If you see a red error: click **Manage app** → **Logs** → scroll to the
> **first** red error (not the last). Copy that error and we can debug it.

---

## ⏱️ Phase 8 — Verify (5 min)

Click through:
- ☐ **Dashboard** — 5 KPI tiles, 3 regulations table, 3 internal artifacts table, risk donut, upcoming dates, DEMO MODE footer
- ☐ **Impact Analysis** → pick "Sample Continuity of Care (Final)" → click **Run analysis** → wait → impact gauge + 3 cards appear
- ☐ **Version Comparison** → Compare button → AI summary callout + diff cards appear
- ☐ **Upload** — page renders, privacy banner if cloud sync is on
- ☐ **Timeline** — events table populated

---

## ⏱️ Phase 9 — Cross-machine test (only if cloud sync configured)

1. On your laptop, upload one small test PDF on the Upload page
2. Click **Sync now** in the sidebar
3. Open the same URL on your phone
4. **You should see the test PDF.** If yes — cross-machine demo works.

---

## What to do the night before your demo

1. **Pre-warm the app** — open the URL once. Streamlit puts apps to sleep
   after 12 hours of idle.
2. **Run a full dress rehearsal** — go through Dashboard → Impact Analysis →
   Version Comparison end-to-end.
3. **Take screenshots** as a fallback in case the app fails on demo day.

## What to do 5 minutes before the demo

1. Open the app URL on the demo machine
2. Wait for dashboard to load
3. Run **one** impact analysis to cache it
4. Run **one** version comparison to cache it
5. Return to dashboard
6. **Don't close the tab**

When you start the demo, everything is warm and instant.

---

## Quick troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Sidebar shows ⚪ Local-only when GH is configured | Wrong `GITHUB_REPO` format | Must be `owner/repo`, no `.git`, no `https://` |
| First page load hangs > 60s | Streamlit Cloud cold start | Wait, then refresh. Normal once. |
| "Mock LLM mode" message | LLM_PROVIDER not set or no Gemini key | Check secrets in Manage app |
| Red error on first deploy | Missing file in repo | Check `streamlit_app.py` is at repo root |
| Upload fails silently | PDF library issue | Try a TXT or MD file instead |

If stuck, paste the **first** red error line from Streamlit Cloud Logs and I'll diagnose it.
