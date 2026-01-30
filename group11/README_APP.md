# Book Recommendation UI

Web UI for the book recommendation engine (content-based + collaborative filtering). No hardcoded users or preferences.

## Run locally

```bash
cd group11
pip install -r requirements.txt
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

## Host on your portfolio

- **Streamlit Community Cloud** (free): Push this repo to GitHub, then at [share.streamlit.io](https://share.streamlit.io) deploy with:
  - Repo: `your-username/Book-Recommendation`
  - Branch: `main`
  - Main file: `group11/src/app.py`
  - App directory: leave empty or `group11` (so working directory has `group11/Dataset/`).

- **Other hosts**: Run `streamlit run src/app.py` with working directory `group11` so `src/app.py` and `Dataset/` are found.

## Modes

1. **By my preferences** – Choose favorite authors and books; get content-based recommendations.
2. **By a user in the dataset** – Pick a demo user (with 5+ ratings); get hybrid (collaborative + content-based) recommendations.
