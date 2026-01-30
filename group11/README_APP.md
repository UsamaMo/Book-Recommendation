# Book Recommendation UI

Web UI for the book recommendation engine (content-based + collaborative filtering). No hardcoded users or preferences.

## Run locally

```bash
cd group11
pip install -r requirements.txt
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

## Host on Streamlit Cloud

1. Push the repo to GitHub (ensure `group11/Dataset/` with Books.csv, Ratings.csv, Users.csv is committed).
2. At [share.streamlit.io](https://share.streamlit.io): **New app** → connect repo → set:
   - **Main file path:** `group11/src/app.py`
   - **Advanced settings → App directory:** `group11`
3. Deploy. The app uses a sample of ratings on Streamlit Cloud to stay within memory limits.
4. If the app shows "Dataset not found", open the **Debug** expander to see paths and cwd, and confirm the Dataset folder is in the repo and App directory is `group11`.

## Modes

1. **By my preferences** – Choose favorite authors and books; get content-based recommendations.
2. **By a user in the dataset** – Pick a demo user (with 5+ ratings); get hybrid (collaborative + content-based) recommendations.
