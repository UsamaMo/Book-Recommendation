"""
Book Recommendation app - runs main.ipynb logic with Streamlit UI.
Minimal changes from the notebook; only pivot column fix and Streamlit inputs/output.
Heavy imports are done in main() so missing deps show an error instead of a blank screen.
"""
import os
import streamlit as st

# ---- Paths: resolve relative to this file so it works locally and on Streamlit Cloud ----
BASE = os.path.dirname(os.path.abspath(__file__))
# Primary: Dataset is group11/Dataset when this file is group11/src/app.py
DATASET = os.path.normpath(os.path.join(BASE, "..", "Dataset"))
# Fallbacks when Streamlit runs from repo root with different cwd
if not os.path.isfile(os.path.join(DATASET, "Books.csv")):
    for candidate in [os.path.join(os.getcwd(), "group11", "Dataset"), os.path.join(os.getcwd(), "Dataset")]:
        if os.path.isfile(os.path.join(candidate, "Books.csv")):
            DATASET = os.path.normpath(candidate)
            break
BOOKS_PATH = os.path.join(DATASET, "Books.csv")
RATINGS_PATH = os.path.join(DATASET, "Ratings.csv")
USERS_PATH = os.path.join(DATASET, "Users.csv")


@st.cache_data
def load_and_preprocess():
    """Notebook cells 2–4: load and preprocess."""
    if not os.path.isfile(BOOKS_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {BOOKS_PATH}. "
            "On Streamlit Cloud: commit the group11/Dataset folder and set 'App directory' to group11."
        )
    books = pd.read_csv(BOOKS_PATH, dtype={"Year-Of-Publication": object})
    # On Streamlit Cloud (limited RAM), load a sample of ratings so the app doesn't OOM
    ratings_max_rows = 80_000 if os.environ.get("STREAMLIT_SERVER_HEADLESS") else None
    ratings = pd.read_csv(RATINGS_PATH, nrows=ratings_max_rows)
    users = pd.read_csv(USERS_PATH)

    books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce").fillna(0).astype(int)
    books.drop(["Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1, inplace=True)

    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")

    users["Age"] = pd.to_numeric(users["Age"], errors="coerce").fillna(users["Age"].median())
    users["Age"] = users["Age"].clip(lower=10, upper=100).astype(int)

    deduped_ratings = ratings.drop_duplicates(["User-ID", "ISBN"])
    train, test = train_test_split(deduped_ratings, test_size=0.2, random_state=42)

    # Use a sample so pivot in collab_recommendations stays feasible (notebook logic unchanged)
    n_sample = min(15_000, len(deduped_ratings))
    ratings_for_rec = deduped_ratings.sample(n=n_sample, random_state=42) if len(deduped_ratings) > n_sample else deduped_ratings

    return books, ratings, users, deduped_ratings, ratings_for_rec, train, test


# ---- Notebook: pivot_ratings (only fix: CSV has 'Book-Rating' not 'Rating') ----
def pivot_ratings(ratings):
    return ratings.pivot(index="User-ID", columns="ISBN", values="Book-Rating").fillna(0)


# ---- Notebook: book_similarity ----
def book_similarity(title1, title2):
    return SequenceMatcher(None, title1, title2).ratio()


# ---- Notebook: content_based_recommendations ----
def content_based_recommendations(user_preferences, books, n_recs=5):
    favorite_authors = user_preferences.get("favorite_authors", [])
    auth_books = books[books["Book-Author"].isin(favorite_authors)]

    favorite_books = user_preferences.get("favorite_books", [])
    similar_books = []
    for book in favorite_books:
        similarities = books.apply(lambda x: book_similarity(x["Book-Title"], book), axis=1)
        similar_book = books.loc[similarities.idxmax()]
        similar_books.append(similar_book)

    recs = pd.concat([auth_books, pd.DataFrame(similar_books)], ignore_index=True)
    return recs[:n_recs]


# ---- Notebook: collab_recommendations (only fix: books index is not ISBN, so filter by ISBN) ----
def collab_recommendations(user_id, ratings, books, n_recs=5):
    user_ratings = ratings[ratings["User-ID"] == user_id]
    other_ratings = ratings[ratings["User-ID"] != user_id]
    if user_ratings.empty or other_ratings.empty:
        return books.head(0)
    user_book_matrix = pivot_ratings(user_ratings)
    other_book_matrix = pivot_ratings(other_ratings)
    similarities = cosine_similarity(user_book_matrix, other_book_matrix)

    similar_users = np.argsort(similarities)[-1:-6:-1]

    top_books = {}
    for user in similar_users:
        other_user_books = other_book_matrix.iloc[user]
        for i, rating in other_user_books.iteritems():
            if i not in user_book_matrix.columns:
                if i not in top_books or top_books[i] < rating:
                    top_books[i] = rating

    isbns = list(top_books.keys())[:n_recs]
    return books[books["ISBN"].isin(isbns)].head(n_recs)


# ---- Notebook: hybrid_recommendations ----
def hybrid_recommendations(user_id, user_prefs, ratings, books, n=5):
    user_ratings = ratings[ratings["User-ID"] == user_id]

    if len(user_ratings) >= 10:
        cf_recs = collab_recommendations(user_id, ratings, books, n // 2)
        cb_recs = content_based_recommendations(user_prefs, books, n // 2)
        return pd.concat([cf_recs, cb_recs]).head(n)

    elif len(user_ratings) >= 5:
        cf_recs = collab_recommendations(user_id, ratings, books, n * 2 // 3)
        cb_recs = content_based_recommendations(user_prefs, books, n // 3)
        return pd.concat([cf_recs, cb_recs]).head(n)

    else:
        return content_based_recommendations(user_prefs, books, n)


# ---- Streamlit UI ----
def main():
    st.set_page_config(page_title="Book Recommendations", page_icon="📚")
    st.title("📚 Book Recommendation Engine")

    # Defer heavy imports so missing deps show an error instead of a blank screen on deploy
    try:
        import pandas as pd
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.model_selection import train_test_split
        from difflib import SequenceMatcher
        import sys
        mod = sys.modules[__name__]
        mod.pd = pd
        mod.np = np
        mod.cosine_similarity = cosine_similarity
        mod.train_test_split = train_test_split
        mod.SequenceMatcher = SequenceMatcher
    except ImportError as e:
        st.error("Missing dependency: " + str(e))
        st.info("Add a **requirements.txt** in the repo root with: `pandas`, `numpy`, `scikit-learn`, `streamlit`")
        return

    try:
        with st.spinner("Loading data…"):
            books, ratings, users, deduped_ratings, ratings_for_rec, train, test = load_and_preprocess()
    except FileNotFoundError as e:
        st.error(str(e))
        with st.expander("Debug (for Streamlit Cloud)"):
            st.code(f"cwd: {os.getcwd()}\nDATASET: {DATASET}\nBOOKS_PATH: {BOOKS_PATH}\nexists: {os.path.isfile(BOOKS_PATH)}")
        st.info(
            "**Deploying on Streamlit Cloud?**\n\n"
            "1. **Main file path:** `group11/src/app.py`\n"
            "2. **Advanced → App directory:** `group11`\n"
            "3. **Commit the Dataset:** Ensure `group11/Dataset/` (Books.csv, Ratings.csv, Users.csv) is in your repo. "
            "If the CSVs are large, run `git status` and `git add group11/Dataset/` then push."
        )
        return
    except Exception as e:
        st.error("Something went wrong loading the app.")
        with st.expander("Error details (share this if you need help)"):
            st.exception(e)
            st.code(f"cwd: {os.getcwd()}\nBOOKS_PATH: {BOOKS_PATH}\nexists: {os.path.isfile(BOOKS_PATH)}")
        return

    if len(ratings_for_rec) < len(deduped_ratings):
        st.caption(f"Using a sample of {len(ratings_for_rec):,} ratings for fast recommendations.")

    # User ID: only IDs present in the sample used for recommendations
    user_ids = sorted(ratings_for_rec["User-ID"].unique())
    user_id = st.selectbox("User ID", options=user_ids, format_func=lambda x: f"{x}")

    # Preferences: choices from data (cap list size for responsive UI)
    authors = sorted(books["Book-Author"].dropna().unique().tolist())[:3000]
    titles = sorted(books["Book-Title"].dropna().unique().tolist(), key=str.lower)[:5000]

    favorite_authors = st.multiselect("Favorite authors", options=authors, default=[])
    favorite_books = st.multiselect("Favorite books (titles)", options=titles, default=[])

    n_recs = st.slider("Number of recommendations", 5, 20, 5)

    if st.button("Get recommendations"):
        if not favorite_authors and not favorite_books:
            st.warning("Pick at least one favorite author or book so we can recommend.")
        else:
            user_preferences = {
                "favorite_authors": favorite_authors,
                "favorite_books": favorite_books,
            }
            recommended_books = hybrid_recommendations(
                user_id, user_preferences, ratings_for_rec, books, n=n_recs
            )
            st.subheader("Recommended books")
            st.dataframe(recommended_books, use_container_width=True)


if __name__ == "__main__":
    main()
