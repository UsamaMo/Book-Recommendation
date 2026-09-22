"""Simple Streamlit interface for book recommendations."""
from pathlib import Path
import sys
import hashlib
import importlib
import re
from urllib.parse import urlsplit

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recommender


@st.cache_resource(show_spinner=False, max_entries=1)
def build_engine(source_version):
    # Reload only on a cache miss, so code changes cannot reuse an older class.
    module = importlib.reload(recommender)
    books, ratings = module.load_data()
    return module.RecommendationEngine(books, ratings)


def get_engine():
    source = Path(recommender.__file__).read_bytes()
    return build_engine(hashlib.sha256(source).hexdigest())


def mark_read(isbn):
    read_books = set(st.session_state.get("read_books", ()))
    read_books.add(isbn)
    # Reuse the displayed list's preferences, not unsubmitted form edits.
    favorites, authors, user_id, n = st.session_state.get(
        "submitted_preferences", ((), (), None, 5)
    )
    results = get_engine().recommend(
        favorites, authors, user_id, n=n, exclude_isbns=read_books
    )
    st.session_state["read_books"] = read_books
    st.session_state["results"] = results


def cover_url(book, engine):
    # Saved recommendations can predate the addition of cover URLs.
    isbn = str(book.get("ISBN", "")).strip()
    index = engine.isbn_index.get_indexer([isbn])[0]
    sources = [book.get("Image-URL-M", "")]
    if index >= 0:
        sources.append(engine.books.iloc[index].get("Image-URL-M", ""))
    for source in sources:
        url = str(source).strip()
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
        except ValueError:
            continue
        url = url.replace("http://", "https://", 1)
        if parsed.hostname == "images.amazon.com":
            url = url.replace("https://images.amazon.com/", "https://images-na.ssl-images-amazon.com/", 1)
        return url
    if re.fullmatch(r"(?:[0-9]{9}[0-9Xx]|[0-9]{13})", isbn):
        return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
    return None


def main():
    st.set_page_config(page_title="Book Recommendations", page_icon="📚")
    st.title("Book Recommendation Engine")
    st.write("Choose your favorite books or authors to get recommendations.")

    try:
        with st.spinner("Loading books…"):
            engine = get_engine()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()

    query = st.text_input("Search books or authors", placeholder="Enter a title or author", key="search")
    matches = engine.search(query, limit=100)
    labels = {row["ISBN"]: f'{row["Book-Title"]} — {row["Book-Author"]}' for _, row in matches.iterrows()}
    # Preserve selected books when the search changes.
    for isbn in st.session_state.get("favorites", []):
        index = engine.isbn_index.get_indexer([isbn])[0]
        if index >= 0:
            row = engine.books.iloc[index]
            labels[isbn] = f'{row["Book-Title"]} — {row["Book-Author"]}'

    favorites = st.multiselect("Favorite books", list(labels), format_func=labels.get, key="favorites")
    authors = st.multiselect(
        "Favorite authors",
        sorted(set(matches["Book-Author"]) | set(st.session_state.get("authors", []))),
        key="authors",
    )
    if query and matches.empty:
        st.info("No matches found. Try a different title or author.")
    st.caption("Search to narrow the choices. Up to 100 matching books are shown.")

    user_id = None
    with st.expander("Advanced"):
        if st.checkbox("Use a reader from the dataset", key="use_reader"):
            user_id = st.selectbox("User ID", engine.user_index.tolist(), key="user_id")

    n = st.slider("Number of recommendations", 5, 20, 5)
    preferences = (tuple(favorites), tuple(authors), user_id, n)
    if st.button("Get recommendations", type="primary"):
        if not favorites and not authors and user_id is None:
            st.warning("Choose at least one book, author, or reader.")
        else:
            with st.spinner("Finding recommendations…"):
                st.session_state["results"] = engine.recommend(
                    favorites, authors, user_id, n=n,
                    exclude_isbns=st.session_state.get("read_books", ()),
                )
                st.session_state["submitted_preferences"] = preferences

    personalized = "submitted_preferences" in st.session_state
    if "results" not in st.session_state:
        st.session_state["results"] = engine.recommend(
            n=5, exclude_isbns=st.session_state.get("read_books", ())
        )

    st.subheader("Recommended books" if personalized else "Popular books")
    if personalized and preferences != st.session_state["submitted_preferences"]:
        st.caption("Preferences changed. Click Get recommendations to update this list.")
    elif not personalized:
        st.caption("Highly rated books to get you started. Choose your favorites above for personal recommendations.")

    if st.session_state.get("read_books"):
        st.caption(f"Already read: {len(st.session_state['read_books'])} books hidden for this session.")

    results = st.session_state["results"]
    if results.empty:
        st.info("No recommendations found. Try different preferences.")
    else:
        for _, book in results.iterrows():
            cover, details = st.columns([1, 5])
            with cover:
                url = cover_url(book, engine)
                if url:
                    st.image(url, width=85)
                else:
                    st.caption("Cover unavailable")
            with details:
                st.write(book["Book-Title"])
                year = int(book["Year-Of-Publication"])
                st.caption(f'{book["Book-Author"]} · {year}' if 1450 <= year <= 2100 else book["Book-Author"])
                count = int(book["rating_count"])
                st.write(f'{book["rating"]:.1f} / 10 · {count:,} ratings' if count else "Not yet rated")
                reasons = {
                    "Community favorite": "Highly rated by the reading community.",
                    "Readers with similar taste": "Readers with similar ratings also rated this book.",
                    "Similar title & author metadata": "Its title or author overlaps with books you like.",
                    "From an author you love": "Written by one of your selected authors.",
                    "Explore the catalog": "Another book to explore from the catalog.",
                }
                st.caption("Why this book? " + reasons.get(book["reason"], book["reason"]))
                st.button(
                    "Already read", key=f"read_{book['ISBN']}",
                    on_click=mark_read, args=(book["ISBN"],),
                    help="Hide this book and suggest another. Remembered for this session.",
                )
            st.divider()
        st.caption("Ratings are out of 10. Covers use the dataset’s image hosts or [Open Library](https://openlibrary.org). Some editions may not have a cover.")


if __name__ == "__main__":
    main()
