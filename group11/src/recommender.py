"""Shared, memory-efficient recommendation engine for the app and notebook."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATASET = Path(__file__).resolve().parents[1] / "Dataset"


def load_data(dataset=DATASET):
    """Load only required columns; zero ratings mean unscored interactions."""
    dataset = Path(dataset)
    required = [dataset / name for name in ("Books.csv", "Ratings.csv")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}. Expected dataset directory: {dataset}")
    books = pd.read_csv(required[0], usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Image-URL-M"],
                        dtype=str).drop_duplicates("ISBN")
    books = books.dropna(subset=["ISBN", "Book-Title"])
    books["Book-Author"] = books["Book-Author"].fillna("Unknown author")
    books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce").fillna(0).astype(int)
    ratings = pd.read_csv(required[1], usecols=["User-ID", "ISBN", "Book-Rating"],
                          dtype={"ISBN": str, "User-ID": "int32", "Book-Rating": "float32"})
    ratings = ratings[ratings["ISBN"].isin(books["ISBN"]) & ratings["Book-Rating"].between(0, 10)]
    ratings = ratings.drop_duplicates(["User-ID", "ISBN"]).reset_index(drop=True)
    return books.reset_index(drop=True), ratings


class RecommendationEngine:
    """Aligned CSR user/item matrix + metadata TF-IDF + Bayesian popularity.

    The fitted engine is read-only during requests, suitable for cache_resource.
    No full user/user or item/item similarity matrix is materialized.
    """

    def __init__(self, books, ratings):
        self.books = books.drop_duplicates("ISBN").reset_index(drop=True).copy()
        self.books["Book-Author"] = self.books["Book-Author"].fillna("Unknown author")
        self.books["Book-Title"] = self.books["Book-Title"].fillna("Untitled")
        self.isbn_index = pd.Index(self.books["ISBN"])
        self.ratings = ratings[ratings["ISBN"].isin(self.isbn_index) & ratings["Book-Rating"].between(0, 10)].drop_duplicates(["User-ID", "ISBN"]).copy()
        explicit = self.ratings[self.ratings["Book-Rating"] > 0]
        stats = explicit.groupby("ISBN")["Book-Rating"].agg(["mean", "count"])
        self.books["rating"] = self.books["ISBN"].map(stats["mean"]).fillna(0)
        self.books["rating_count"] = self.books["ISBN"].map(stats["count"]).fillna(0).astype(int)
        prior = float(explicit["Book-Rating"].mean()) if len(explicit) else 5.0
        count = self.books["rating_count"].to_numpy()
        self.popularity = ((count * self.books["rating"].to_numpy() + 20 * prior) / (count + 20) / 10).astype(np.float32)
        # Unrated books must not outrank books with actual community support.
        self.popularity[count == 0] = 0
        self.user_index = pd.Index(sorted(explicit["User-ID"].unique()))
        self.matrix = csr_matrix((explicit["Book-Rating"].to_numpy(dtype=np.float32),
                                  (self.user_index.get_indexer(explicit["User-ID"]),
                                   self.isbn_index.get_indexer(explicit["ISBN"]))),
                                 shape=(len(self.user_index), len(self.books)), dtype=np.float32)
        self.normalized_users = normalize(self.matrix, axis=1) if len(self.user_index) else self.matrix.copy()
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=30_000, dtype=np.float32,
                                          strip_accents="unicode", token_pattern=r"(?u)\b\w+\b")
        metadata = self.books["Book-Title"] + " " + self.books["Book-Author"]
        self.content = self.vectorizer.fit_transform(metadata)
        self.title_keys = self.books["Book-Title"].str.casefold().str.strip()
        self.search_text = (self.books["Book-Title"] + " " + self.books["Book-Author"]).str.casefold()

    def search(self, query="", limit=100):
        """Literal catalog search, ranked by evidence rather than alphabetic cutoff."""
        mask = np.ones(len(self.books), dtype=bool)
        for token in query.casefold().split():
            mask &= self.search_text.str.contains(token, regex=False).to_numpy()
        return self.books.loc[mask].sort_values(["rating_count", "ISBN"], ascending=[False, True]).drop_duplicates("Book-Title").head(limit)

    def recommend(self, favorite_isbns=(), favorite_authors=(), user_id=None, n=6,
                  mode="Hybrid", min_ratings=0, exclude_isbns=()):
        if n < 1:
            return self.books.head(0).assign(score=0.0, reason="")
        seeds = self.isbn_index.get_indexer(list(favorite_isbns))
        seeds = seeds[seeds >= 0]
        seen = np.array([], dtype=int)
        if user_id is not None:
            history = self.ratings[self.ratings["User-ID"] == user_id]
            seen = self.isbn_index.get_indexer(history["ISBN"])
            liked = history[history["Book-Rating"] >= 7].nlargest(20, "Book-Rating")
            seeds = np.unique(np.concatenate([seeds, self.isbn_index.get_indexer(liked["ISBN"])]))
        content_scores = np.zeros(len(self.books), dtype=np.float32)
        if len(seeds):
            profile = normalize(csr_matrix(self.content[seeds].mean(axis=0)))
            content_scores = (self.content @ profile.T).toarray().ravel()
        author_match = self.books["Book-Author"].isin(favorite_authors).to_numpy()
        content_scores = np.maximum(content_scores, author_match.astype(np.float32))
        collaborative = np.zeros(len(self.books), dtype=np.float32)
        if len(self.user_index):
            if user_id in self.user_index:
                query = self.normalized_users[self.user_index.get_loc(user_id)]
            elif len(seeds):
                query = normalize(csr_matrix((np.ones(len(seeds), dtype=np.float32),
                                              (np.zeros(len(seeds), dtype=int), seeds)), shape=(1, len(self.books))))
            else:
                query = None
            if query is not None:
                similarities = (self.normalized_users @ query.T).toarray().ravel()
                if user_id in self.user_index:
                    similarities[self.user_index.get_loc(user_id)] = 0
                neighbors = np.argsort(-similarities, kind="stable")[:40]
                neighbors = neighbors[similarities[neighbors] > 0]
                if len(neighbors):
                    weights = similarities[neighbors]
                    collaborative = np.asarray(self.matrix[neighbors].T @ weights).ravel() / (weights.sum() * 10)
        if mode not in {"Hybrid", "Content", "Community"}:
            raise ValueError(f"Unknown recommendation mode: {mode}")
        c_weight, cf_weight = {"Hybrid": (0.55, 0.35), "Content": (0.9, 0.0), "Community": (0.0, 0.9)}[mode]
        score = c_weight * content_scores + cf_weight * collaborative + 0.1 * self.popularity
        read = self.isbn_index.get_indexer(list(exclude_isbns))
        read = read[read >= 0]
        excluded = np.unique(np.concatenate([seen, seeds, read]))
        excluded_titles = set(self.title_keys.iloc[excluded])
        eligible = (~self.title_keys.isin(excluded_titles)).to_numpy() & (self.books["rating_count"].to_numpy() >= min_ratings)
        order = np.argsort(-score, kind="stable")
        order = order[eligible[order]]
        result = self.books.iloc[order].copy()
        result["score"] = score[order]
        reasons = np.full(len(self.books), "Community favorite", dtype=object)
        reasons[self.popularity == 0] = "Explore the catalog"
        if cf_weight:
            reasons[collaborative > 0] = "Readers with similar taste"
        if c_weight:
            reasons[content_scores > 0] = "Similar title & author metadata"
            reasons[author_match] = "From an author you love"
        result["reason"] = reasons[order]
        result = result.loc[~self.title_keys.loc[result.index].duplicated()].head(n)
        return result.reset_index(drop=True)

    def summary(self):
        return {"books": len(self.books), "interactions": len(self.ratings),
                "explicit_ratings": int(self.matrix.nnz), "readers": len(self.user_index),
                "rating_matrix_mib": round((self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes) / 1024**2, 2)}
