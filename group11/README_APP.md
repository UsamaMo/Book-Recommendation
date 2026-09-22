# Book Recommendation App

See the [root README](../README.md) for setup and deployment.

From the repository root:

```bash
pip install -r requirements.txt
streamlit run group11/src/app.py
```

The page starts with popular books. Choose favorite books or authors and click **Get recommendations** to get a list with covers, ratings, and a short explanation. Results stay visible until you request a new list. The optional dataset reader selector is under **Advanced**.

Use **Already read** to replace a book and hide it (including editions with the same title) from future recommendations during the current session.
