# Book Recommendation

A simple Streamlit app that recommends books based on favorite books, authors, or an existing reader's ratings.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run group11/src/app.py
```

The app uses `Books.csv` and `Ratings.csv` in `group11/Dataset`. The app and notebook share `group11/src/recommender.py`, which uses sparse matrices to avoid loading a large dense ratings table into memory.

For `group11/src/main.ipynb`, install `ipykernel` in the same environment and select it as the notebook kernel.

## Streamlit Cloud

Push your changes to the branch connected to your Streamlit app. Use `group11/src/app.py` as the main file. Python 3.13 was tested locally. Keep the dataset CSVs and root `requirements.txt` in the repository.

Existing app: https://book-recommendation01.streamlit.app/
