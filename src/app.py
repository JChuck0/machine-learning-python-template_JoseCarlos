"""Production inference path.

Reads data/raw/reviews.csv, predicts a 1-5 star sentiment for every review with
the pinned nlptown model and writes data/processed/reviews_with_sentiment.csv.

Usage (from the repo root):
    python src/app.py
"""
from pathlib import Path

import pandas as pd

from utils import MODEL_NAME, MODEL_REVISION, load_classifier, predict, to_band

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Same minimal cleaning as the notebook: trim whitespace, drop empty texts."""
    df = df.copy()
    df["review_text"] = df["review_text"].astype("string").str.strip()
    return df[df["review_text"].notna() & (df["review_text"] != "")]


def main() -> None:
    df = clean(pd.read_csv(RAW_PATH, encoding="utf-8"))

    print(f"Loading {MODEL_NAME}@{MODEL_REVISION[:7]} ...")
    clf = load_classifier()  # loaded once, reused for every review

    df["pred_stars"], df["pred_score"] = predict(clf, df["review_text"])
    df["sentiment_band"] = df["pred_stars"].apply(to_band)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"Wrote {len(df)} rows to {OUT_PATH.relative_to(ROOT)}")
    print(df["sentiment_band"].value_counts(normalize=True).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
