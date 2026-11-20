"""Shared sentiment helpers used by both src/explore.ipynb and src/app.py.

Keeping the model id, the pinned revision and the star -> band mapping in one
place guarantees the notebook and the production script produce identical
labels.
"""
import os

# Silence a harmless Windows-only warning about symlinks in the HF cache.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import pipeline

# Pinned model: name + exact commit on the Hugging Face Hub.
# Never rely on "latest": if the author pushes new weights, results would change silently.
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"

BAND_ORDER = ["Negative", "Neutral", "Positive"]


def load_classifier():
    """Load the sentiment pipeline. Call this ONCE, before any inference loop.

    The weights are downloaded to the local Hugging Face cache on first use;
    they are never stored in this repository.
    """
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        revision=MODEL_REVISION,
        truncation=True,  # BERT reads at most 512 tokens
    )


def stars_from_label(label: str) -> int:
    """'4 stars' -> 4, '1 star' -> 1."""
    return int(label.split()[0])


def to_band(stars: int) -> str:
    """Map a 1-5 star value to a sentiment band (1-2 Negative, 3 Neutral, 4-5 Positive)."""
    if stars <= 2:
        return "Negative"
    if stars == 3:
        return "Neutral"
    return "Positive"


def predict(clf, texts, batch_size=16):
    """Run the (already loaded) classifier over a list of texts.

    Returns two lists: predicted stars (1-5) and model confidence (0-1).
    """
    results = clf(list(texts), batch_size=batch_size)
    stars = [stars_from_label(r["label"]) for r in results]
    scores = [round(float(r["score"]), 4) for r in results]
    return stars, scores
