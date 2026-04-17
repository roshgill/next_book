"""
build_features.py

Precompute and persist the feature artifacts consumed by all three
recommenders at inference time:

  1. TF-IDF vectorizer fit on book descriptions, and the resulting
     sparse document-term matrix. Consumed by the classical model for
     similarity search, and by the MLP re-ranker as an input feature.

  2. MiniLM sentence embeddings (all-MiniLM-L6-v2, 384-dim) for every
     description. Consumed by the deep model's stage-1 retrieval and by
     the MLP re-ranker as an input feature.

  3. The row-order isbn13 index so downstream code can map matrix rows
     back to book identities.

All artifacts are saved to models/ so main.py can load them once at
startup and run inference without any recomputation.

Usage:
    python scripts/build_features.py \\
        --input data/processed/catalog_clean.parquet \\
        --output-dir models/
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


# TF-IDF vectorizer settings. These are defensible defaults, not magic:
#   - max_features caps vocabulary at 20k; above that, tail terms are
#     almost never shared between books and add noise without signal.
#   - min_df=2 drops single-occurrence tokens (typos, rare names).
#   - max_df=0.85 drops tokens appearing in more than 85% of books
#     (essentially stop-word behavior on top of the english list).
#   - ngram_range=(1,2) captures short phrases ("civil war", "coming
#     of age") that single tokens miss.
TFIDF_SETTINGS = dict(
    max_features=20_000,
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 2),
    stop_words="english",
    lowercase=True,
    strip_accents="unicode",
)

# MiniLM model: 384-dim embeddings, ~90MB to download, fast on CPU.
# This is the de facto default for cheap-and-good semantic embeddings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FeatureBuilder:
    """Builds and persists TF-IDF and embedding artifacts."""

    def __init__(
        self,
        tfidf_settings: dict | None = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.tfidf_settings = tfidf_settings or TFIDF_SETTINGS
        self.embedding_model_name = embedding_model_name

    def build_tfidf(
        self, descriptions: list[str]
    ) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
        """Fit the TF-IDF vectorizer and transform all descriptions."""
        logger.info(
            "Fitting TF-IDF on %d descriptions (settings: %s)",
            len(descriptions),
            self.tfidf_settings,
        )
        vectorizer = TfidfVectorizer(**self.tfidf_settings)
        matrix = vectorizer.fit_transform(descriptions)
        logger.info(
            "TF-IDF matrix: shape=%s, nnz=%d, density=%.4f",
            matrix.shape,
            matrix.nnz,
            matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
        )
        logger.info("TF-IDF vocabulary size: %d", len(vectorizer.vocabulary_))
        return vectorizer, matrix

    def build_embeddings(self, descriptions: list[str]) -> np.ndarray:
        """Compute MiniLM embeddings for all descriptions.

        Returned matrix is L2-normalized row-wise, so downstream cosine
        similarity reduces to a dot product. This is standard for
        retrieval systems and saves a normalization step at query time.
        """
        # Lazy import so the script works even if sentence-transformers
        # isn't installed yet; the first failure will be informative.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.embedding_model_name)
        model = SentenceTransformer(self.embedding_model_name)

        logger.info(
            "Encoding %d descriptions (may take ~30s on CPU, ~5s on GPU)",
            len(descriptions),
        )
        embeddings = model.encode(
            descriptions,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine-as-dot
        )
        logger.info(
            "Embeddings: shape=%s, dtype=%s",
            embeddings.shape,
            embeddings.dtype,
        )
        return embeddings.astype(np.float32)


class ArtifactSaver:
    """Handles writing feature artifacts to disk with predictable names."""

    TFIDF_VECTORIZER = "tfidf_vectorizer.pkl"
    TFIDF_MATRIX = "tfidf_matrix.npz"
    EMBEDDINGS = "embeddings.npy"
    ISBN_INDEX = "isbn_index.npy"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_tfidf(
        self, vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix
    ) -> None:
        v_path = self.output_dir / self.TFIDF_VECTORIZER
        m_path = self.output_dir / self.TFIDF_MATRIX
        with open(v_path, "wb") as f:
            pickle.dump(vectorizer, f)
        sparse.save_npz(m_path, matrix)
        logger.info("Saved TF-IDF vectorizer to %s", v_path)
        logger.info("Saved TF-IDF matrix to %s", m_path)

    def save_embeddings(self, embeddings: np.ndarray) -> None:
        e_path = self.output_dir / self.EMBEDDINGS
        np.save(e_path, embeddings)
        logger.info(
            "Saved embeddings to %s (%.1f MB)",
            e_path,
            embeddings.nbytes / 1024 / 1024,
        )

    def save_isbn_index(self, isbn_index: np.ndarray) -> None:
        i_path = self.output_dir / self.ISBN_INDEX
        np.save(i_path, isbn_index)
        logger.info("Saved ISBN index to %s (%d entries)", i_path, len(isbn_index))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/catalog_clean.parquet"),
        help="Cleaned catalog produced by clean_catalog.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/"),
        help="Directory where feature artifacts will be written.",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding computation (faster iteration when tuning TF-IDF).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}. Run clean_catalog.py first."
        )

    logger.info("Loading cleaned catalog from %s", args.input)
    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows", len(df))

    # Validate required columns.
    for col in ("isbn13", "description"):
        if col not in df.columns:
            raise ValueError(f"Missing required column in input: {col}")

    # Reset index so row positions in the matrices align with df row positions.
    df = df.reset_index(drop=True)
    descriptions = df["description"].astype(str).tolist()
    isbn_index = df["isbn13"].astype(str).to_numpy()

    saver = ArtifactSaver(args.output_dir)
    saver.save_isbn_index(isbn_index)

    builder = FeatureBuilder()

    vectorizer, tfidf_matrix = builder.build_tfidf(descriptions)
    saver.save_tfidf(vectorizer, tfidf_matrix)

    if args.skip_embeddings:
        logger.warning("Skipping embedding computation (--skip-embeddings).")
    else:
        embeddings = builder.build_embeddings(descriptions)
        saver.save_embeddings(embeddings)

    logger.info("Feature build complete.")


if __name__ == "__main__":
    main()
