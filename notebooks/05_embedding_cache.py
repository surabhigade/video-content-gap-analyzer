"""
05_embedding_cache.py — Persistent cache for Marengo video embeddings.

Implements and demonstrates the SQLite-backed embedding cache used by the
plugin to avoid redundant Twelve Labs calls. The cache keys on the SHA-256
of each video's first 10 MB, so the same file reappearing in a different
FiftyOne dataset still hits the cache.

Storage: ``~/.video_gap_analyzer/embeddings.db``

Run:
    python notebooks/05_embedding_cache.py path/to/a_video.mp4

The `EmbeddingCache` class below is the same implementation the plugin
imports from `embedding_cache.py`; having it inline here keeps this notebook
self-contained like the other stage notebooks.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np


# =====================================================================
# 1. Configuration
# =====================================================================

logger = logging.getLogger(__name__)

# First 10 MB is enough to tell videos apart while staying cheap on large
# files. Reading less risks collisions on shared intros (logos, bumpers).
HASH_BYTES = 10 * 1024 * 1024

DEFAULT_CACHE_DIR = Path.home() / ".video_gap_analyzer"
DEFAULT_CACHE_DB = DEFAULT_CACHE_DIR / "embeddings.db"

# Marengo embeddings arrive as JSON-serialized floats; float32 is lossless
# relative to the payload and halves the on-disk footprint vs float64.
DEFAULT_DTYPE = "float32"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    file_hash  TEXT PRIMARY KEY,
    embedding  BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    dtype      TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


# =====================================================================
# 2. Content-based hashing
# =====================================================================

def compute_video_hash(filepath: Union[str, os.PathLike], num_bytes: int = HASH_BYTES) -> str:
    """Return the SHA-256 hex digest of the first ``num_bytes`` of ``filepath``."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read(num_bytes))
    return h.hexdigest()


# =====================================================================
# 3. The cache itself
# =====================================================================

class EmbeddingCache:
    """SQLite-backed cache mapping file-content hashes to embedding vectors."""

    def __init__(self, db_path: Optional[Union[str, os.PathLike]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_CACHE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def get(self, filepath: Union[str, os.PathLike]) -> Optional[np.ndarray]:
        """Return the cached embedding for ``filepath``'s content, or ``None``."""
        try:
            file_hash = compute_video_hash(filepath)
        except OSError as e:
            logger.warning("Could not hash %s: %s", filepath, e)
            return None

        row = self.conn.execute(
            "SELECT embedding, dim, dtype FROM embeddings WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        if row is None:
            return None

        blob, dim, dtype = row
        arr = np.frombuffer(blob, dtype=np.dtype(dtype))
        if arr.size != dim:
            logger.warning(
                "Cache entry %s is corrupted (size %d != dim %d); ignoring",
                file_hash[:12], arr.size, dim,
            )
            return None
        return arr.copy()

    def put(
        self,
        filepath: Union[str, os.PathLike],
        embedding,
    ) -> str:
        """Store ``embedding`` under ``filepath``'s content hash. Returns the hash."""
        file_hash = compute_video_hash(filepath)
        arr = np.asarray(embedding, dtype=np.dtype(DEFAULT_DTYPE)).reshape(-1)
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(file_hash, embedding, dim, dtype, created_at) VALUES (?, ?, ?, ?, ?)",
            (file_hash, arr.tobytes(), int(arr.size), DEFAULT_DTYPE, time.time()),
        )
        self.conn.commit()
        return file_hash

    def clear(self) -> int:
        """Delete all cache entries. Returns the number of rows removed."""
        cur = self.conn.execute("DELETE FROM embeddings")
        self.conn.commit()
        return int(cur.rowcount)

    def stats(self) -> dict:
        """Return {entries, db_path, size_bytes} describing the cache on disk."""
        count = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "entries": int(count),
            "db_path": str(self.db_path),
            "size_bytes": int(size_bytes),
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# =====================================================================
# 4. Demo
# =====================================================================

def main(sample_path: str) -> None:
    """Walk through the cache lifecycle against a real file on disk."""
    sample_file = Path(sample_path)
    if not sample_file.exists():
        raise SystemExit(f"No such file: {sample_file}")

    print(f"Hashing first 10 MB of: {sample_file}")
    file_hash = compute_video_hash(sample_file)
    print(f"  sha256 = {file_hash}")

    with EmbeddingCache() as cache:
        print(f"\nCache location: {cache.db_path}")

        hit = cache.get(sample_file)
        if hit is not None:
            print(f"  existing cache hit: shape={hit.shape}, dtype={hit.dtype}")
        else:
            print("  cache miss — inserting a fake 512-d vector for demo purposes")
            fake_embedding = np.random.default_rng(0).standard_normal(512).astype("float32")
            cache.put(sample_file, fake_embedding)

            round_trip = cache.get(sample_file)
            assert round_trip is not None, "put-then-get failed"
            assert np.allclose(round_trip, fake_embedding), "round trip mismatch"
            print(f"  round trip ok: shape={round_trip.shape}, dtype={round_trip.dtype}")

        print(f"\nStats after operations: {cache.stats()}")

        # Leave the entry behind so future runs show a cache hit. Uncomment
        # to wipe the cache entirely:
        # deleted = cache.clear()
        # print(f"  cleared {deleted} entries")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path:
        raise SystemExit("Usage: python notebooks/05_embedding_cache.py <video_file>")
    main(path)
