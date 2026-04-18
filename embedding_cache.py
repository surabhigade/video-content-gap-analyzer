"""
Embedding Cache — persistent SQLite-backed cache for Marengo video embeddings.

Keying by the SHA-256 of each file's first 10 MB lets us skip the Twelve Labs
API entirely when the same video reappears, whether in a re-run of the pipeline
or a different FiftyOne dataset pointing at the same file on disk.

The cache lives at ``~/.video_gap_analyzer/embeddings.db`` by default.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# First 10 MB of a file is enough to discriminate videos while staying cheap on
# large files. Reading less risks collisions on re-encoded intros (logos, etc.).
HASH_BYTES = 10 * 1024 * 1024

DEFAULT_CACHE_DIR = Path.home() / ".video_gap_analyzer"
DEFAULT_CACHE_DB = DEFAULT_CACHE_DIR / "embeddings.db"

# Marengo returns float values; float32 is lossless relative to the API payload
# (which is already JSON-serialized floats) and halves the disk footprint vs f64.
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


def compute_video_hash(filepath: Union[str, os.PathLike], num_bytes: int = HASH_BYTES) -> str:
    """Return the SHA-256 hex digest of the first ``num_bytes`` of ``filepath``."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read(num_bytes))
    return h.hexdigest()


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
