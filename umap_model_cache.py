"""
umap_model_cache.py — disk-backed cache for fitted UMAP models.

Phase 6.1: instead of re-fitting UMAP from scratch on every run, fit it
once and reuse the trained model for subsequent runs. When new videos are
added, only *transform* the new points — unless the dataset has drifted
enough that the cached embedding is stale (default: retrain when more
than 30% of samples are new).

Two backends are supported:

* **Parametric UMAP** (`umap.parametric_umap.ParametricUMAP`) — neural-net
  based; gives an exact, fast ``transform`` for out-of-sample points.
  Available only when TensorFlow is installed
  (``pip install umap-learn[parametric_umap]``).
* **Pickle + regular UMAP** — fallback when TF isn't present. Regular
  ``umap.UMAP`` still supports ``transform`` via nearest-neighbour
  approximation, and pickles cleanly.

The interface is identical for both: ``save_umap_model`` / ``load_umap_model``
/ ``should_retrain`` / ``clear_umap_model`` — the caller in `__init__.py`
doesn't have to care which backend was used.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_UMAP_CACHE_DIR = Path.home() / ".video_gap_analyzer" / "umap_models"

# When the dataset has this fraction of samples that weren't in the saved
# training set, retrain from scratch instead of transforming. 0.30 matches
# the Phase 6.1 spec — small additions reuse the model, major refreshes
# rebuild it.
UMAP_RETRAIN_THRESHOLD = 0.30

# Filenames inside each model's directory
_PICKLE_NAME = "reducer.pkl"
_PARAMETRIC_SUBDIR = "parametric"
_SAMPLE_IDS_NAME = "sample_ids.json"
_META_NAME = "meta.json"


def _safe_name(dataset_name: str) -> str:
    """Normalize a dataset name into a filesystem-safe directory fragment."""
    return "".join(
        c if (c.isalnum() or c in ("-", "_")) else "_"
        for c in (dataset_name or "unnamed")
    )


def _model_dir(
    dataset_name: str, base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Per-dataset cache directory, always under ``base_dir``."""
    base = Path(base_dir) if base_dir else DEFAULT_UMAP_CACHE_DIR
    return base / _safe_name(dataset_name)


def _parametric_available() -> bool:
    """True when ParametricUMAP is importable (i.e. TensorFlow installed)."""
    try:
        from umap.parametric_umap import ParametricUMAP  # noqa: F401
        return True
    except Exception:
        return False


def should_retrain(
    saved_ids: Iterable[str],
    current_ids: Iterable[str],
    threshold: float = UMAP_RETRAIN_THRESHOLD,
) -> tuple:
    """Decide whether the model should be retrained.

    Returns ``(retrain: bool, reason: str)``. The reason string is
    human-readable and gets surfaced in progress logs.

    Drift ratio = (samples in current NOT in saved) / max(|saved|, 1).
    A ratio above ``threshold`` triggers a rebuild.
    """
    saved = set(saved_ids)
    current = set(current_ids)

    if not saved:
        return True, "no saved sample ids — training fresh"

    new = current - saved
    drift = len(new) / max(len(saved), 1)

    if drift > threshold:
        return True, (
            f"{len(new)} new samples = {drift * 100:.1f}% drift "
            f"(> {threshold * 100:.0f}% threshold) — retraining"
        )
    return False, (
        f"{len(new)} new samples = {drift * 100:.1f}% drift "
        f"(<= {threshold * 100:.0f}% threshold) — reusing cached model"
    )


def save_umap_model(
    dataset_name: str,
    reducer: Any,
    sample_ids: Iterable[str],
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Persist a fitted UMAP model + training sample IDs to disk.

    The on-disk layout per dataset looks like::

        <base>/<safe_name>/
            reducer.pkl              (regular UMAP fallback)
            parametric/              (ParametricUMAP save-directory)
            sample_ids.json
            meta.json                ({backend, n_train_samples, ...})

    Either ``reducer.pkl`` OR ``parametric/`` exists depending on the
    available backend. Returns the cache directory on success, or
    ``None`` when we couldn't persist (e.g. ParametricUMAP + Keras 3 —
    a known incompatibility that's logged but tolerated so the pipeline
    doesn't crash).
    """
    out = _model_dir(dataset_name, base_dir=base_dir)
    # Clear any previous state so stale files from the other backend
    # can't pollute the next load.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Detect whether this is a ParametricUMAP WITHOUT forcing a TF import
    # when it isn't installed.
    is_parametric = False
    try:
        from umap.parametric_umap import ParametricUMAP  # type: ignore
        is_parametric = isinstance(reducer, ParametricUMAP)
    except ImportError:
        pass

    backend: Optional[str] = None

    if is_parametric:
        # ParametricUMAP has its own save() that writes a TF SavedModel.
        # Pickle is NOT an option for this class (the Keras model inside
        # breaks pickling), so on save failure we bail out entirely.
        try:
            reducer.save(str(out / _PARAMETRIC_SUBDIR))
            backend = "parametric"
        except Exception as e:
            logger.warning(
                "ParametricUMAP.save() failed (%s). Common cause: Keras 3 "
                "serialization regression — pin tensorflow<2.16 or use "
                "regular UMAP for now. Skipping model cache.",
                e,
            )
            shutil.rmtree(out)
            return None
    else:
        # Regular UMAP → pickle
        try:
            with open(out / _PICKLE_NAME, "wb") as f:
                pickle.dump(reducer, f)
            backend = "pickle"
        except Exception as e:
            logger.warning("Pickling UMAP reducer failed: %s", e)
            shutil.rmtree(out)
            return None

    ids_list = list(sample_ids)
    (out / _SAMPLE_IDS_NAME).write_text(json.dumps(ids_list))
    (out / _META_NAME).write_text(json.dumps({
        "backend": backend,
        "n_train_samples": len(ids_list),
    }))

    logger.info(
        "UMAP model saved (backend=%s, n_samples=%d) to %s",
        backend, len(ids_list), out,
    )
    return out


def load_umap_model(
    dataset_name: str,
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> Optional[dict]:
    """Load a previously-saved UMAP model + its training sample IDs.

    Returns ``{"reducer": ..., "sample_ids": [...], "backend": "..."}``
    on success, or ``None`` when no cached model exists, the files are
    corrupt, or (for parametric models) TensorFlow is missing.
    """
    path = _model_dir(dataset_name, base_dir=base_dir)
    if not path.exists():
        return None

    try:
        ids = json.loads((path / _SAMPLE_IDS_NAME).read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read saved UMAP sample ids at %s: %s", path, e)
        return None

    # Try parametric first since saving prefers it when available.
    parametric_path = path / _PARAMETRIC_SUBDIR
    if parametric_path.exists():
        try:
            from umap.parametric_umap import load_ParametricUMAP  # type: ignore
            reducer = load_ParametricUMAP(str(parametric_path))
            return {"reducer": reducer, "sample_ids": ids, "backend": "parametric"}
        except Exception as e:
            logger.warning(
                "Found parametric UMAP at %s but couldn't load it (%s); "
                "try `pip install umap-learn[parametric_umap]`",
                parametric_path, e,
            )
            return None

    pickle_path = path / _PICKLE_NAME
    if pickle_path.exists():
        try:
            with open(pickle_path, "rb") as f:
                reducer = pickle.load(f)
            return {"reducer": reducer, "sample_ids": ids, "backend": "pickle"}
        except Exception as e:
            logger.warning("Failed to unpickle UMAP at %s: %s", pickle_path, e)
            return None

    return None


def clear_umap_model(
    dataset_name: str,
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """Remove a dataset's cached UMAP model. Returns True iff something was deleted."""
    path = _model_dir(dataset_name, base_dir=base_dir)
    if not path.exists():
        return False
    shutil.rmtree(path)
    logger.info("Cleared UMAP model cache at %s", path)
    return True
