"""MART-compatible NBA test windows and original coordinate arithmetic."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .config import DataConfig
from .data import DataProtocolError, TrajectoryDataset


MART_SPLITS = {
    "train": ("nba_train.npy", 0, 32500),
    "val": ("nba_train.npy", 32500, 37500),
    "test": ("nba_test.npy", 0, 12500),
}
NBA_COORDINATE_SCALE = 28.0 / 94.0
NBA_SHAPE = (30, 11, 2)
NBA_SAMPLE_HZ = 5.0


def _source(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _read_array(path: Path) -> np.ndarray:
    """Read a fixed-axis numeric array; reject pickle and silent transposition."""
    if not path.is_file():
        raise FileNotFoundError(f"Required NBA source is missing: {path}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, TypeError) as error:
        raise DataProtocolError(f"NBA source must be an unpickled numeric .npy array: {path}") from error
    if not isinstance(array, np.ndarray) or array.ndim != 4 or array.shape[1:] != NBA_SHAPE:
        raise DataProtocolError(f"Expected NBA [samples,30,11,2] without transposition: {path}")
    if array.dtype.kind != "f":
        raise DataProtocolError(f"NBA coordinates must use a real floating-point dtype: {path}")
    for start in range(0, len(array), 1024):
        if not np.isfinite(array[start : start + 1024]).all():
            raise DataProtocolError(f"Non-finite NBA coordinates: {path}")
    return array


def _make_records(array: np.ndarray, path: Path, indices, cfg: DataConfig) -> list[dict]:
    records = []
    path = path.resolve()
    for source_index in indices:
        source_index = int(source_index)
        # Match MART's in-place division in the source floating dtype before the
        # final float32 conversion; multiplying by a rounded inverse can differ.
        coordinates = np.array(array[source_index], copy=True)
        coordinates /= 94.0 / 28.0
        trajectory = np.ascontiguousarray(coordinates.transpose(1, 0, 2), dtype=np.float32)
        if not np.isfinite(trajectory).all():
            raise DataProtocolError(f"NBA coordinates exceed float32 range after scaling: {path}, row {source_index}")
        records.append({
            "trajectory": trajectory,
            "metadata": {
                "scene_name": f"nba_sample_{path.stem}_{source_index}",
                "scene_id_scope": "synthetic_sample_identifier_not_game_id",
                "actor_ids": list(range(11)),
                "actor_slot_indices": list(range(11)),
                "actor_id_scope": "within_window_slots_not_verified_persistent_ids",
                "frame_ids": list(range(30)),
                "frame_id_scope": "relative_sample_steps_not_global_source_frames",
                "source_file": str(path),
                "source_index": source_index,
                "frame_step": 1.0,
                "sample_hz": NBA_SAMPLE_HZ,
                "ball_slot": 10,
                "true_game_event_frame_ids_available": False,
                "provenance": "MART_prebuilt_anonymous_numpy_window",
            },
        })
    return records


def _validate_config(cfg: DataConfig) -> None:
    if cfg.dataset.lower() != "nba" or cfg.split != "mart_official":
        raise DataProtocolError("NBA requires dataset='nba' and split='mart_official'")
    if (cfg.obs_len, cfg.pred_len) != (10, 20):
        raise DataProtocolError("MART NBA arrays require 10 observed and 20 predicted steps (5 Hz)")
    if cfg.stride != 1:
        raise DataProtocolError("NBA uses prebuilt windows; stride must equal 1")
    if not 1 <= cfg.min_agents <= 11:
        raise DataProtocolError("NBA windows contain exactly 11 entities, including the ball")
    if not math.isfinite(cfg.coordinate_scale) or not math.isclose(
        cfg.coordinate_scale, NBA_COORDINATE_SCALE, rel_tol=0, abs_tol=1e-12
    ):
        raise DataProtocolError("MART NBA requires the benchmark coordinate scale 28/94")
    if cfg.unit != "nba_m":
        raise DataProtocolError("NBA benchmark coordinates must be labelled 'nba_m' (28/94 scale)")
    if min(cfg.limit_train, cfg.limit_val, cfg.limit_test) < 0:
        raise DataProtocolError("NBA dataset limits must be nonnegative")


def load_nba_test_records(root: Path, cfg: DataConfig):
    _validate_config(cfg)
    filename, start, stop = MART_SPLITS["test"]
    path = root / filename
    array = _read_array(path)
    if len(array) < stop:
        raise DataProtocolError(f"Official NBA test slice requires {stop} rows; found {len(array)}")
    records = _make_records(array, path, range(start, stop), cfg)
    return records, [_source(path)], {"insufficient_complete_actors": 0}
