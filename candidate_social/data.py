"""Test-only trajectory loading, retaining the original window/actor order."""
from __future__ import annotations

import copy
import hashlib
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig


class DataProtocolError(ValueError):
    """The requested data cannot support the declared evaluation protocol."""


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _identifier(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, str) and value:
        return value
    raise DataProtocolError(f"Invalid actor/frame identifier: {value!r}")


def _scene_from_path(path: Path) -> str:
    # Official ETH/UCY source filenames append only the partition suffix.
    return re.sub(r"_(train|val|test)$", "", path.stem)


class TrajectoryDataset(Dataset):
    """A record retains actor order and declared observed + future frame metadata."""

    def __init__(self, records: list[dict], obs_len: int, pred_len: int, protocol: dict):
        self.records = records
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.protocol = copy.deepcopy(protocol)
        self.split = self.protocol["role"]
        self.protocol["windows"] = len(records)
        self.protocol["actor_windows"] = sum(len(record["trajectory"]) for record in records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # Clone so caller-side augmentation cannot change the stored evaluation data.
        trajectory = torch.from_numpy(record["trajectory"]).clone()
        return {
            "obs": trajectory[:, : self.obs_len],
            "future": trajectory[:, self.obs_len :],
            "metadata": copy.deepcopy(record["metadata"]),
        }


def pad_collate(batch: list[dict]) -> dict:
    """Pad actor rows only; metadata keeps exactly the unpadded actor order."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    obs_len, pred_len = batch[0]["obs"].shape[1], batch[0]["future"].shape[1]
    counts = [item["obs"].shape[0] for item in batch]
    if min(counts) < 1:
        raise ValueError("Every window must contain at least one valid actor")
    obs = torch.zeros(len(batch), max(counts), obs_len, 2, dtype=torch.float32)
    future = torch.zeros(len(batch), max(counts), pred_len, 2, dtype=torch.float32)
    mask = torch.zeros(len(batch), max(counts), dtype=torch.bool)
    for row, (item, count) in enumerate(zip(batch, counts)):
        if item["obs"].shape != (count, obs_len, 2) or item["future"].shape != (count, pred_len, 2):
            raise ValueError("Inconsistent trajectory shapes in batch")
        if not torch.isfinite(item["obs"]).all() or not torch.isfinite(item["future"]).all():
            raise ValueError("Non-finite coordinates in batch")
        obs[row, :count] = item["obs"]
        future[row, :count] = item["future"]
        mask[row, :count] = True
    return {
        "obs": obs,
        "future": future,
        "agent_mask": mask,
        "num_agents": torch.tensor(counts, dtype=torch.long),
        "metadata": [copy.deepcopy(item["metadata"]) for item in batch],
    }


def _read_eth_partition(directory: Path, cfg: DataConfig):
    if not directory.is_dir():
        raise FileNotFoundError(f"Required explicit ETH/UCY partition is missing: {directory}")
    paths = sorted(directory.glob("*.txt"))
    if not paths:
        raise DataProtocolError(f"No ETH/UCY trajectory files in {directory}")
    records, coverage, diagnostics = [], defaultdict(set), Counter()
    seq_len = cfg.obs_len + cfg.pred_len
    for path in paths:
        scene = _scene_from_path(path)
        frame_data = defaultdict(dict)
        for line_index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            values = line.replace(",", " ").split()
            if len(values) < 4:
                raise DataProtocolError(f"Expected frame, actor, x, y at {path}:{line_index}")
            try:
                frame, actor, x, y = map(float, values[:4])
            except ValueError as error:
                raise DataProtocolError(f"Invalid numeric trajectory row at {path}:{line_index}") from error
            if not all(math.isfinite(v) for v in (frame, actor, x, y)):
                raise DataProtocolError(f"Non-finite trajectory row at {path}:{line_index}")
            if actor in frame_data[frame]:
                raise DataProtocolError(f"Duplicate (frame, actor) at {path}:{line_index}")
            frame_data[frame][actor] = (x, y)
            coverage[scene].add(_identifier(frame))
        frames = sorted(frame_data)
        if len(frames) < seq_len:
            diagnostics["short_source_files"] += 1
            continue
        # The model's LS velocity assumes a uniform sampling interval. Empty-scene
        # gaps must never be compressed into a fictitious single sampling step.
        nominal_step = min(np.diff(frames))
        for start in range(0, len(frames) - seq_len + 1, cfg.stride):
            window_frames = frames[start : start + seq_len]
            if not np.allclose(np.diff(window_frames), nominal_step, rtol=0, atol=1e-8):
                diagnostics["nonuniform_time_windows"] += 1
                continue
            actors = set(frame_data[window_frames[0]])
            for frame in window_frames[1:]:
                actors.intersection_update(frame_data[frame])
            actor_ids = sorted(actors)
            if len(actor_ids) < cfg.min_agents:
                diagnostics["insufficient_complete_actors"] += 1
                continue
            trajectory = np.asarray(
                [[frame_data[frame][actor] for frame in window_frames] for actor in actor_ids],
                dtype=np.float32,
            ) * cfg.coordinate_scale
            records.append({
                "trajectory": np.ascontiguousarray(trajectory, dtype=np.float32),
                "metadata": {
                    "scene_name": scene,
                    "actor_ids": [_identifier(actor) for actor in actor_ids],
                    "frame_ids": [_identifier(frame) for frame in window_frames],
                    "source_file": str(path.resolve()),
                    "source_index": start,
                    "frame_step": float(nominal_step),
                    "provenance": "original_text_frame_actor_rows",
                },
            })
    return records, dict(coverage), [_source(path) for path in paths], dict(diagnostics)


class _NumpyDataUnpickler(pickle.Unpickler):
    """Load plain containers + numeric NumPy arrays, never arbitrary classes."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle global {module}.{name}")
        return getattr(__import__(module, fromlist=[name]), name)


def _read_sdd_file(path: Path, cfg: DataConfig):
    if not path.is_file():
        raise FileNotFoundError(f"Required SDD partition is missing: {path}")
    with path.open("rb") as handle:
        raw_records = _NumpyDataUnpickler(handle).load()
    if not isinstance(raw_records, (list, tuple)):
        raise DataProtocolError(f"SDD file must contain a list of window dictionaries: {path}")
    records, coverage, diagnostics = [], defaultdict(set), Counter()
    seq_len = cfg.obs_len + cfg.pred_len
    for index, raw in enumerate(raw_records):
        if isinstance(raw, np.ndarray):
            # MART's published SDD file stores only the complete coordinates.
            # Its ordered float32 arrays match the evaluated metadata version.
            if raw.dtype.kind not in "iuf" or raw.ndim != 3 or raw.shape[1:] != (seq_len, 2):
                raise DataProtocolError(f"Expected SDD [N,{seq_len},2], record {index}")
            trajectory = np.ascontiguousarray(raw, dtype=np.float32) * cfg.coordinate_scale
            if not np.isfinite(trajectory).all():
                raise DataProtocolError(f"Non-finite SDD coordinates, record {index}")
            if len(trajectory) < cfg.min_agents:
                diagnostics["insufficient_complete_actors"] += 1
                continue
            records.append({
                "trajectory": trajectory,
                "metadata": {
                    "scene_name": f"sdd_test_window_{index}",
                    "actor_ids": list(range(len(trajectory))),
                    "frame_ids": list(range(seq_len)),
                    "source_file": str(path.resolve()),
                    "source_index": index,
                    "frame_step": 1.0,
                    "provenance": "MART_prebuilt_test_window_relative_slots_only",
                },
            })
            continue
        if not isinstance(raw, dict):
            raise DataProtocolError("SDD windows require scene_name, frames and agent_ids metadata; anonymous arrays cannot be safely split")
        frames = raw.get("frames", raw.get("frame_ids"))
        status = raw.get("match_status", "explicit_metadata")
        if status == "augmented_nn":
            # Existing ViTE records with NN-recovered identities have no original
            # timestamps. Keep the strict subset instead of claiming exact provenance.
            diagnostics["excluded_augmented_nn_unverified_provenance"] += 1
            continue
        scene = raw.get("scene_name")
        actors = raw.get("agent_ids", raw.get("actor_ids"))
        if not isinstance(scene, str) or not scene.strip() or frames is None or actors is None:
            raise DataProtocolError(f"Missing auditable scene/frame/actor metadata at {path}, record {index}; supply explicit metadata-rich partitions")
        trajectory = np.asarray(raw.get("traj", raw.get("trajectory")))
        if trajectory.dtype.kind not in "iuf" or trajectory.ndim != 3 or trajectory.shape[-1] != 2:
            raise DataProtocolError(f"Expected numeric SDD trajectory [N,T,2] or [T,N,2], record {index}")
        if trajectory.shape[1] != seq_len and trajectory.shape[0] == seq_len:
            trajectory = trajectory.transpose(1, 0, 2)
        if trajectory.shape[1] != seq_len:
            raise DataProtocolError(f"SDD record {index} has {trajectory.shape[1]} steps; requested {seq_len}")
        frame_ids = [_identifier(value) for value in frames]
        actor_ids = [_identifier(value) for value in actors]
        if len(frame_ids) != seq_len or len(actor_ids) != len(trajectory):
            raise DataProtocolError(f"Metadata lengths disagree with SDD trajectory, record {index}")
        if len(set(actor_ids)) != len(actor_ids):
            raise DataProtocolError(f"Duplicate actor IDs within SDD window, record {index}")
        differences = np.diff(np.asarray(frame_ids, dtype=np.float64))
        if not np.all(differences > 0) or not np.allclose(differences, differences[0], rtol=0, atol=1e-8):
            raise DataProtocolError(f"SDD frames must be increasing and uniformly sampled, record {index}")
        if not np.isfinite(trajectory).all():
            raise DataProtocolError(f"Non-finite SDD coordinates, record {index}")
        coverage[scene].update(frame_ids)
        if len(trajectory) < cfg.min_agents:
            diagnostics["insufficient_complete_actors"] += 1
            continue
        records.append({
            "trajectory": np.ascontiguousarray(trajectory * cfg.coordinate_scale, dtype=np.float32),
            "metadata": {
                "scene_name": scene,
                "actor_ids": actor_ids,
                "frame_ids": frame_ids,
                "source_file": str(path.resolve()),
                "source_index": index,
                "frame_step": float(differences[0]),
                "provenance": status,
            },
        })
    return records, dict(coverage), [_source(path)], dict(diagnostics)


def build_test_dataset(config: DataConfig) -> TrajectoryDataset:
    """Read only the published test partition; no training data are required."""
    if config.obs_len < 4 or config.pred_len < 1 or config.stride < 1 or config.min_agents < 1:
        raise ValueError("Invalid history, horizon, stride or min_agents")
    if not math.isfinite(config.coordinate_scale) or config.coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be finite and positive")
    if config.limit_test < 0:
        raise ValueError("limit_test must be nonnegative")
    root = Path(config.root).expanduser().resolve()
    name = config.dataset.lower()
    if name in {"ethucy", "eth_ucy"}:
        if config.split not in {"eth", "hotel", "univ", "zara1", "zara2"}:
            raise ValueError("Unknown ETH/UCY held-out split")
        records, _, sources, diagnostics = _read_eth_partition(root / config.split / "test", config)
        strategy = "provided_test_text_partition"
    elif name == "sdd":
        records, _, sources, diagnostics = _read_sdd_file(root / "sdd_test.pkl", config)
        strategy = "MART_prebuilt_test_windows"
    elif name == "nba":
        from .nba_data import load_nba_test_records
        records, sources, diagnostics = load_nba_test_records(root, config)
        strategy = "MART_official_test_slice_0_12500"
    else:
        raise ValueError("Supported datasets are ethucy, sdd and nba")
    count = len(records)
    if not count:
        raise DataProtocolError("The test partition has no valid trajectory windows")
    selected = records[:config.limit_test] if config.limit_test else records
    protocol = {
        "role": "test", "dataset": name, "held_out_split": config.split,
        "source_files": sources, "partition_strategy": strategy,
        "diagnostics": diagnostics, "windows_before_limit": count,
        "limit": config.limit_test, "is_complete_test_set": len(selected) == count,
        "obs_len": config.obs_len, "pred_len": config.pred_len,
        "coordinate_scale": config.coordinate_scale, "unit": config.unit,
        "min_agents": config.min_agents,
        "stride": config.stride if name in {"ethucy", "eth_ucy"} else "prebuilt_windows",
        "averaging": "all valid actor-windows; overlaps retained",
    }
    if name == "nba":
        protocol.update(entities=11, ball_slot=10, ball_included=True,
                        sample_hz=5.0, coordinate_arithmetic="source dtype divide by 94/28, then float32",
                        coordinate_unit_note="NBA benchmark 28/94 scaling")
    return TrajectoryDataset(selected, config.obs_len, config.pred_len, protocol)
