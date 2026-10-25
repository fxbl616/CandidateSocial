"""Load a portable checkpoint and evaluate or predict without training files."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig
from .data import build_test_dataset, pad_collate
from .metrics import METRIC_DEFINITIONS, MetricAccumulator
from .model import CandidateSocialModel


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return selected


def load_checkpoint(path, device="auto"):
    """Load the complete model with strict names/shapes, including basis and scale."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("release_format") != 1:
        raise ValueError("Expected a CandidateSocial release_format=1 checkpoint")
    config = ModelConfig(**payload["model_config"])
    data_config = DataConfig(**payload["data_config"])
    if (config.obs_len, config.pred_len) != (data_config.obs_len, data_config.pred_len):
        raise ValueError("Checkpoint model and dataset horizons disagree")
    model = CandidateSocialModel(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.set_statistics(model.basis, model.scale)
    selected = resolve_device(device)
    return model.to(selected).eval(), payload, selected


def _check_sources(payload, protocol):
    """Optional content checks are portable: file names, never original paths."""
    expected = payload.get("test_source_sha256", {})
    if not expected:
        return "not_recorded"
    actual = {Path(item["path"]).name: item["sha256"]
              for item in protocol["source_files"]}
    if actual != expected:
        raise ValueError("Test data content differs from the checkpoint release: "
                         f"expected {expected}; found {actual}")
    return "matched"


def _output_path(path, suffix):
    target = Path(path).expanduser()
    if target.suffix.lower() != suffix:
        raise ValueError(f"Output must end in {suffix}")
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}; use a new path")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@torch.inference_mode()
def evaluate_checkpoint(checkpoint, data_root, output, *, device="auto",
                        batch_size=4, num_workers=0, max_windows=0):
    if batch_size < 1 or num_workers < 0 or max_windows < 0:
        raise ValueError("Invalid batch size, workers or max-windows")
    target = _output_path(output, ".json")
    model, payload, selected = load_checkpoint(checkpoint, device)
    data_values = dict(payload["data_config"])
    data_values.update(root=str(data_root), limit_test=max_windows)
    dataset = build_test_dataset(DataConfig(**data_values))
    checksum_status = _check_sources(payload, dataset.protocol)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=pad_collate)
    accumulator = MetricAccumulator()
    started = time.perf_counter()
    for batch in loader:
        obs = batch["obs"].to(selected)
        mask = batch["agent_mask"].to(selected)
        future = batch["future"].to(selected)
        result = model(obs, mask)
        accumulator.update(result["pred"], future, mask,
                           initial_pred=result["initial_pred"])
    elapsed = time.perf_counter() - started
    protocol = dict(dataset.protocol)
    protocol["source_files"] = [dict(name=Path(item["path"]).name,
                                     sha256=item["sha256"])
                                for item in protocol["source_files"]]
    report = {
        "release_format": 1,
        "checkpoint": Path(checkpoint).name,
        "seed": payload.get("seed"),
        "selection": payload.get("selection", {}),
        "model_config": payload["model_config"],
        "split": "test",
        "metrics": accumulator.compute(),
        "metric_definitions": METRIC_DEFINITIONS,
        "protocol": protocol,
        "source_checksum": checksum_status,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__, "device": str(selected),
                    "batch_size": batch_size, "elapsed_seconds": elapsed},
    }
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def _prediction_inputs(input_path, payload):
    with np.load(input_path, allow_pickle=False) as data:
        if "obs" not in data or "agent_mask" not in data:
            raise ValueError("Prediction NPZ needs obs and agent_mask arrays")
        obs = np.asarray(data["obs"], dtype=np.float32)
        raw_mask = np.asarray(data["agent_mask"])
        if not np.isin(raw_mask, [0, 1]).all():
            raise ValueError("agent_mask must be boolean or contain only 0 and 1")
        mask = raw_mask.astype(bool)
        metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    if not isinstance(metadata, dict):
        raise ValueError("NPZ metadata must be a scalar JSON object string")
    if obs.ndim == 3:
        obs, mask = obs[None], mask[None]
    horizon = payload["model_config"]["obs_len"]
    if obs.ndim != 4 or obs.shape[-2:] != (horizon, 2) or mask.shape != obs.shape[:2]:
        raise ValueError(f"Expected obs[B,N,{horizon},2] and agent_mask[B,N], "
                         "or both arrays without the batch axis")
    if not len(obs) or not mask.any(axis=1).all() or not np.isfinite(obs[mask]).all():
        raise ValueError("Every window needs finite coordinates for at least one valid actor")
    data_config = payload["data_config"]
    unit = data_config.get("unit", "m")
    scale = float(data_config.get("coordinate_scale", 1.0))
    if "unit" in metadata and metadata["unit"] != unit:
        raise ValueError("Input unit differs from the checkpoint unit")
    if metadata.get("coordinates_are_in_model_units", True) is not True:
        raise ValueError("Prediction inputs must already use model coordinate units")
    if "coordinate_scale_applied" in metadata and not np.isclose(
            float(metadata["coordinate_scale_applied"]), scale, rtol=1e-8, atol=0):
        raise ValueError("Input coordinate_scale_applied differs from the checkpoint")
    if scale != 1.0 and (metadata.get("coordinates_are_in_model_units") is not True
                         or "coordinate_scale_applied" not in metadata):
        raise ValueError("Scaled datasets require NPZ metadata with "
                         "coordinates_are_in_model_units=true and coordinate_scale_applied")
    return np.where(mask[..., None, None], obs, 0.0), mask, metadata


@torch.inference_mode()
def predict_checkpoint(checkpoint, input_path, output, *, device="auto", batch_size=4):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target = _output_path(output, ".npz")
    model, payload, selected = load_checkpoint(checkpoint, device)
    obs, mask, _ = _prediction_inputs(input_path, payload)
    predictions, initial = [], []
    for start in range(0, len(obs), batch_size):
        result = model(torch.as_tensor(obs[start:start+batch_size], device=selected),
                       torch.as_tensor(mask[start:start+batch_size], device=selected))
        predictions.append(result["pred"].cpu().numpy())
        initial.append(result["initial_pred"].cpu().numpy())
    pred = np.concatenate(predictions)
    metadata = {"checkpoint": Path(checkpoint).name, "seed": payload.get("seed"),
                "unit": payload["data_config"]["unit"],
                "coordinate_scale_applied": payload["data_config"].get("coordinate_scale", 1.0),
                "coordinates_are_in_model_units": True,
                "prediction_axes": ["batch", "agent", "candidate", "future_step", "xy"]}
    with target.open("wb") as stream:
        np.savez_compressed(stream, pred=pred, initial_pred=np.concatenate(initial),
                            agent_mask=mask, metadata=np.asarray(json.dumps(metadata)))
    return {"output": str(target), "prediction_shape": list(pred.shape)}
