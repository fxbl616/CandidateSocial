"""Behavior checks for portable loading, evaluation and prediction."""
import json
import pickle

import numpy as np
import pytest
import torch

from candidate_social.config import DataConfig, ModelConfig
from candidate_social.data import build_test_dataset
from candidate_social.inference import evaluate_checkpoint, load_checkpoint, predict_checkpoint
from candidate_social.metrics import MetricAccumulator
from candidate_social.model import CandidateSocialModel


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    config = ModelConfig(obs_len=8, pred_len=12, num_modes=3, rank=4, d_model=16,
                         num_heads=2, temporal_layers=1, neighbors=2, rounds=2,
                         interaction_stability="normalized_v1")
    model = CandidateSocialModel(config).eval()
    path = tmp_path / "test.pt"
    data = DataConfig(dataset="sdd", split="sdd", unit="px").to_dict()
    data.pop("root")
    torch.save({"release_format": 1, "model_config": config.to_dict(),
                "data_config": data, "model_state": model.state_dict(), "seed": 17}, path)
    return path, model


def test_checkpoint_prediction_preserves_valid_actors_and_masks_padding(checkpoint, tmp_path):
    path, original = checkpoint
    rng = np.random.default_rng(17)
    obs = rng.normal(size=(2, 3, 8, 2)).astype(np.float32)
    mask = np.array([[True, True, False], [True, False, False]])
    obs[~mask] = np.nan
    source = tmp_path / "input.npz"
    output = tmp_path / "predictions.npz"
    np.savez(source, obs=obs, agent_mask=mask)
    predict_checkpoint(path, source, output, device="cpu", batch_size=1)
    with torch.no_grad():
        expected = original(torch.as_tensor(obs), torch.as_tensor(mask))["pred"].numpy()
    with np.load(output, allow_pickle=False) as result:
        assert result["pred"].shape == (2, 3, 3, 12, 2)
        np.testing.assert_allclose(result["pred"][mask], expected[mask], rtol=1e-5, atol=1e-6)
        assert (result["pred"][~mask] == 0).all()
        assert not np.isnan(result["pred"]).any()
    with pytest.raises(FileExistsError):
        predict_checkpoint(path, source, output, device="cpu")


def test_evaluate_needs_only_sdd_test_file_and_marks_subset(checkpoint, tmp_path):
    path, _ = checkpoint
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(5)
    raw = [rng.normal(size=(n, 20, 2)) for n in (1, 3)]
    with (root / "sdd_test.pkl").open("wb") as handle:
        pickle.dump(raw, handle)
    report = evaluate_checkpoint(path, root, tmp_path / "metrics.json", device="cpu", batch_size=1)
    assert report["metrics"]["actors"] == 4
    assert report["protocol"]["windows"] == 2
    assert report["protocol"]["is_complete_test_set"] is True
    assert report["metrics"]["num_modes"] == 3
    subset = evaluate_checkpoint(path, root, tmp_path / "subset.json", device="cpu", max_windows=1)
    assert subset["metrics"]["actors"] == 1
    assert subset["protocol"]["is_complete_test_set"] is False


def test_metric_aggregation_weights_actors_not_batches():
    metrics = MetricAccumulator()
    # One actor has error 1; two actors in a second batch each have error 3.
    for errors in ([1.0], [3.0, 3.0]):
        pred = torch.zeros(1, len(errors), 2, 2, 2)
        pred[0, :, :, :, 0] = torch.tensor(errors)[:, None, None]
        future = torch.zeros(1, len(errors), 2, 2)
        metrics.update(pred, future, torch.ones(1, len(errors), dtype=torch.bool))
    result = metrics.compute()
    assert result["minADE"] == pytest.approx(7 / 3)
    assert result["minFDE"] == pytest.approx(7 / 3)


def test_checksum_mismatch_is_not_silently_evaluated(checkpoint, tmp_path):
    path, _ = checkpoint
    payload = torch.load(path, weights_only=True)
    payload["test_source_sha256"] = {"sdd_test.pkl": "0" * 64}
    torch.save(payload, path)
    with (tmp_path / "sdd_test.pkl").open("wb") as handle:
        pickle.dump([np.zeros((1, 20, 2), dtype=np.float64)], handle)
    with pytest.raises(ValueError, match="data content differs"):
        evaluate_checkpoint(path, tmp_path, tmp_path / "result.json", device="cpu")


def test_scaled_prediction_requires_explicit_units(checkpoint, tmp_path):
    path, _ = checkpoint
    payload = torch.load(path, weights_only=True)
    payload["data_config"].update(coordinate_scale=28 / 94, unit="nba_m")
    torch.save(payload, path)
    source = tmp_path / "input.npz"
    np.savez(source, obs=np.zeros((1, 8, 2)), agent_mask=np.array([True]))
    with pytest.raises(ValueError, match="Scaled datasets require"):
        predict_checkpoint(path, source, tmp_path / "result.npz", device="cpu")


def test_partial_checkpoint_is_rejected(checkpoint):
    path, _ = checkpoint
    payload = torch.load(path, weights_only=True)
    del payload["model_state"]["basis"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="basis"):
        load_checkpoint(path, device="cpu")


def test_eth_loader_rejects_gapped_windows(tmp_path):
    folder = tmp_path / "eth" / "test"
    folder.mkdir(parents=True)
    frames = list(range(10)) + list(range(11, 21))
    text = "\n".join(f"{t} 1 {t * 0.1} 0" for t in frames)
    (folder / "biwi_eth.txt").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="no valid trajectory windows"):
        build_test_dataset(DataConfig(root=str(tmp_path)))
