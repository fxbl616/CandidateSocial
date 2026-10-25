"""Actor-weighted metrics for *marginal* trajectory candidate sets.

Energy score uses the empirical uniform distribution on K complete trajectories:
mean_k ||vec(pred_k-future)||_2 - mean_kl ||vec(pred_k-pred_l)||_2 / 2.
The second mean includes the K zero diagonal terms (denominator K squared).
No metric in this module is a scene-level joint-mode metric.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def candidate_errors(pred: torch.Tensor, future: torch.Tensor):
    """Return ADE/FDE [..., K], with pred [..., K, F, 2]."""
    if pred.ndim < 3 or pred.shape[-1] != 2 or future.shape[-1] != 2:
        raise ValueError("Expected 2D trajectory coordinates")
    if pred.shape[:-3] != future.shape[:-2] or pred.shape[-2] != future.shape[-2]:
        raise ValueError("Prediction and future shapes do not match")
    distances = torch.linalg.vector_norm(pred - future.unsqueeze(-3), dim=-1)
    return distances.mean(dim=-1), distances[..., -1]


def empirical_energy_score(pred: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """Per-actor energy score; inputs [actors,K,F,2] and [actors,F,2]."""
    if pred.ndim != 4 or future.ndim != 3:
        raise ValueError("Energy score expects actor-major trajectories")
    flat = pred.flatten(start_dim=-2)
    target = future.flatten(start_dim=-2)
    distance_to_target = torch.linalg.vector_norm(flat - target[:, None], dim=-1).mean(-1)
    pair_distances = torch.cdist(flat, flat, p=2)
    pair_distances.diagonal(dim1=-2, dim2=-1).zero_()
    return distance_to_target - 0.5 * pair_distances.mean(dim=(-2, -1))


class MetricAccumulator:
    """Aggregate sums over valid actors, never unweighted batch means."""

    def __init__(self):
        self.actors = 0
        self.sums: dict[str, float] = {}
        self.mode_counts: Optional[torch.Tensor] = None
        self.scorer_available: Optional[bool] = None
        self.has_initial: Optional[bool] = None
        self.harm_count = 0
        self.improved_count = 0
        self.mode_harm_count = 0

    def _add(self, key: str, values: torch.Tensor):
        self.sums[key] = self.sums.get(key, 0.0) + float(values.double().sum().item())

    @torch.no_grad()
    def update(self, pred, future, agent_mask, *, initial_pred=None, scores=None):
        if pred.ndim != 5 or agent_mask.shape != pred.shape[:2]:
            raise ValueError("Expected pred[B,N,K,F,2] and agent_mask[B,N]")
        mask = agent_mask.bool()
        count = int(mask.sum().item())
        if count == 0:
            return
        p = pred[mask].detach().float()
        target = future[mask].detach().float()
        if not torch.isfinite(p).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Non-finite valid prediction or ground truth")
        available = scores is not None
        if self.scorer_available is not None and available != self.scorer_available:
            raise ValueError("Cannot mix scored and unscored evaluation batches")
        self.scorer_available = available
        if self.has_initial is not None and (initial_pred is not None) != self.has_initial:
            raise ValueError("Cannot mix batches with and without initial predictions")
        self.has_initial = initial_pred is not None
        ade, fde = candidate_errors(p, target)
        best_ade, ade_winner = ade.min(dim=-1)
        paired_fde = fde.gather(-1, ade_winner[:, None]).squeeze(-1)
        self._add("minADE", best_ade)
        self._add("minFDE", fde.min(dim=-1).values)
        self._add("pairedFDE", paired_fde)
        self._add("meanADE", ade.mean(dim=-1))
        self._add("meanFDE", fde.mean(dim=-1))
        self._add("energy_score", empirical_energy_score(p, target))
        modes = p.shape[1]
        if self.mode_counts is None:
            self.mode_counts = torch.zeros(modes, dtype=torch.long)
        if len(self.mode_counts) != modes:
            raise ValueError("Candidate count changed across batches")
        self.mode_counts += torch.bincount(ade_winner.cpu(), minlength=modes)

        if initial_pred is not None:
            if initial_pred.shape != pred.shape:
                raise ValueError("Initial/final prediction shapes differ")
            initial = initial_pred[mask].detach().float()
            if not torch.isfinite(initial).all():
                raise FloatingPointError("Non-finite initial predictions")
            initial_ade, initial_fde = candidate_errors(initial, target)
            initial_best, initial_winner = initial_ade.min(dim=-1)
            self._add("initial_minADE", initial_best)
            self._add("initial_minFDE", initial_fde.min(-1).values)
            self._add("initial_pairedFDE", initial_fde.gather(-1, initial_winner[:, None]).squeeze(-1))
            self.harm_count += int((best_ade > initial_best + 1e-7).sum().item())
            self.improved_count += int((best_ade < initial_best - 1e-7).sum().item())
            self.mode_harm_count += int((ade > initial_ade + 1e-7).sum().item())

        if scores is not None:
            if scores.shape != pred.shape[:3]:
                raise ValueError("Scores must have shape [B,N,K]")
            score = scores[mask].detach().float()
            if not torch.isfinite(score).all():
                raise FloatingPointError("Non-finite valid candidate scores")
            selected = score.argmin(dim=-1)
            self._add("Top1ADE", ade.gather(-1, selected[:, None]).squeeze(-1))
            self._add("Top1FDE", fde.gather(-1, selected[:, None]).squeeze(-1))
            self._add("score_MSE", (score - ade).square().mean(-1))
            self._add("score_MAE", (score - ade).abs().mean(-1))
        self.actors += count

    def compute(self):
        if not self.actors:
            raise ValueError("No valid actors were evaluated")
        result = {key: value / self.actors for key, value in self.sums.items()}
        available = bool(self.scorer_available)
        result.update(actors=self.actors, scorer_available=available,
                      top1_status="available" if available else "unavailable_untrained_scorer")
        if not available:
            result.update(Top1ADE=None, Top1FDE=None, score_MSE=None, score_MAE=None)
        counts = self.mode_counts.tolist()
        probabilities = self.mode_counts.double() / self.actors
        nonzero = probabilities[probabilities > 0]
        entropy = float(-(nonzero * nonzero.log()).sum().item())
        result.update(num_modes=len(counts), winner_mode_counts=counts,
                      mode_utilization=probabilities.tolist(), active_modes=sum(x > 0 for x in counts),
                      mode_entropy=entropy / math.log(len(counts)) if len(counts) > 1 else 0.0)
        if self.has_initial:
            result.update(
                refinement_harm_rate=self.harm_count / self.actors,
                refinement_improvement_rate=self.improved_count / self.actors,
                per_mode_refinement_harm_rate=self.mode_harm_count / (self.actors * len(counts)),
                refinement_minADE_change=result["minADE"] - result["initial_minADE"],
            )
        else:
            result.update(refinement_harm_rate=None, refinement_improvement_rate=None,
                          per_mode_refinement_harm_rate=None, refinement_minADE_change=None)
        return result


METRIC_DEFINITIONS = {
    "averaging": "valid actors; overlapping windows remain separate actor-window observations",
    "minADE": "minimum mean Euclidean error over K, independently per actor",
    "minFDE": "minimum endpoint Euclidean error over K, independently per actor",
    "pairedFDE": "endpoint error of the same candidate that minimizes ADE",
    "Top1": "argmin predicted ADE; unavailable unless a trained scorer is loaded",
    "energy_score": "uniform empirical K; full future flattened; mean target L2 minus half mean pairwise L2 including zero diagonal; no time normalization",
    "mode_utilization": "fraction of actor-window ADE winners assigned to each mode; not an estimated future probability",
    "refinement_harm_rate": "fraction of actors whose final minADE exceeds initial minADE by >1e-7 original-coordinate units",
    "per_mode_refinement_harm_rate": "fraction of actor-mode pairs with final ADE > initial ADE + 1e-7",
    "joint_scene": False,
}
