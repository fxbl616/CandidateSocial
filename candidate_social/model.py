"""Candidate-conditioned, synchronous trajectory refinement in a fixed basis.

The ``self`` control uses each actor's own K candidates as its only candidate
bank. It reuses the complete interaction/refinement modules and never reads
another actor. Variants have the same registered parameters, but ``none``,
``history`` and ``flat`` activate different computations; equal FLOPs or equal
effective parameter counts are not claimed.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .geometry import edge_frames, local_frame, multi_scale_features, neighbor_edges, pair_geometry, to_world


class CandidateSocialModel(nn.Module):
    """A2 generator. Inputs/outputs use original dataset coordinates.

    ``basis`` [2F,r] and scalar ``scale`` are frozen registered buffers. When
    omitted, the basis is an orthonormal placeholder for checkpoint loading;
    training must replace it with statistics fitted on the source training set.
    No mode index is assumed to align across actors.
    """

    geometry_dim = 32
    attention_dim = 32

    def __init__(self, config: ModelConfig, basis=None, scale=1.0):
        super().__init__()
        self.config = config
        d, rank = config.d_model, config.rank
        self.register_buffer("basis", torch.eye(2 * config.pred_len)[:, :rank].clone())
        self.register_buffer("scale", torch.tensor(1.0))
        self.input_projection = nn.Linear(11, d)
        self.time_embedding = nn.Parameter(torch.empty(config.obs_len, d))
        self.mode_queries = nn.Parameter(torch.empty(config.num_modes, d))
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.mode_queries, std=0.02)
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, nhead=config.num_heads, dim_feedforward=2 * d,
                dropout=config.dropout, activation=F.gelu, batch_first=True,
                norm_first=True, layer_norm_eps=config.epsilon,
            ) for _ in range(config.temporal_layers)
        ])
        for layer in self.temporal_layers:
            # A2 temporal Q/K/V and output projections have no bias; FFN and
            # layer-normalization affine parameters retain their own biases.
            layer.self_attn = nn.MultiheadAttention(
                d, config.num_heads, dropout=config.dropout, bias=False,
                batch_first=True,
            )
        self.history_norm = nn.LayerNorm(d, eps=config.epsilon)
        self.query_norm = nn.LayerNorm(d, eps=config.epsilon)
        self.coefficient_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, rank))
        self.trajectory_embedding = nn.Linear(2 * config.pred_len, d)
        self.candidate_norm = nn.LayerNorm(d, eps=config.epsilon)

        self.geometry_embedding = nn.Sequential(nn.Linear(27, self.geometry_dim), nn.GELU())
        self.mode_query = nn.Linear(d, self.attention_dim, bias=False)
        self.mode_key = nn.Linear(d, self.attention_dim, bias=False)
        self.geometry_query = nn.Linear(d, self.geometry_dim, bias=False)
        self.value_projection = nn.Linear(d, d, bias=False)
        self.edge_projection = nn.Linear(self.geometry_dim, d, bias=False)
        self.neighbor_query = nn.Linear(d, self.attention_dim, bias=False)
        self.neighbor_message = nn.Linear(d, self.attention_dim, bias=False)
        self.history_edge_projection = nn.Linear(6, self.attention_dim, bias=False)
        self.null_score = nn.Linear(d, 1)
        self.refinement_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, rank))
        # Start near the initial trajectories while retaining nonzero gradients
        # through the interaction block from the first optimization step.
        nn.init.normal_(self.refinement_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.refinement_head[-1].bias)
        self.set_statistics(self.basis if basis is None else basis, scale)

    @property
    def score_feature_dim(self) -> int:
        return 3 * self.config.d_model + self.config.rank

    @torch.no_grad()
    def set_statistics(self, basis, scale):
        basis = torch.as_tensor(basis, device=self.basis.device, dtype=self.basis.dtype).detach()
        scale = torch.as_tensor(scale, device=self.scale.device, dtype=self.scale.dtype).detach()
        if basis.shape != self.basis.shape or not bool(torch.isfinite(basis).all()):
            raise ValueError(f"basis must be finite with shape {tuple(self.basis.shape)}")
        identity = torch.eye(self.config.rank, dtype=basis.dtype, device=basis.device)
        if not torch.allclose(basis.T @ basis, identity, atol=1e-4, rtol=1e-4):
            raise ValueError("basis columns must be orthonormal")
        if scale.numel() != 1 or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError("scale must be one finite positive scalar")
        self.basis.copy_(basis)
        self.scale.copy_(scale.reshape(()))
        return self

    def _encode_history(self, local_obs: Tensor, mask: Tensor) -> Tensor:
        actors = mask.numel()
        features = multi_scale_features(local_obs).reshape(actors, self.config.obs_len, 11)
        encoded = self.input_projection(features) + self.time_embedding[None]
        for layer in self.temporal_layers:
            encoded = layer(encoded)
        history = self.history_norm(encoded.mean(dim=1))
        return torch.where(mask[:, None], history, torch.zeros_like(history))

    def _local_predictions(self, coefficients: Tensor, reference: Tensor) -> Tensor:
        residual = F.linear(coefficients, self.basis).unflatten(-1, (self.config.pred_len, 2))
        return reference[:, None] + residual

    def _candidate_features(self, history: Tensor, predictions: Tensor, mask: Tensor) -> Tensor:
        features = self.candidate_norm(
            history[:, None] + self.mode_queries[None]
            + self.trajectory_embedding(predictions.flatten(-2))
        )
        return torch.where(mask[:, None, None], features, torch.zeros_like(features))

    def _unit(self, value: Tensor) -> Tensor:
        return F.normalize(value, dim=-1, eps=self.config.epsilon)

    def _message_norm(self, value: Tensor) -> Tensor:
        # No trainable affine scale can reintroduce unbounded message norms.
        return F.layer_norm(value, (value.shape[-1],), eps=self.config.epsilon)

    def _null_logits(self, features: Tensor) -> Tensor:
        logits = self.null_score(features).squeeze(-1)
        if self.config.interaction_stability == "normalized_v1":
            logits = 2.0 * torch.tanh(logits / 2.0)
        return logits

    def _edge_block(self, target_features: Tensor, neighbor_features: Tensor,
                    target_prediction: Tensor, neighbor_prediction: Tensor,
                    historical_edge: Tensor, relative_rotation: Tensor) -> tuple[Tensor, Tensor]:
        """Reduce an edge block before retaining outputs; never align mode IDs."""
        geometry = pair_geometry(target_prediction, neighbor_prediction, historical_edge,
                                 relative_rotation, self.config.geometry_temperature)
        stable = self.config.interaction_stability == "normalized_v1"
        # asinh is smooth at zero, sign preserving, and logarithmic at large
        # magnitudes. Coordinates remain in the training-derived local units.
        phi = self.geometry_embedding(torch.asinh(geometry) if stable else geometry)
        if stable:
            phi = self._message_norm(phi)
        query = self.mode_query(target_features)
        key = self.mode_key(neighbor_features)
        geometry_query = self.geometry_query(target_features)
        if stable:
            # Fixed cosine scale 2 per branch: pair logits lie in [-4,4].
            compatibility = 2.0 * (
                torch.einsum("eka,ela->ekl", self._unit(query), self._unit(key))
                + torch.einsum("ekg,eklg->ekl", self._unit(geometry_query), self._unit(phi)))
        else:
            compatibility = (torch.einsum("eka,ela->ekl", query, key) / math.sqrt(self.attention_dim)
                             + torch.einsum("ekg,eklg->ekl", geometry_query, phi) / math.sqrt(self.geometry_dim))
        # Candidate temperature is fixed to the A2 starting value 1.0.
        mode_weights = compatibility.softmax(dim=-1)
        feature_summary = torch.einsum("ekl,eld->ekd", mode_weights, neighbor_features)
        geometry_summary = torch.einsum("ekl,eklg->ekg", mode_weights, phi)
        messages = self.value_projection(feature_summary) + self.edge_projection(geometry_summary)
        if stable:
            messages = self._message_norm(messages)
        if self.config.interaction == "flat":
            # These log masses make the subsequent edge reduction exactly one
            # softmax over all neighboring (actor, mode) pairs, plus a null key.
            edge_logits = compatibility.logsumexp(dim=-1)
        else:
            history_input = torch.asinh(historical_edge) if stable else historical_edge
            edge_keys = self.neighbor_message(messages) + self.history_edge_projection(history_input)[:, None]
            if stable:
                edge_logits = 2.0 * (self._unit(self.neighbor_query(target_features))
                                     * self._unit(edge_keys)).sum(dim=-1)
            else:
                edge_logits = (self.neighbor_query(target_features) * edge_keys).sum(dim=-1) / math.sqrt(self.attention_dim)
        return messages, edge_logits

    @staticmethod
    def _reduce_edges(messages: Tensor, logits: Tensor, sources: Tensor,
                      null_logits: Tensor, actor_count: int) -> tuple[Tensor, Tensor]:
        """Stable segment softmax over actual neighbors; empty groups stay zero."""
        modes, feature_dim = messages.shape[1:]
        indices = sources[:, None].expand(-1, modes)
        maximum = logits.new_full((actor_count, modes), -torch.inf)
        maximum = maximum.scatter_reduce(0, indices, logits, reduce="amax", include_self=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum)).detach()
        exponentials = (logits - maximum[sources]).exp()
        sums = logits.new_zeros(actor_count, modes).index_add(0, sources, exponentials)
        weights = exponentials / sums[sources]
        conditional_message = messages.new_zeros(actor_count, modes, feature_dim).index_add(
            0, sources, weights[..., None] * messages
        )
        # sums are >=1 for every nonempty group after max subtraction.
        log_mass = sums.clamp_min(torch.finfo(sums.dtype).tiny).log() + maximum
        gate = torch.sigmoid(log_mass - null_logits)
        gate = torch.where(sums > 0, gate, torch.zeros_like(gate))
        return conditional_message, gate

    def _interact(self, features: Tensor, local_prediction: Tensor, history: Tensor,
                  reference: Tensor, edges: tuple[Tensor, Tensor], historical_edge: Tensor,
                  relative_rotation: Tensor) -> tuple[Tensor, Tensor]:
        actors, modes, width = features.shape
        sources, destinations = edges
        if sources.numel() == 0:
            return features.new_zeros(actors, modes, width), features.new_zeros(actors, modes)

        variant = self.config.interaction
        if variant == "history":
            # One history/CV-based message is shared by all target hypotheses.
            edge_features = history[:, None]
            edge_predictions = reference[:, None]
            null_logits = self._null_logits(history)[:, None].expand(-1, modes)
        else:
            edge_features, edge_predictions = features, local_prediction
            null_logits = self._null_logits(features)
        message_blocks, logit_blocks = [], []
        for begin in range(0, sources.numel(), self.config.edge_chunk_size):
            end = begin + self.config.edge_chunk_size
            src, dst = sources[begin:end], destinations[begin:end]
            inputs = (edge_features[src], edge_features[dst], edge_predictions[src],
                      edge_predictions[dst], historical_edge[begin:end], relative_rotation[begin:end])
            if self.training and torch.is_grad_enabled():
                # Recompute K^2 temporal/geometry tensors in backward instead of
                # retaining them for all chunks in the training autograd graph.
                message, logit = checkpoint(self._edge_block, *inputs, use_reentrant=False)
            else:
                message, logit = self._edge_block(*inputs)
            if variant == "history":
                message = message.expand(-1, modes, -1)
                logit = logit.expand(-1, modes)
            message_blocks.append(message)
            logit_blocks.append(logit)
        return self._reduce_edges(torch.cat(message_blocks), torch.cat(logit_blocks), sources,
                                  null_logits, actors)

    def forward(self, obs: Tensor, agent_mask: Tensor) -> dict[str, Tensor]:
        if obs.ndim != 4 or obs.shape[-2:] != (self.config.obs_len, 2):
            raise ValueError(f"obs must have shape [B, N, {self.config.obs_len}, 2]")
        if agent_mask.shape != obs.shape[:2] or obs.shape[0] < 1 or obs.shape[1] < 1:
            raise ValueError("agent_mask must match nonempty batch and actor axes")
        agent_mask = agent_mask.to(device=obs.device, dtype=torch.bool)
        batch, actors = obs.shape[:2]
        flat_mask = agent_mask.reshape(-1)
        # Clean padding before any geometry or attention, including NaN padding.
        safe_obs = torch.where(agent_mask[..., None, None], obs, torch.zeros_like(obs))
        frame = local_frame(safe_obs, self.scale, epsilon=self.config.epsilon, pred_len=self.config.pred_len)
        reference = frame["reference"].reshape(-1, self.config.pred_len, 2)
        history = self._encode_history(frame["local_obs"], flat_mask)
        query_features = self.query_norm(history[:, None] + self.mode_queries[None])
        initial_coefficients = self.coefficient_head(query_features)
        initial_coefficients = torch.where(flat_mask[:, None, None], initial_coefficients,
                                           torch.zeros_like(initial_coefficients))
        initial_local = self._local_predictions(initial_coefficients, reference)
        coefficients, local_prediction = initial_coefficients, initial_local
        last_message = history.new_zeros(batch * actors, self.config.num_modes, self.config.d_model)
        last_gate = history.new_zeros(batch * actors, self.config.num_modes)
        coefficient_steps, gate_steps = [coefficients], []

        if self.config.interaction != "none" and self.config.rounds > 0:
            if self.config.interaction == "self":
                sources = flat_mask.nonzero(as_tuple=True)[0]
                edges = sources, sources
            else:
                edges = neighbor_edges(frame["origin"], agent_mask, self.config.neighbors)
            historical_edge, relative_rotation = edge_frames(frame, *edges, self.scale)
            for _ in range(self.config.rounds):
                # All messages read the same complete snapshot. The new state is
                # assigned only after every edge block has been reduced.
                features = self._candidate_features(history, local_prediction, flat_mask)
                last_message, last_gate = self._interact(features, local_prediction, history,
                                                       reference, edges, historical_edge, relative_rotation)
                proposed_delta = self.refinement_head(torch.cat((features, last_message), dim=-1))
                coefficients = coefficients + last_gate[..., None] * proposed_delta
                local_prediction = self._local_predictions(coefficients, reference)
                coefficient_steps.append(coefficients)
                gate_steps.append(last_gate)

        final_features = self._candidate_features(history, local_prediction, flat_mask)
        score_features = torch.cat((history[:, None].expand(-1, self.config.num_modes, -1),
                                    final_features, last_message, coefficients), dim=-1)
        score_features = torch.where(flat_mask[:, None, None], score_features, torch.zeros_like(score_features))
        shape = (batch, actors, self.config.num_modes, self.config.pred_len, 2)
        initial_pred = to_world(initial_local.reshape(shape), frame["origin"], frame["rotation"], self.scale)
        pred = to_world(local_prediction.reshape(shape), frame["origin"], frame["rotation"], self.scale)
        output_mask = agent_mask[:, :, None, None, None]
        pred = torch.where(output_mask, pred, torch.zeros_like(pred))
        initial_pred = torch.where(output_mask, initial_pred, torch.zeros_like(initial_pred))
        coefficient_shape = (batch, actors, self.config.num_modes, self.config.rank)
        return {
            "pred": pred, "initial_pred": initial_pred,
            "score_features": score_features.reshape(batch, actors, self.config.num_modes, self.score_feature_dim),
            "gate": last_gate.reshape(batch, actors, self.config.num_modes),
            "coefficients": coefficients.reshape(coefficient_shape),
            "initial_coefficients": initial_coefficients.reshape(coefficient_shape),
            "coefficient_history": torch.stack(coefficient_steps).reshape(-1, *coefficient_shape),
            "gate_history": (torch.stack(gate_steps).reshape(-1, batch, actors, self.config.num_modes)
                             if gate_steps else last_gate.new_zeros(0, batch, actors, self.config.num_modes)),
        }
