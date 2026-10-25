"""Differentiable geometry in per-actor coordinates; all velocities are per step."""
from __future__ import annotations

import torch
from torch import Tensor


def fit_velocity(obs: Tensor) -> Tensor:
    """Least-squares slope of the last four positions, with a fitted intercept."""
    if obs.ndim < 2 or obs.shape[-1] != 2 or obs.shape[-2] < 4:
        raise ValueError("obs must have shape [..., H >= 4, 2]")
    recent = obs[..., -4:, :]
    time = obs.new_tensor([-1.5, -0.5, 0.5, 1.5])
    centered = recent - recent.mean(dim=-2, keepdim=True)
    return (centered * time[:, None]).sum(dim=-2) / 5.0


def _broadcast_frame(points: Tensor, origin: Tensor, rotation: Tensor):
    if points.shape[-1] != 2 or origin.shape[-1] != 2:
        raise ValueError("points and origin must end in a two-coordinate axis")
    if rotation.shape != (*origin.shape[:-1], 2, 2):
        raise ValueError("rotation must match origin's leading axes")
    leading = origin.ndim - 1
    if points.ndim - 1 < leading or points.shape[:leading] != origin.shape[:-1]:
        raise ValueError("points must begin with origin's leading axes")
    extra = points.ndim - 1 - leading
    origin = origin.reshape(*origin.shape[:-1], *((1,) * extra), 2)
    rotation = rotation.reshape(*rotation.shape[:-2], *((1,) * extra), 2, 2)
    return origin, rotation


def to_local(points: Tensor, origin: Tensor, rotation: Tensor, scale) -> Tensor:
    """Convert [..., (optional mode/time axes), 2] world points to local points."""
    origin, rotation = _broadcast_frame(points, origin, rotation)
    return torch.matmul(rotation.transpose(-1, -2), (points - origin).unsqueeze(-1)).squeeze(-1) / scale


def to_world(points: Tensor, origin: Tensor, rotation: Tensor, scale) -> Tensor:
    """Inverse of to_local, with automatic mode/time-axis broadcasting."""
    origin, rotation = _broadcast_frame(points, origin, rotation)
    return origin + scale * torch.matmul(rotation, points.unsqueeze(-1)).squeeze(-1)


def local_frame(obs: Tensor, scale=1.0, epsilon: float = 1e-6, pred_len: int = 12) -> dict[str, Tensor]:
    """Use the latest position and fitted direction; low speed only resets axes."""
    if epsilon <= 0 or pred_len < 1:
        raise ValueError("epsilon and pred_len must be positive")
    scale = torch.as_tensor(scale, dtype=obs.dtype, device=obs.device)
    if scale.numel() != 1 or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("scale must be one finite positive scalar")
    origin = obs[..., -1, :]
    raw_velocity = fit_velocity(obs)
    speed = torch.linalg.vector_norm(raw_velocity, dim=-1, keepdim=True)
    fallback = torch.zeros_like(raw_velocity)
    fallback[..., 0] = 1.0
    direction = torch.where(speed > epsilon, raw_velocity / speed.clamp_min(epsilon), fallback)
    perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    rotation = torch.stack((direction, perpendicular), dim=-1)
    local_obs = to_local(obs, origin, rotation, scale)
    velocity = torch.matmul(rotation.transpose(-1, -2), raw_velocity.unsqueeze(-1)).squeeze(-1) / scale
    times = torch.arange(1, pred_len + 1, dtype=obs.dtype, device=obs.device)
    reference = velocity.unsqueeze(-2) * times[:, None]
    return {"origin": origin, "rotation": rotation, "local_obs": local_obs,
            "velocity": velocity, "raw_velocity": raw_velocity, "reference": reference}


def multi_scale_features(local_obs: Tensor) -> Tensor:
    """Position, mean displacement at spans 1/2/4, then three validity flags."""
    history = local_obs.shape[-2]
    displacements, validity = [], []
    for span in (1, 2, 4):
        delta = torch.zeros_like(local_obs)
        valid = local_obs.new_zeros(*local_obs.shape[:-1], 1)
        if history > span:
            delta[..., span:, :] = (local_obs[..., span:, :] - local_obs[..., :-span, :]) / span
            valid[..., span:, :] = 1.0
        displacements.append(delta)
        validity.append(valid)
    return torch.cat([local_obs, *displacements, *validity], dim=-1)


def neighbor_edges(origin: Tensor, agent_mask: Tensor, max_neighbors: int) -> tuple[Tensor, Tensor]:
    """Flattened source/destination indices of historical nearest-neighbor edges."""
    batch, actors, _ = origin.shape
    count = min(max_neighbors, max(actors - 1, 0))
    if count == 0 or actors == 0:
        empty = torch.empty(0, dtype=torch.long, device=origin.device)
        return empty, empty
    distances = torch.cdist(origin, origin)
    valid = agent_mask[:, :, None] & agent_mask[:, None, :]
    valid = valid & ~torch.eye(actors, dtype=torch.bool, device=origin.device)[None]
    distances = distances.masked_fill(~valid, float("inf"))
    values, neighbors = distances.topk(count, dim=-1, largest=False, sorted=True)
    batch_index, source_index, slot = torch.isfinite(values).nonzero(as_tuple=True)
    destination_index = neighbors[batch_index, source_index, slot]
    return batch_index * actors + source_index, batch_index * actors + destination_index


def edge_frames(frame: dict[str, Tensor], source: Tensor, destination: Tensor, scale) -> tuple[Tensor, Tensor]:
    """Return history6 and R_source.T @ R_neighbor for each directed edge."""
    origins = frame["origin"].reshape(-1, 2)
    rotations = frame["rotation"].reshape(-1, 2, 2)
    velocities = frame["raw_velocity"].reshape(-1, 2)
    source_inverse = rotations[source].transpose(-1, -2)
    relative_rotation = source_inverse @ rotations[destination]
    position = (source_inverse @ (origins[destination] - origins[source]).unsqueeze(-1)).squeeze(-1) / scale
    relative_velocity = (source_inverse @ (velocities[destination] - velocities[source]).unsqueeze(-1)).squeeze(-1) / scale
    heading = relative_rotation[..., :, 0]
    return torch.cat((position, relative_velocity, heading), dim=-1), relative_rotation


def pair_geometry(target_future: Tensor, neighbor_future: Tensor, history: Tensor,
                  relative_rotation: Tensor, temperature: float = 1.0) -> Tensor:
    """A2 history6 + mean7 + distance-weighted7 + endpoint7.

    Futures are [E, K_target/K_neighbor, F, 2] in their OWN actor frames.
    history[E,6] contains relative position, relative velocity and cos/sin;
    relative_rotation[E,2,2] maps neighbor-local coordinates to target-local.
    The output is [E, K_target, K_neighbor, 27].
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if target_future.ndim != 4 or neighbor_future.ndim != 4:
        raise ValueError("candidate futures must have shape [E, K, F, 2]")
    if target_future.shape[0] != neighbor_future.shape[0] or target_future.shape[-2:] != neighbor_future.shape[-2:]:
        raise ValueError("edge and future coordinate axes must match")
    edge_count, target_modes, future, _ = target_future.shape
    neighbor_modes = neighbor_future.shape[1]
    if future < 1 or history.shape != (edge_count, 6) or relative_rotation.shape != (edge_count, 2, 2):
        raise ValueError("invalid history or relative rotation shape")
    neighbor_in_target = torch.matmul(neighbor_future, relative_rotation.transpose(-1, -2)[:, None])
    relative_position = (history[:, None, None, None, :2]
                         + neighbor_in_target[:, None] - target_future[:, :, None])
    start = history[:, None, None, None, :2].expand(edge_count, target_modes, neighbor_modes, 1, 2)
    relative_step = torch.diff(relative_position, dim=-2, prepend=start)
    squared_distance = relative_position.square().sum(dim=-1)
    radial_step = (relative_position * relative_step).sum(dim=-1)
    normalized_time = torch.arange(1, future + 1, device=target_future.device, dtype=target_future.dtype) / future
    normalized_time = normalized_time.reshape(1, 1, 1, future, 1).expand(edge_count, target_modes, neighbor_modes, future, 1)
    temporal = torch.cat((relative_position, relative_step, squared_distance[..., None],
                          radial_step[..., None], normalized_time), dim=-1)
    time_weights = torch.softmax(-squared_distance / temperature, dim=-1)
    return torch.cat((history[:, None, None].expand(edge_count, target_modes, neighbor_modes, 6),
                      temporal.mean(dim=-2),
                      (time_weights[..., None] * temporal).sum(dim=-2),
                      temporal[..., -1, :]), dim=-1)
