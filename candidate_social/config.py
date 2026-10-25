"""Serializable experiment configuration; paths are resolved by the CLI."""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    obs_len: int = 8
    pred_len: int = 12
    num_modes: int = 20
    rank: int = 8
    d_model: int = 128
    num_heads: int = 4
    temporal_layers: int = 2
    neighbors: int = 8
    rounds: int = 2
    interaction: str = "hierarchical"
    dropout: float = 0.0
    geometry_temperature: float = 1.0
    epsilon: float = 1e-6
    edge_chunk_size: int = 32
    # Missing in old checkpoints: retain their exact original computations.
    interaction_stability: str = "legacy"

    def __post_init__(self):
        if self.obs_len < 4 or self.pred_len < 1:
            raise ValueError("obs_len must be >= 4 and pred_len must be positive")
        if not 1 <= self.rank <= 2 * self.pred_len:
            raise ValueError("rank must be between 1 and 2 * pred_len")
        if self.num_modes < 1 or self.d_model < 1 or self.num_heads < 1:
            raise ValueError("mode, feature, and head counts must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.neighbors < 0 or self.rounds < 0 or self.temporal_layers < 1:
            raise ValueError("invalid neighbor/round/layer count")
        if self.interaction not in {"none", "self", "history", "flat", "hierarchical"}:
            raise ValueError("unknown interaction variant")
        if self.interaction_stability not in {"legacy", "normalized_v1"}:
            raise ValueError("unknown interaction stability version")
        if self.geometry_temperature <= 0 or self.epsilon <= 0 or self.edge_chunk_size < 1:
            raise ValueError("temperature, epsilon, and edge chunk size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)


@dataclass
class DataConfig:
    dataset: str = "ethucy"
    root: str = "."
    split: str = "eth"
    obs_len: int = 8
    pred_len: int = 12
    stride: int = 1
    min_agents: int = 1
    limit_train: int = 0
    limit_val: int = 0
    limit_test: int = 0
    unit: str = "m"
    coordinate_scale: float = 1.0
    val_fraction: float = 0.1
    seed: int = 42

    def to_dict(self):
        return asdict(self)
