"""Portable CandidateSocial model and evaluation release."""

from .config import DataConfig, ModelConfig
from .model import CandidateSocialModel

__version__ = "1.0.0"
__all__ = ["CandidateSocialModel", "DataConfig", "ModelConfig"]
