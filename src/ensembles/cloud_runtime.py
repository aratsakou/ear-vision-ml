"""Cloud ensemble runtime.

Provides a runtime for executing ensembles of models in the cloud.
Supports soft voting (weighted averaging) of probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class EnsembleMemberSpec:
    model_path: str
    weight: float = 1.0


def soft_vote(probs_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Combine predictions using weighted soft voting.
    
    Args:
        probs_list: List of probability arrays [Batch, Classes]
        weights: List of weights for each model
        
    Returns:
        Weighted average probabilities [Batch, Classes]
    """
    w = np.asarray(weights, dtype=np.float32)
    w = w / max(1e-8, w.sum())
    stacked = np.stack(probs_list, axis=0)  # [Models, Batch, Classes]
    # Broadcast weights: [Models, 1, 1]
    return (stacked * w[:, None, None]).sum(axis=0)


class CloudEnsembleRuntime:
    """Runtime for executing model ensembles."""
    
    def __init__(self, members: list[EnsembleMemberSpec]):
        self.members = members
        self.models: list[tf.keras.Model] = []
        self._load_models()
        
    def _load_models(self) -> None:
        """Load all member models."""
        for member in self.members:
            # Load SavedModel
            # Note: In production, we might want to load TFLite for speed,
            # but for cloud runtime, SavedModel is standard.
            try:
                model = tf.keras.models.load_model(member.model_path)
                self.models.append(model)
            except Exception as e:
                raise RuntimeError(f"Failed to load ensemble member {member.model_path}: {e}")

    def predict(self, inputs: np.ndarray | tf.Tensor) -> np.ndarray:
        """
        Run inference on the ensemble.
        
        Args:
            inputs: Input tensor [Batch, H, W, C]
            
        Returns:
            Ensemble predictions [Batch, Classes]
        """
        probs_list = []
        weights = []
        
        for model, member in zip(self.models, self.members):
            # Run inference
            # Use predict_on_batch for potentially better performance in loop
            probs = model.predict_on_batch(inputs)
            probs_list.append(probs)
            weights.append(member.weight)
            
        return soft_vote(probs_list, weights)
