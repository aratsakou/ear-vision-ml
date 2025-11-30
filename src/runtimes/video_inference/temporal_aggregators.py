from typing import Any

import numpy as np


def aggregate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregates per-frame predictions.
    
    Args:
        predictions: List of dicts, each containing 'timestamp', 'probs', 'label'.
        
    Returns:
        Aggregated result.
    """
    if not predictions:
        return {"final_label": None, "confidence": 0.0}
    
    # Simple averaging of probabilities
    all_probs = np.array([p['probs'] for p in predictions])
    mean_probs = np.mean(all_probs, axis=0)
    final_label_idx = np.argmax(mean_probs)
    confidence = float(mean_probs[final_label_idx])
    
    return {
        "final_label_index": int(final_label_idx),
        "confidence": confidence,
        "num_frames": len(predictions),
        "mean_probs": mean_probs.tolist()
    }
