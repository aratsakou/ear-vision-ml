import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.runtimes.video_inference.frame_sampler import FrameSampler
from src.runtimes.video_inference.temporal_aggregators import aggregate_predictions

# Mock model interface for now, or could use loaded Keras model
ModelFn = Callable[[np.ndarray], np.ndarray]

def run_video_inference(
    video_path: Path,
    model_fn: ModelFn,
    output_path: Path,
    sample_rate_hz: float = 1.0
) -> None:
    sampler = FrameSampler(video_path, sample_rate_hz)
    predictions = []
    
    for timestamp, frame in sampler.sample():
        # Preprocess: resize to 224x224, norm
        # In real app, use the pipeline registry
        import cv2
        frame_resized = cv2.resize(frame, (224, 224))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        batch = np.expand_dims(frame_norm, axis=0)
        
        probs = model_fn(batch)[0]
        label = int(np.argmax(probs))
        
        predictions.append({
            "timestamp": timestamp,
            "probs": probs.tolist(),
            "label": label
        })
        
    result = aggregate_predictions(predictions)
    result["frame_predictions"] = predictions
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
