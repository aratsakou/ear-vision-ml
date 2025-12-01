import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import tensorflow as tf
import numpy as np
import cv2
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class ClassificationAttributor:
    def __init__(self, cfg: DictConfig, artifacts_dir: Path, model: tf.keras.Model):
        self.cfg = cfg
        self.artifacts_dir = artifacts_dir
        self.model = model
        self.output_dir = artifacts_dir / "overlays"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_attribution(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs classification attribution (Integrated Gradients).
        """
        log.info("Running Classification Attribution...")
        
        # Select samples (e.g., from validation set)
        # We need a tf.data.Dataset or similar
        dataset = datasets.get("val") or datasets.get("test") or datasets.get("train")
        if not dataset:
            log.warning("No dataset available for attribution.")
            return {}
            
        # Limit samples
        max_samples = self.cfg.explainability.max_samples
        
        # Iterate and compute
        results = []
        count = 0
        
        # Assuming dataset yields (image, label) or (image, label, metadata)
        # We need to handle different dataset structures.
        # StandardTrainer datasets usually yield (images, labels).
        
        for batch in dataset.take(max_samples):
            # Handle batch unpacking
            if isinstance(batch, tuple):
                images = batch[0]
                labels = batch[1]
            else:
                images = batch
                labels = None
                
            # If batch size > 1, iterate over batch
            batch_size = tf.shape(images)[0]
            for i in range(batch_size):
                if count >= max_samples:
                    break
                    
                img = images[i]
                label = labels[i] if labels is not None else None
                
                # Expand dims for model input
                img_input = tf.expand_dims(img, 0)
                
                # Predict to get target class if label is not provided or one-hot
                preds = self.model(img_input)
                target_class_idx = tf.argmax(preds[0]).numpy()
                
                # Compute Integrated Gradients
                heatmap = self._integrated_gradients(img_input, target_class_idx)
                
                # Save visualization
                self._save_visualization(img.numpy(), heatmap, count, target_class_idx)
                
                results.append({
                    "sample_id": count,
                    "target_class": int(target_class_idx),
                    "confidence": float(preds[0][target_class_idx]),
                    "heatmap_path": str(self.output_dir / f"heatmap_{count}.png")
                })
                
                count += 1
                
        # Save summary
        summary_path = self.artifacts_dir / "attribution_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        
        return {
            "attribution_summary_json": str(summary_path),
            "overlays_dir": str(self.output_dir)
        }

    def _integrated_gradients(self, image_tensor: tf.Tensor, target_class_idx: int, steps: int = 50) -> np.ndarray:
        """
        Computes Integrated Gradients for a single image.
        """
        # Baseline (black image)
        baseline = tf.zeros_like(image_tensor)
        
        # Generate alphas
        alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
        
        # Generate interpolated inputs
        # shape: (steps+1, H, W, C)
        interpolated_images = []
        for alpha in alphas:
            interpolated_images.append(baseline + alpha * (image_tensor - baseline))
        
        interpolated_images = tf.concat(interpolated_images, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            preds = self.model(interpolated_images)
            target_scores = preds[:, target_class_idx]
            
        grads = tape.gradient(target_scores, interpolated_images)
        
        # Approximate integral using Trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)
        
        # Compute IG
        integrated_gradients = (image_tensor - baseline) * avg_grads
        
        # Visualize: sum across channels and take absolute value
        heatmap = tf.reduce_sum(tf.abs(integrated_gradients), axis=-1)[0]
        
        # Normalize to [0, 1]
        heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap) + 1e-8)
        
        return heatmap.numpy()

    def _save_visualization(self, image: np.ndarray, heatmap: np.ndarray, sample_id: int, target_class: int):
        """Saves the heatmap overlay."""
        # Image is likely normalized [0, 1] or [-1, 1]. Convert to [0, 255] uint8.
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        # Resize heatmap to image size if needed (though IG produces same size)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        
        # Save
        out_path = self.output_dir / f"heatmap_{sample_id}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
