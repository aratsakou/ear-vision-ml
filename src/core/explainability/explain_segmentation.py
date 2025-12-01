import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import tensorflow as tf
import numpy as np
import cv2
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class SegmentationExplainer:
    def __init__(self, cfg: DictConfig, artifacts_dir: Path, model: tf.keras.Model):
        self.cfg = cfg
        self.artifacts_dir = artifacts_dir
        self.model = model
        self.output_dir = artifacts_dir / "segmentation_maps"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_explainability(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs segmentation explainability (Uncertainty Maps).
        """
        log.info("Running Segmentation Explainability...")
        
        dataset = datasets.get("val") or datasets.get("test") or datasets.get("train")
        if not dataset:
            log.warning("No dataset available for segmentation explainability.")
            return {}
            
        max_samples = self.cfg.explainability.max_samples
        results = []
        count = 0
        
        for batch in dataset.take(max_samples):
            if isinstance(batch, tuple):
                images = batch[0]
            else:
                images = batch
                
            batch_size = tf.shape(images)[0]
            
            # Predict
            preds = self.model(images)
            
            for i in range(batch_size):
                if count >= max_samples:
                    break
                    
                img = images[i]
                pred_mask = preds[i]
                
                # Compute Uncertainty (Entropy)
                uncertainty_map = self._compute_entropy(pred_mask)
                
                # Save visualization
                self._save_visualization(img.numpy(), pred_mask.numpy(), uncertainty_map, count)
                
                # Compute metrics
                mean_uncertainty = float(np.mean(uncertainty_map))
                
                results.append({
                    "sample_id": count,
                    "mean_uncertainty": mean_uncertainty,
                    "uncertainty_map_path": str(self.output_dir / f"uncertainty_{count}.png")
                })
                
                count += 1
                
        summary_path = self.artifacts_dir / "seg_explain.json"
        summary_path.write_text(json.dumps(results, indent=2))
        
        return {
            "seg_explain_json": str(summary_path),
            "segmentation_maps_dir": str(self.output_dir)
        }

    def _compute_entropy(self, pred_mask: tf.Tensor) -> np.ndarray:
        """
        Computes entropy map from prediction mask.
        Handles binary (sigmoid) and multi-class (softmax) outputs.
        """
        # pred_mask shape: (H, W, C) or (H, W, 1)
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-7
        probs = tf.clip_by_value(pred_mask, epsilon, 1.0 - epsilon)
        
        if probs.shape[-1] == 1:
            # Binary case: p and 1-p
            p = probs
            entropy = - (p * tf.math.log(p) + (1 - p) * tf.math.log(1 - p))
            entropy = entropy[:, :, 0] # Flatten channel
        else:
            # Multi-class case
            entropy = - tf.reduce_sum(probs * tf.math.log(probs), axis=-1)
            
        return entropy.numpy()

    def _save_visualization(self, image: np.ndarray, pred_mask: np.ndarray, uncertainty: np.ndarray, sample_id: int):
        """Saves image, prediction, and uncertainty map."""
        # Normalize image
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        # Process prediction mask for visualization
        if pred_mask.shape[-1] == 1:
            mask_vis = (pred_mask[:, :, 0] * 255).astype(np.uint8)
        else:
            mask_vis = (np.argmax(pred_mask, axis=-1) * (255 // pred_mask.shape[-1])).astype(np.uint8)
            
        mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_VIRIDIS)
        
        # Process uncertainty map
        # Normalize to [0, 255]
        unc_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        unc_vis = (unc_norm * 255).astype(np.uint8)
        unc_color = cv2.applyColorMap(unc_vis, cv2.COLORMAP_INFERNO)
        
        # Combine: Image | Mask | Uncertainty
        # Ensure all have same height
        h, w, _ = image.shape
        combined = np.hstack([image, mask_color, unc_color])
        
        out_path = self.output_dir / f"uncertainty_{sample_id}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
