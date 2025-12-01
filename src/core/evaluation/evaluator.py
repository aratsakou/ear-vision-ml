from typing import Any, Dict, Optional
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging

from src.core.interfaces import DataLoader
from src.core.evaluation.ab_test import ABTestAnalyzer
from src.core.training.component_factory import TrainingComponentFactory

log = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Service for evaluating models and performing A/B tests.
    """
    def __init__(self, 
                 data_loader: DataLoader, 
                 component_factory: TrainingComponentFactory):
        self.data_loader = data_loader
        self.component_factory = component_factory
        self.ab_analyzer = ABTestAnalyzer()
        log.debug(f"ModelEvaluator initialized with loader={data_loader.__class__.__name__}")

    def evaluate(self, model: tf.keras.Model, cfg: Any, split: str = "test") -> Dict[str, float]:
        """
        Evaluate model on a specific split.
        """
        log.info(f"Evaluating model on {split} set...")
        
        # Load data
        if split == "test":
            # Workaround: Check if it's ManifestDataLoader and use _load
            if hasattr(self.data_loader, "_load"):
                log.debug("Using internal _load for test set")
                ds = self.data_loader._load(cfg, "test")
            else:
                # Fallback to val if test not explicitly supported in interface
                log.warning("load_test not in interface, using load_val")
                ds = self.data_loader.load_val(cfg)
        elif split == "val":
            ds = self.data_loader.load_val(cfg)
        else:
            ds = self.data_loader.load_train(cfg)

        log.debug(f"Dataset loaded for {split}. Element spec: {ds.element_spec}")

        # Evaluate
        results = model.evaluate(ds, return_dict=True)
        log.debug(f"Evaluation results: {results}")
        return results

    def compare_models(self, 
                       baseline_model: tf.keras.Model, 
                       candidate_model: tf.keras.Model, 
                       cfg: Any,
                       metric: str = "accuracy") -> Dict[str, Any]:
        """
        Compare two models using A/B testing methodology on the test set.
        """
        log.info(f"Comparing models using metric: {metric}...")
        
        # Load test data
        if hasattr(self.data_loader, "_load"):
            ds = self.data_loader._load(cfg, "test")
        else:
            log.warning("Using validation set for comparison (test set not found)")
            ds = self.data_loader.load_val(cfg)
            
        # Get predictions
        log.info("Generating predictions for Baseline model...")
        baseline_preds = baseline_model.predict(ds)
        log.debug(f"Baseline predictions shape: {baseline_preds.shape}")
        
        log.info("Generating predictions for Candidate model...")
        candidate_preds = candidate_model.predict(ds)
        log.debug(f"Candidate predictions shape: {candidate_preds.shape}")
        
        # Get labels
        labels = []
        for _, y in ds:
            labels.append(y.numpy())
        labels = np.concatenate(labels, axis=0)
        log.debug(f"Labels shape: {labels.shape}")
        
        # Calculate metric per sample (for binary/categorical accuracy)
        # Assuming classification for now
        
        # Convert one-hot to index
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            y_true = np.argmax(labels, axis=1)
        else:
            y_true = labels.flatten()
        
        baseline_pred_cls = np.argmax(baseline_preds, axis=1)
        candidate_pred_cls = np.argmax(candidate_preds, axis=1)
        
        baseline_correct = (baseline_pred_cls == y_true).astype(int)
        candidate_correct = (candidate_pred_cls == y_true).astype(int)
        
        # Run A/B Test
        baseline_acc = np.mean(baseline_correct)
        candidate_acc = np.mean(candidate_correct)
        
        log.info(f"Baseline Accuracy: {baseline_acc:.4f}")
        log.info(f"Candidate Accuracy: {candidate_acc:.4f}")
        
        # Use ABTestAnalyzer
        ab_results = self.ab_analyzer.compare_proportions(
            control_success=int(np.sum(baseline_correct)),
            control_total=len(baseline_correct),
            treatment_success=int(np.sum(candidate_correct)),
            treatment_total=len(candidate_correct)
        )
        
        log.debug(f"A/B Test Results: {ab_results}")
        
        return {
            "baseline_metric": float(baseline_acc),
            "candidate_metric": float(candidate_acc),
            "ab_test_results": ab_results
        }
