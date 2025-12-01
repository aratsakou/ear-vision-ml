from typing import Any, Dict, Optional
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.core.interfaces import DataLoader
from src.core.evaluation.ab_test import ABTestAnalyzer
from src.core.training.component_factory import TrainingComponentFactory

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

    def evaluate(self, model: tf.keras.Model, cfg: Any, split: str = "test") -> Dict[str, float]:
        """
        Evaluate model on a specific split.
        """
        print(f"Evaluating model on {split} set...")
        
        # Load data
        if split == "test":
            # We assume DataLoader has load_test or we use _load directly if exposed
            # For now, let's assume we can use the internal _load if available or fallback
            # ManifestDataLoader has _load but it's protected. 
            # Let's check if we can add load_test to DataLoader interface or use a workaround.
            # For this implementation, we'll try to use a public method or assume one exists.
            # If not, we might need to extend DataLoader interface.
            
            # Workaround: Check if it's ManifestDataLoader and use _load
            if hasattr(self.data_loader, "_load"):
                ds = self.data_loader._load(cfg, "test")
            else:
                # Fallback to val if test not explicitly supported in interface
                print("Warning: load_test not in interface, using load_val")
                ds = self.data_loader.load_val(cfg)
        elif split == "val":
            ds = self.data_loader.load_val(cfg)
        else:
            ds = self.data_loader.load_train(cfg)

        # Evaluate
        results = model.evaluate(ds, return_dict=True)
        return results

    def compare_models(self, 
                       baseline_model: tf.keras.Model, 
                       candidate_model: tf.keras.Model, 
                       cfg: Any,
                       metric: str = "accuracy") -> Dict[str, Any]:
        """
        Compare two models using A/B testing methodology on the test set.
        """
        print(f"Comparing models using metric: {metric}...")
        
        # Load test data
        # We need raw predictions for A/B testing, not just aggregated metrics
        # So we iterate through the dataset
        
        if hasattr(self.data_loader, "_load"):
            ds = self.data_loader._load(cfg, "test")
        else:
            ds = self.data_loader.load_val(cfg)
            
        # Get predictions
        print("Generating predictions for Baseline model...")
        baseline_preds = baseline_model.predict(ds)
        print("Generating predictions for Candidate model...")
        candidate_preds = candidate_model.predict(ds)
        
        # Get labels
        # We need to extract labels from the dataset
        labels = []
        for _, y in ds:
            labels.append(y.numpy())
        labels = np.concatenate(labels, axis=0)
        
        # Calculate metric per sample (for binary/categorical accuracy)
        # Assuming classification for now
        
        # Convert one-hot to index
        y_true = np.argmax(labels, axis=1)
        
        baseline_pred_cls = np.argmax(baseline_preds, axis=1)
        candidate_pred_cls = np.argmax(candidate_preds, axis=1)
        
        baseline_correct = (baseline_pred_cls == y_true).astype(int)
        candidate_correct = (candidate_pred_cls == y_true).astype(int)
        
        # Run A/B Test
        # We compare the "success" (correct prediction) rates
        
        baseline_acc = np.mean(baseline_correct)
        candidate_acc = np.mean(candidate_correct)
        
        print(f"Baseline Accuracy: {baseline_acc:.4f}")
        print(f"Candidate Accuracy: {candidate_acc:.4f}")
        
        # Use ABTestAnalyzer
        # We treat this as comparing proportions of correct predictions
        ab_results = self.ab_analyzer.compare_proportions(
            control_success=int(np.sum(baseline_correct)),
            control_total=len(baseline_correct),
            treatment_success=int(np.sum(candidate_correct)),
            treatment_total=len(candidate_correct)
        )
        
        return {
            "baseline_metric": float(baseline_acc),
            "candidate_metric": float(candidate_acc),
            "ab_test_results": ab_results
        }
