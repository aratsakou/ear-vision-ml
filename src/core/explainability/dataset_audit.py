import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class DatasetAuditor:
    def __init__(self, cfg: DictConfig, artifacts_dir: Path):
        self.cfg = cfg
        self.artifacts_dir = artifacts_dir
        self.audit_results = {}

    def run_audit(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the dataset audit.
        
        Args:
            datasets: Dictionary of datasets (train, val, test). 
                      Can be tf.data.Dataset or other iterables.
                      
        Returns:
            Dictionary containing the path to the generated audit report.
        """
        log.info("Running Dataset Audit...")
        
        # 1. Load Manifests if available (preferred for speed)
        manifests = self._load_manifests()
        
        if manifests:
            self._audit_from_manifests(manifests)
        else:
            log.warning("Manifests not found or mode is not 'manifest'. Falling back to dataset iteration (slow).")
            self._audit_from_datasets(datasets)
            
        # 2. Generate Reports
        self._save_json_report()
        self._save_markdown_report()
        
        return {
            "dataset_audit_json": str(self.artifacts_dir / "dataset_audit.json"),
            "dataset_audit_md": str(self.artifacts_dir / "dataset_audit.md")
        }

    def _load_manifests(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Attempts to load manifests based on config."""
        if self.cfg.data.dataset.mode != "manifest":
            return None
            
        manifest_path = Path(self.cfg.data.dataset.manifest_path)
        if not manifest_path.exists():
            return None
            
        # Assuming the manifest path points to a file that describes the split
        # OR we might have separate manifests for train/val/test.
        # The current config structure usually has one manifest path per split or a directory.
        # Let's check how the data config is structured.
        # Usually: data.dataset.manifest_path is a single file for the current split?
        # Or does it point to a directory?
        # In `local.yaml`, it's `manifest_path`.
        
        # If we are in a training job, we might only have the training manifest?
        # But for audit, we ideally want all splits to check leakage.
        # If we can't find all splits, we can only audit the current one.
        
        # For now, let's try to load what we can.
        # If the manifest is a JSON file, load it.
        try:
            df = pd.read_json(manifest_path)
            # If the manifest contains a 'split' column, we can separate them.
            if "split" in df.columns:
                return {split: df[df["split"] == split] for split in df["split"].unique()}
            else:
                # Assume it's the split defined in config
                return {self.cfg.data.dataset.split: df}
        except Exception as e:
            log.warning(f"Failed to load manifest from {manifest_path}: {e}")
            return None

    def _audit_from_manifests(self, manifests: Dict[str, pd.DataFrame]):
        """Performs audit using pandas DataFrames."""
        
        # Class Distribution
        if self.cfg.explainability.dataset_audit.class_distribution.enabled:
            dist = {}
            for split, df in manifests.items():
                if "label" in df.columns:
                    counts = df["label"].value_counts().to_dict()
                    dist[split] = counts
                else:
                    log.warning(f"Label column not found in manifest for split {split}")
            self.audit_results["class_distribution"] = dist

        # Leakage Check
        if self.cfg.explainability.dataset_audit.leakage_check.enabled:
            self._check_leakage(manifests)

    def _audit_from_datasets(self, datasets: Dict[str, Any]):
        """Performs audit by iterating over tf.data.Datasets."""
        # This is a fallback and can be slow.
        # We'll implement a simplified version that just counts labels if possible.
        # For now, we'll skip deep iteration to avoid stalling training.
        log.warning("Audit from datasets not fully implemented to avoid performance impact.")
        self.audit_results["note"] = "Audit skipped due to missing manifests."

    def _check_leakage(self, manifests: Dict[str, pd.DataFrame]):
        """Checks for overlap between splits."""
        splits = list(manifests.keys())
        leakage = {}
        
        match_on = self.cfg.explainability.dataset_audit.leakage_check.match_on
        # Default to 'image_uri' or 'file_path'
        key = "image_uri" if "image_uri" in manifests[splits[0]].columns else "file_path"
        
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                split_a = splits[i]
                split_b = splits[j]
                
                set_a = set(manifests[split_a][key].values)
                set_b = set(manifests[split_b][key].values)
                
                overlap = set_a.intersection(set_b)
                if overlap:
                    leakage[f"{split_a}_vs_{split_b}"] = {
                        "count": len(overlap),
                        "examples": list(overlap)[:5]
                    }
        
        self.audit_results["leakage"] = leakage

    def _save_json_report(self):
        out_path = self.artifacts_dir / "dataset_audit.json"
        out_path.write_text(json.dumps(self.audit_results, indent=2))

    def _save_markdown_report(self):
        out_path = self.artifacts_dir / "dataset_audit.md"
        
        md = "# Dataset Audit Report\n\n"
        
        # Class Distribution
        if "class_distribution" in self.audit_results:
            md += "## Class Distribution\n\n"
            for split, counts in self.audit_results["class_distribution"].items():
                md += f"### {split.capitalize()}\n"
                md += "| Class | Count |\n|---|---|\n"
                for label, count in counts.items():
                    md += f"| {label} | {count} |\n"
                md += "\n"
        
        # Leakage
        if "leakage" in self.audit_results:
            md += "## Data Leakage\n\n"
            if not self.audit_results["leakage"]:
                md += "✅ No leakage detected between splits.\n"
            else:
                md += "⚠️ **Leakage Detected!**\n\n"
                for pair, info in self.audit_results["leakage"].items():
                    md += f"- **{pair}**: {info['count']} overlapping samples.\n"
                    md += f"  - Examples: {', '.join(map(str, info['examples']))}\n"
        
        out_path.write_text(md)
