import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class ROIAuditor:
    def __init__(self, cfg: DictConfig, artifacts_dir: Path):
        self.cfg = cfg
        self.artifacts_dir = artifacts_dir
        self.audit_results = {}

    def run_audit(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the ROI audit.
        
        Args:
            datasets: Dictionary of datasets (train, val, test).
                      
        Returns:
            Dictionary containing the path to the generated audit report.
        """
        log.info("Running ROI Audit...")
        
        # Load Manifests (preferred)
        manifests = self._load_manifests()
        
        if manifests:
            self._audit_from_manifests(manifests)
        else:
            log.warning("Manifests not found or mode is not 'manifest'. ROI audit skipped.")
            self.audit_results["note"] = "Audit skipped due to missing manifests."
            
        # Generate Reports
        self._save_json_report()
        self._save_markdown_report()
        
        return {
            "roi_audit_json": str(self.artifacts_dir / "roi_audit.json"),
            "roi_audit_md": str(self.artifacts_dir / "roi_audit.md")
        }

    def _load_manifests(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Attempts to load manifests based on config."""
        if self.cfg.data.dataset.mode != "manifest":
            return None
            
        manifest_path = Path(self.cfg.data.dataset.manifest_path)
        if not manifest_path.exists():
            return None
            
        try:
            df = pd.read_json(manifest_path)
            if "split" in df.columns:
                return {split: df[df["split"] == split] for split in df["split"].unique()}
            else:
                return {self.cfg.data.dataset.split: df}
        except Exception as e:
            log.warning(f"Failed to load manifest from {manifest_path}: {e}")
            return None

    def _audit_from_manifests(self, manifests: Dict[str, pd.DataFrame]):
        """Performs audit using pandas DataFrames."""
        
        validity_cfg = self.cfg.explainability.roi.validity_check
        jitter_cfg = self.cfg.explainability.roi.jitter_check
        
        for split, df in manifests.items():
            split_results = {
                "total_samples": len(df),
                "valid_roi_count": 0,
                "invalid_roi_count": 0,
                "issues": []
            }
            
            # Check if ROI columns exist
            # We assume 'bbox' column exists and contains [ymin, xmin, ymax, xmax] or similar
            # Or 'roi' object.
            # Let's assume 'bbox' column with list/tuple.
            
            if "bbox" not in df.columns:
                log.warning(f"No 'bbox' column in manifest for split {split}")
                split_results["note"] = "No 'bbox' column found."
                self.audit_results[split] = split_results
                continue
                
            for idx, row in df.iterrows():
                bbox = row["bbox"]
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    split_results["invalid_roi_count"] += 1
                    split_results["issues"].append(f"Invalid bbox format at index {idx}")
                    continue
                    
                ymin, xmin, ymax, xmax = bbox
                area = (ymax - ymin) * (xmax - xmin)
                
                # Validity Check
                is_valid = True
                if validity_cfg.enabled:
                    if area < validity_cfg.min_area_ratio:
                        is_valid = False
                        split_results["issues"].append(f"Small area ({area:.4f}) at index {idx}")
                    
                    if "roi_confidence" in row and row["roi_confidence"] < validity_cfg.min_confidence:
                        is_valid = False
                        split_results["issues"].append(f"Low confidence ({row['roi_confidence']:.2f}) at index {idx}")
                
                if is_valid:
                    split_results["valid_roi_count"] += 1
                else:
                    split_results["invalid_roi_count"] += 1
            
            # Jitter Check (if video_id/sequence_id exists)
            if jitter_cfg.enabled and "video_id" in df.columns:
                self._check_jitter(df, split_results, jitter_cfg.max_center_variance)
                
            self.audit_results[split] = split_results

    def _check_jitter(self, df: pd.DataFrame, results: Dict[str, Any], max_variance: float):
        """Checks for jitter in bounding boxes within video sequences."""
        jitter_issues = 0
        
        # Group by video_id
        for video_id, group in df.groupby("video_id"):
            if len(group) < 2:
                continue
                
            # Sort by timestamp or frame_index if available
            if "frame_index" in group.columns:
                group = group.sort_values("frame_index")
            
            # Calculate center points
            centers = []
            for _, row in group.iterrows():
                bbox = row["bbox"]
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    ymin, xmin, ymax, xmax = bbox
                    cy = (ymin + ymax) / 2
                    cx = (xmin + xmax) / 2
                    centers.append([cy, cx])
            
            if not centers:
                continue
                
            centers = np.array(centers)
            # Calculate variance of centers
            variance = np.var(centers, axis=0).sum()
            
            if variance > max_variance:
                jitter_issues += 1
                results["issues"].append(f"High jitter (var={variance:.2f}) in video {video_id}")
        
        results["jitter_issues_count"] = jitter_issues

    def _save_json_report(self):
        out_path = self.artifacts_dir / "roi_audit.json"
        out_path.write_text(json.dumps(self.audit_results, indent=2))

    def _save_markdown_report(self):
        out_path = self.artifacts_dir / "roi_audit.md"
        
        md = "# ROI Audit Report\n\n"
        
        for split, res in self.audit_results.items():
            if split == "note":
                md += f"**Note:** {res}\n"
                continue
                
            md += f"## {split.capitalize()}\n"
            md += f"- **Total Samples:** {res.get('total_samples', 0)}\n"
            md += f"- **Valid ROIs:** {res.get('valid_roi_count', 0)}\n"
            md += f"- **Invalid ROIs:** {res.get('invalid_roi_count', 0)}\n"
            
            if "jitter_issues_count" in res:
                md += f"- **Jitter Issues:** {res['jitter_issues_count']}\n"
            
            if res.get("issues"):
                md += "\n### Top Issues\n"
                for issue in res["issues"][:10]:
                    md += f"- {issue}\n"
                if len(res["issues"]) > 10:
                    md += f"- ... and {len(res['issues']) - 10} more.\n"
            md += "\n"
        
        out_path.write_text(md)
