import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime, timezone

from omegaconf import DictConfig

log = logging.getLogger(__name__)

def run_explainability(
    cfg: DictConfig,
    run_ctx: Dict[str, Any],
    model: Any,
    datasets: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint for the Explainability Framework.
    Orchestrates audits and explainability methods based on configuration.
    
    Args:
        cfg: Hydra configuration.
        run_ctx: Context about the current run (run_id, artifacts_dir, etc.).
        model: The trained model (Keras model).
        datasets: Dictionary of datasets (train, val, test).
        extras: Additional context (e.g., class names, preprocessing info).
        
    Returns:
        Dictionary of generated artifacts and their paths.
    """
    if not cfg.explainability.enabled:
        log.info("Explainability is disabled. Skipping.")
        return {}

    log.info("Starting Explainability Pipeline...")
    
    artifacts_dir = Path(cfg.explainability.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    generated_artifacts = {}
    
    # 1. Dataset Audit
    if cfg.explainability.dataset_audit.class_distribution.enabled or cfg.explainability.dataset_audit.leakage_check.enabled:
        from src.core.explainability.dataset_audit import DatasetAuditor
        auditor = DatasetAuditor(cfg, artifacts_dir)
        audit_res = auditor.run_audit(datasets)
        generated_artifacts.update(audit_res)

    # 2. ROI Audit
    if cfg.explainability.roi.validity_check.enabled or cfg.explainability.roi.jitter_check.enabled:
        from src.core.explainability.roi_audit import ROIAuditor
        roi_auditor = ROIAuditor(cfg, artifacts_dir)
        roi_res = roi_auditor.run_audit(datasets)
        generated_artifacts.update(roi_res)

    # 3. Model Explainability (Classification/Segmentation)
    task_type = cfg.model.type
    if task_type == "classification":
        from src.core.explainability.attribution_classification import ClassificationAttributor
        attributor = ClassificationAttributor(cfg, artifacts_dir, model)
        attr_res = attributor.run_attribution(datasets)
        generated_artifacts.update(attr_res)
        
    elif task_type == "segmentation":
        from src.core.explainability.explain_segmentation import SegmentationExplainer
        explainer = SegmentationExplainer(cfg, artifacts_dir, model)
        seg_res = explainer.run_explainability(datasets)
        generated_artifacts.update(seg_res)

    # 4. Generate Manifest
    manifest = {
        "run_id": run_ctx.get("run_id"),
        "task_name": cfg.task.name,
        "model_name": cfg.model.name,
        "dataset_id": cfg.data.get("dataset_id", "unknown"),
        "preprocess_pipeline_id": cfg.preprocess.pipeline_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": generated_artifacts,
        "config_snapshot": {
            "explainability": str(cfg.explainability)
        }
    }
    
    manifest_path = artifacts_dir / "explainability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Explainability manifest written to {manifest_path}")
    
    return {"explainability_manifest": str(manifest_path)}
