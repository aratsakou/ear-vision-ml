import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.core.monitoring.drift_detector import DriftDetector

log = logging.getLogger(__name__)

@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Monitoring Entrypoint.
    Detects drift between a baseline dataset (from training) and a target dataset (e.g. production/new batch).
    """
    log.info(f"Starting monitoring task for {cfg.task.name}")
    
    # 1. Load Baseline Stats
    # In a real scenario, we'd load this from the model manifest or dataset stats.json
    # Here we assume cfg.monitoring.baseline_stats_path is provided
    baseline_path = Path(cfg.monitoring.baseline_stats_path)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline stats not found at {baseline_path}")
    
    log.info(f"Loading baseline stats from {baseline_path}")
    baseline_stats = json.loads(baseline_path.read_text())
    
    # Extract baseline histograms/stats to reconstruct distribution approximation or use stats directly
    # For KS-test we need raw data, but we only have stats. 
    # DriftDetector.detect_drift expects raw arrays.
    # If we only have histograms, we can only compute PSI.
    # For this MVP, let's assume we are comparing two datasets by loading them, 
    # OR we adapt DriftDetector to work with histograms for PSI.
    
    # Let's assume we load the target dataset
    target_data_path = Path(cfg.monitoring.target_data_path)
    log.info(f"Loading target data from {target_data_path}")
    # Assume parquet for now
    df = pd.read_parquet(target_data_path)
    
    # We need raw baseline data for KS-test. 
    # If we don't have it, we can only do PSI if we implemented histogram-based PSI.
    # But our DriftDetector currently takes raw arrays.
    # To support the "stats-only" baseline, we would need to refactor.
    # For MVP, let's assume we point to the baseline dataset PARQUET file as well.
    
    baseline_data_path = Path(cfg.monitoring.baseline_data_path)
    log.info(f"Loading baseline data from {baseline_data_path}")
    baseline_df = pd.read_parquet(baseline_data_path)
    
    # Select features
    features = cfg.monitoring.features
    
    baseline_dict = {f: baseline_df[f].values for f in features if f in baseline_df.columns}
    current_dict = {f: df[f].values for f in features if f in df.columns}
    
    # Detect Drift
    detector = DriftDetector()
    results = detector.detect_drift(baseline_dict, current_dict)
    
    # Log results
    log.info("Drift Detection Results:")
    log.info(json.dumps(results, indent=2))
    
    # Save results
    out_dir = Path(cfg.run.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drift_report.json").write_text(json.dumps(results, indent=2))
    
    # Alerting (Simplified)
    drifted_features = [f for f, r in results.items() if r["drift_detected"]]
    if drifted_features:
        log.warning(f"DRIFT DETECTED in features: {drifted_features}")
        # In production, send alert to Slack/PagerDuty
    else:
        log.info("No significant drift detected.")

if __name__ == "__main__":
    main()
