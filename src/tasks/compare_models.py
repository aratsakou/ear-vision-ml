import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from pathlib import Path
import json
import logging

from src.core.di import get_container
from src.core.registry import register_core_services
from src.core.interfaces import DataLoader
from src.core.evaluation.evaluator import ModelEvaluator
from src.core.training.component_factory import TrainingComponentFactory
from src.core.logging_utils import setup_logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    
    log.info("Comparing Models...")
    log.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Register services
    register_core_services(cfg)
    container = get_container()
    
    # Resolve dependencies
    data_loader = container.resolve(DataLoader)
    
    # Create Evaluator
    component_factory = container.resolve(TrainingComponentFactory)
    evaluator = ModelEvaluator(data_loader, component_factory)
    
    # Load Models
    baseline_path = cfg.get("baseline_model_path")
    candidate_path = cfg.get("candidate_model_path")
    
    if not baseline_path or not candidate_path:
        log.error("Both baseline_model_path and candidate_model_path must be provided")
        raise ValueError("Both baseline_model_path and candidate_model_path must be provided")
        
    log.info(f"Loading Baseline: {baseline_path}")
    baseline_model = tf.keras.models.load_model(baseline_path)
    
    log.info(f"Loading Candidate: {candidate_path}")
    candidate_model = tf.keras.models.load_model(candidate_path)
    
    # Run Comparison
    results = evaluator.compare_models(baseline_model, candidate_model, cfg)
    
    # Print Results
    log.info("\n" + "="*40)
    log.info("COMPARISON RESULTS")
    log.info("="*40)
    log.info(f"Baseline Accuracy:  {results['baseline_metric']:.4f}")
    log.info(f"Candidate Accuracy: {results['candidate_metric']:.4f}")
    log.info("-" * 40)
    
    ab = results['ab_test_results']
    log.info(f"Lift: {ab['lift_percent']:.2f}%")
    log.info(f"P-Value: {ab['p_value']:.4f}")
    log.info(f"Significant (p<0.05): {ab['significant']}")
    
    if ab['significant']:
        if ab['lift_percent'] > 0:
            log.info("✅ Candidate is SIGNIFICANTLY BETTER")
        else:
            log.info("❌ Candidate is SIGNIFICANTLY WORSE")
    else:
        log.info("⚠️ No significant difference detected")
        
    # Save results
    output_dir = Path(cfg.run.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {output_dir / 'comparison_results.json'}")

if __name__ == "__main__":
    main()
