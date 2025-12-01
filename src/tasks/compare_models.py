import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from pathlib import Path
import json

from src.core.di import get_container
from src.core.registry import register_core_services
from src.core.interfaces import DataLoader
from src.core.evaluation.evaluator import ModelEvaluator
from src.core.training.component_factory import TrainingComponentFactory

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Comparing Models...")
    
    # Register services
    register_core_services(cfg)
    container = get_container()
    
    # Resolve dependencies
    # Note: We might need a specialized DataLoader if we want to ensure test set loading
    # but for now we use the standard one
    data_loader = container.resolve(DataLoader)
    
    # Create Evaluator
    # We can also register Evaluator in DI, but for now manual instantiation is fine
    # as it's a specific task service
    component_factory = container.resolve(TrainingComponentFactory)
    evaluator = ModelEvaluator(data_loader, component_factory)
    
    # Load Models
    baseline_path = cfg.get("baseline_model_path")
    candidate_path = cfg.get("candidate_model_path")
    
    if not baseline_path or not candidate_path:
        raise ValueError("Both baseline_model_path and candidate_model_path must be provided")
        
    print(f"Loading Baseline: {baseline_path}")
    baseline_model = tf.keras.models.load_model(baseline_path)
    
    print(f"Loading Candidate: {candidate_path}")
    candidate_model = tf.keras.models.load_model(candidate_path)
    
    # Run Comparison
    results = evaluator.compare_models(baseline_model, candidate_model, cfg)
    
    # Print Results
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"Baseline Accuracy:  {results['baseline_metric']:.4f}")
    print(f"Candidate Accuracy: {results['candidate_metric']:.4f}")
    print("-" * 40)
    
    ab = results['ab_test_results']
    print(f"Lift: {ab['lift_percent']:.2f}%")
    print(f"P-Value: {ab['p_value']:.4f}")
    print(f"Significant (p<0.05): {ab['significant']}")
    
    if ab['significant']:
        if ab['lift_percent'] > 0:
            print("✅ Candidate is SIGNIFICANTLY BETTER")
        else:
            print("❌ Candidate is SIGNIFICANTLY WORSE")
    else:
        print("⚠️ No significant difference detected")
        
    # Save results
    output_dir = Path(cfg.run.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'comparison_results.json'}")

if __name__ == "__main__":
    main()
