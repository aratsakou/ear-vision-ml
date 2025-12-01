import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from pathlib import Path
import json

from src.core.di import get_container
from src.core.registry import register_core_services
from src.core.interfaces import DataLoader, ModelBuilder
from src.core.evaluation.evaluator import ModelEvaluator
from src.core.training.component_factory import TrainingComponentFactory

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Starting Segmentation Evaluation...")
    
    # Register services
    register_core_services(cfg)
    container = get_container()
    
    # Resolve dependencies
    data_loader = container.resolve(DataLoader)
    model_builder = container.resolve(ModelBuilder)
    component_factory = container.resolve(TrainingComponentFactory)
    
    # Load Test Data
    log.info("Loading test dataset...")
    test_ds = data_loader.load_test(cfg) if hasattr(data_loader, 'load_test') else data_loader.load_val(cfg)
    
    # Load Model
    model = None
    artifacts_dir = Path(cfg.run.artifacts_dir)
    saved_model_path = artifacts_dir / "saved_model"
    
    if saved_model_path.exists():
        log.info(f"Loading SavedModel from {saved_model_path}")
        model = tf.keras.models.load_model(saved_model_path)
    else:
        log.info("SavedModel not found, building model and looking for weights...")
        model = model_builder.build(cfg)
        weights_path = artifacts_dir / "checkpoints" / "best_model.keras"
        if weights_path.exists():
             log.info(f"Loading weights from {weights_path}")
             model.load_weights(weights_path)
        else:
            log.warning("No weights found! Evaluating initialized model.")

    # Evaluate
    evaluator = ModelEvaluator(data_loader, component_factory)
    metrics = evaluator.evaluate(model, test_ds, cfg)
    
    log.info(f"Evaluation Metrics: {metrics}")
    
    # Save results
    with open(artifacts_dir / "evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    log.info("Evaluation complete.")

if __name__ == "__main__":
    main()
