import logging
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf

from src.core.registry import register_core_services
from src.core.di import get_container
from src.core.tuning.optuna_tuner import OptunaTuner
from src.core.training.standard_trainer import StandardTrainer
from src.core.training.component_factory import TrainingComponentFactory

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    from src.core.logging_utils import setup_logging
    setup_logging()
    
    log.info("Starting Local Hyperparameter Tuning with Optuna...")
    
    # Register services
    register_core_services(cfg)
    
    # Initialize Tuner
    tuner = OptunaTuner(storage="sqlite:///tuning.db")
    
    # Define Search Space (Example)
    # In a real scenario, this could come from a separate config file
    study_config = {
        "display_name": "otoscopic_optimization",
        "metrics": {"goal": "MAXIMIZE"},
        "parameters": {
            "learning_rate": {
                "type": "DOUBLE",
                "min_value": 1e-5,
                "max_value": 1e-2,
                "scale": "log"
            },
            "dropout": {
                "type": "DOUBLE",
                "min_value": 0.0,
                "max_value": 0.5,
                "scale": "linear"
            },
            "regularizer_l2": {
                "type": "DOUBLE",
                "min_value": 1e-6,
                "max_value": 1e-3,
                "scale": "log"
            }
        }
    }
    
    study_id = tuner.create_study(study_config)
    
    # Tuning Loop
    num_trials = 10 # Small number for demo
    
    for i in range(num_trials):
        log.info(f"\n--- Trial {i+1}/{num_trials} ---")
        
        # 1. Get Suggestions
        trials = tuner.get_suggestions(study_id, count=1)
        trial = trials[0]
        params = trial.parameters
        log.info(f"Suggested Params: {params}")
        
        # 2. Override Config
        # We need to clone and update the config
        trial_cfg = cfg.copy()
        
        # Map params to config structure
        trial_cfg.training.learning_rate = params["learning_rate"]
        trial_cfg.model.dropout = params["dropout"]
        
        # Update regularizer if present
        if "regularizer_l2" in params:
            trial_cfg.training.regularizer.enabled = True
            trial_cfg.training.regularizer.l2 = params["regularizer_l2"]
            
        # Update run name to avoid overwriting artifacts
        trial_cfg.run.name = f"{cfg.run.name}_trial_{trial.id}"
        trial_cfg.run.artifacts_dir = f"{cfg.run.artifacts_dir}/trial_{trial.id}"
        
        # 3. Train
        try:
            # Re-resolve trainer to ensure clean state if needed, 
            # but StandardTrainer is stateless mostly.
            # However, we need to ensure data loaders are fresh if they have state.
            container = get_container()
            from src.core.interfaces import Trainer
            trainer = container.resolve(Trainer)
            
            # Load Data
            from src.core.data.dataset_loader import DataLoaderFactory
            loader = DataLoaderFactory.get_loader(trial_cfg)
            train_ds = loader.load_train(trial_cfg)
            val_ds = loader.load_val(trial_cfg)
            
            # Build Model
            from src.core.models.factories.model_factory import build_model
            model = build_model(trial_cfg)
            
            # Train
            result = trainer.train(model, train_ds, val_ds, trial_cfg)
            
            # 4. Report Result
            # Assume val_accuracy is the metric
            val_acc = result.history.history.get("val_acc", [0])[-1]
            log.info(f"Trial {trial.id} Result: val_acc={val_acc}")
            
            tuner.complete_trial(trial.id, {"val_acc": val_acc})
            
        except Exception as e:
            log.error(f"Trial {trial.id} failed: {e}")
            # Mark as infeasible/failed?
            # Optuna handles exceptions if we were using the closure interface, 
            # but here we are manual. We should report failure.
            # For now, just log.
            
    log.info("Tuning Complete.")

if __name__ == "__main__":
    main()
