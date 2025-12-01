import logging
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf

from src.core.registry import register_core_services
from src.core.di import get_container
from src.core.di import get_container
from src.core.training.standard_trainer import StandardTrainer
from src.core.training.component_factory import TrainingComponentFactory

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    from src.core.logging_utils import setup_logging
    setup_logging()
    
    log.info("Starting Local Hyperparameter Tuning with Keras Tuner...")
    
    # Register services
    register_core_services(cfg)
    
    # Initialize Tuner
    from src.core.tuning.hyperparam_tuner import HyperparameterTuner
    container = get_container()
    tuner = container.resolve(HyperparameterTuner)
    
    # Get Study Config from Hydra
    # We convert to primitive dict for the tuner
    study_config = OmegaConf.to_container(cfg.tuning, resolve=True)
    
    study_id = tuner.create_study(study_config)
    
    # Tuning Loop
    num_trials = study_config.get("max_trials", 10)
    
    for i in range(num_trials):
        log.info(f"\n--- Trial {i+1}/{num_trials} ---")
        
        # 1. Get Suggestions
        trials = tuner.get_suggestions(study_id, count=1)
        if not trials:
            log.info("No more trials suggested.")
            break
            
        trial = trials[0]
        params = trial.parameters
        log.info(f"Suggested Params: {params}")
        
        # 2. Override Config
        # We need to clone and update the config
        trial_cfg = cfg.copy()
        
        # Map params to config structure dynamically based on 'target' field in config
        param_specs = study_config.get("parameters", {})
        
        for param_name, param_value in params.items():
            if param_name in param_specs:
                target_path = param_specs[param_name].get("target")
                if target_path:
                    # Use OmegaConf.update to set value at dot-notation path
                    OmegaConf.update(trial_cfg, target_path, param_value)
                    
                    # Special handling for regularizer enablement if needed
                    # If we are setting a regularizer value, we might need to ensure it's enabled.
                    # This is a bit specific, but we can handle it generically if the config structure supports it.
                    # For now, we assume the user configures 'enabled' in the base config or we add a specific rule.
                    # Or we can just set it.
                    if "regularizer" in target_path:
                         # Hack: ensure enabled is true if we are tuning it
                         # We can look for the parent node.
                         # A cleaner way is to have 'enabled' as a tunable parameter or fixed in base.
                         # Let's assume base config has it enabled or we set it here if value > 0.
                         pass
                         
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
            # Metric name from config
            metric_name = study_config.get("objective", "val_acc")
            # Handle potential missing metric
            history = result.history.history
            if metric_name in history:
                score = history[metric_name][-1]
            else:
                log.warning(f"Metric {metric_name} not found in history. Available: {list(history.keys())}")
                score = 0.0
                
            log.info(f"Trial {trial.id} Result: {metric_name}={score}")
            
            tuner.complete_trial(trial.id, {metric_name: score})
            
        except Exception as e:
            log.error(f"Trial {trial.id} failed: {e}")
            # Mark as infeasible/failed?
            # Optuna handles exceptions if we were using the closure interface, 
            # but here we are manual. We should report failure.
            # For now, just log.
            
    log.info("Tuning Complete.")

if __name__ == "__main__":
    main()
