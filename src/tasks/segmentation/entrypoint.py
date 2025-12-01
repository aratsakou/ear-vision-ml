import logging
import tensorflow as tf
import hydra
from omegaconf import DictConfig, OmegaConf

from src.core.di import get_container
from src.core.registry import register_core_services
from src.core.interfaces import Trainer, DataLoader, ModelBuilder

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.core.logging_utils import setup_logging
    setup_logging()
    
    log.info(f"Resolved Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Register services
    register_core_services(cfg)
    container = get_container()
    
    # Resolve dependencies
    data_loader = container.resolve(DataLoader)
    trainer = container.resolve(Trainer)
    
    # Load data
    train_ds = data_loader.load_train(cfg)
    val_ds = data_loader.load_val(cfg)
    
    # Build model
    model_builder = container.resolve(ModelBuilder)
    model = model_builder.build(cfg)
    
    history = trainer.train(model, train_ds, val_ds, cfg)
    
    # Save artifacts
    import json
    from pathlib import Path
    
    artifacts_dir = Path(cfg.run.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run metadata
    run_meta = {
        "run_id": cfg.run.name,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "final_metrics": {k: v[-1] for k, v in history.history.history.items()}
    }
    with open(artifacts_dir / "run.json", "w") as f:
        json.dump(run_meta, f, indent=2)
        
    # Save metrics
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(history.history.history, f, indent=2)
        
    # Export model using DI
    from src.core.interfaces import Exporter
    exporter = container.resolve(Exporter)
    exporter.export(model, cfg, artifacts_dir)
    
    log.info("Training complete.")

if __name__ == "__main__":
    main()
