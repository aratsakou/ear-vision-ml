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
    print(OmegaConf.to_yaml(cfg))
    
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
        
    # Export model
    if cfg.export.saved_model.enabled:
        model.export(artifacts_dir / "saved_model")
    
    if cfg.export.tflite.enabled:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(artifacts_dir / "model.tflite", "wb") as f:
            f.write(tflite_model)

    log.info("Training complete.")

if __name__ == "__main__":
    main()
