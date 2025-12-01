import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import tensorflow as tf
import json

from src.core.explainability.registry import run_explainability
from src.core.models.factories.model_factory import build_model
from src.core.data.dataset_loader import DataLoaderFactory

log = logging.getLogger(__name__)

@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    CLI Entrypoint for Explainability.
    Runs explainability on an existing run or with a fresh config.
    """
    log.info("Starting Explainability CLI...")
    
    # 1. Load Model
    # Check if we should load weights
    weights_path = cfg.get("weights_path")
    
    log.info(f"Building model: {cfg.model.name}")
    model = build_model(cfg)
    
    if weights_path and Path(weights_path).exists():
        log.info(f"Loading weights from {weights_path}")
        model.load_weights(weights_path)
    elif cfg.run.get("artifacts_dir"):
        # Try to find saved_model in artifacts
        saved_model_path = Path(cfg.run.artifacts_dir) / "saved_model"
        if saved_model_path.exists():
             log.info(f"Loading SavedModel from {saved_model_path}")
             model = tf.keras.models.load_model(saved_model_path)
    
    # 2. Load Datasets
    log.info("Loading datasets...")
    loader = DataLoaderFactory.get_loader(cfg)
    datasets = {
        "train": loader.load_train(cfg),
        "val": loader.load_val(cfg)
    }
    
    # 3. Run Explainability
    run_ctx = {
        "run_id": cfg.run.name,
        "artifacts_dir": cfg.run.artifacts_dir
    }
    
    run_explainability(cfg, run_ctx, model, datasets)

if __name__ == "__main__":
    main()
