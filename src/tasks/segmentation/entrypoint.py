import logging
import tensorflow as tf
import hydra
from omegaconf import DictConfig, OmegaConf

from src.core.training.standard_trainer import StandardTrainer
from src.core.models.factories.model_factory import build_model
from src.core.data.dataset_loader import DataLoaderFactory

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Segmentation Task Entrypoint")
    log.info(f"Resolved Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # 1. Build Model
    log.info(f"Building model: {cfg.model.name}")
    model = build_model(cfg)
    
    # 2. Load Data
    log.info("Loading datasets...")
    loader = DataLoaderFactory.get_loader(cfg)
    train_ds = loader.load_train(cfg)
    val_ds = loader.load_val(cfg)
    
    # 3. Train
    log.info("Starting training...")
    trainer = StandardTrainer()
    history = trainer.train(model, train_ds, val_ds, cfg)
    
    log.info("Training complete.")

if __name__ == "__main__":
    main()
