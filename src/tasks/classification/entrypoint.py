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
    
    # 4. Export (Optional - handled by callbacks usually, but we can do explicit export if needed)
    # The StandardTrainer handles callbacks which save the model.
    
    log.info("Training complete.")

if __name__ == "__main__":
    main()
