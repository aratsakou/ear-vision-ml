import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Cropper Task Entrypoint")
    log.info(f"Resolved Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Register services
    from src.core.registry import register_core_services
    from src.core.di import get_container
    from src.core.interfaces import Trainer, DataLoader, ModelBuilder, Exporter
    
    register_core_services(cfg)
    container = get_container()
    
    # Resolve dependencies (Example usage)
    # data_loader = container.resolve(DataLoader)
    # trainer = container.resolve(Trainer)
    # model_builder = container.resolve(ModelBuilder)
    # exporter = container.resolve(Exporter)
    
    log.info("DI Container initialized and services registered.")

if __name__ == "__main__":
    main()
