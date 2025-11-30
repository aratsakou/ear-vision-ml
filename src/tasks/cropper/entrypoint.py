import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Cropper Task Entrypoint")
    log.info(f"Resolved Configuration:\n{OmegaConf.to_yaml(cfg)}")

if __name__ == "__main__":
    main()
