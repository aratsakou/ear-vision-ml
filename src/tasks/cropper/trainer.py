from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from src.core.data.dataset_loader import DataLoaderFactory
from src.core.di import get_container
from src.core.export.exporter import StandardExporter
from src.core.interfaces import Exporter, ModelBuilder, Trainer
from src.core.logging.local_logger import init_local_run, write_run_record
from src.core.models.factories.model_factory import _builder as global_model_builder
from src.core.training.standard_trainer import StandardTrainer


def configure_services() -> None:
    container = get_container()
    container.register_singleton(ModelBuilder, global_model_builder)  # type: ignore[type-abstract]
    container.register_singleton(Exporter, StandardExporter())  # type: ignore[type-abstract]
    container.register_singleton(Trainer, StandardTrainer())  # type: ignore[type-abstract]

def run_cropper(cfg: Any) -> None:
    configure_services()
    container = get_container()
    
    ctx = init_local_run(cfg.run.artifacts_dir, cfg.run.name)
    
    OmegaConf.set_struct(cfg, False)
    cfg.run.artifacts_dir = str(ctx.artifacts_dir)
    OmegaConf.set_struct(cfg, True)

    model_builder = container.resolve(ModelBuilder)  # type: ignore[type-abstract]
    trainer = container.resolve(Trainer)  # type: ignore[type-abstract]
    exporter = container.resolve(Exporter)  # type: ignore[type-abstract]
    
    data_loader = DataLoaderFactory.get_loader(cfg)
    
    model = model_builder.build(cfg)
    
    ds_train = data_loader.load_train(cfg)
    ds_val = data_loader.load_val(cfg)

    result = trainer.train(model, ds_train, ds_val, cfg)

    paths_dict = exporter.export(model, cfg, ctx.artifacts_dir)

    record = {
        "run_id": ctx.run_id,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "export": {"model_manifest": str(paths_dict.get("manifest_path"))},
        "final_metrics": {k: float(v[-1]) for k, v in result.history.history.items() if len(v) > 0},
    }
    write_run_record(ctx, record)
