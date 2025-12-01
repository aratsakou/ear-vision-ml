from typing import Any

from src.core.di import get_container
from src.core.interfaces import Trainer, DataLoader
from src.core.training.standard_trainer import StandardTrainer
from src.core.training.component_factory import TrainingComponentFactory
from src.core.data.dataset_loader import DataLoaderFactory

def register_core_services(cfg: Any):
    """
    Register core services with the DI container.
    """
    container = get_container()
    
    # Register DataLoader factory
    # Now it's a proper class
    container.register_singleton(DataLoaderFactory, DataLoaderFactory())
    
    # We also register a factory for DataLoader interface that uses the factory class
    def create_dataloader(cfg: Any = cfg) -> DataLoader:
        factory = container.resolve(DataLoaderFactory)
        return factory.get_loader(cfg)
    
    container.register_factory(DataLoader, create_dataloader)
    
    # Register Trainer factory
    def create_trainer() -> Trainer:
        component_factory = container.resolve(TrainingComponentFactory)
        return StandardTrainer(component_factory)
        
    container.register_factory(Trainer, create_trainer)
    container.register_singleton(TrainingComponentFactory, TrainingComponentFactory())
    
    # Register ModelBuilder
    from src.core.models.factories.model_factory import _builder, RegistryModelBuilder
    # Register the singleton instance we already have populated with decorators
    container.register_singleton(RegistryModelBuilder, _builder)
    # Also register as the interface
    from src.core.interfaces import ModelBuilder
    container.register_singleton(ModelBuilder, _builder)

    # Register Exporter
    from src.core.interfaces import Exporter
    from src.core.export.exporter import StandardExporter
    container.register_factory(Exporter, lambda: StandardExporter())

    # Register Tuner
    from src.core.tuning.hyperparam_tuner import HyperparameterTuner
    from src.core.tuning.keras_tuner_impl import KerasTunerImpl
    
    def create_tuner() -> HyperparameterTuner:
        # We can use cfg to configure the tuner if needed
        # For now, we stick to a default directory or use one from cfg if available
        tuning_dir = "tuning_results"
        if hasattr(cfg, "tuning") and hasattr(cfg.tuning, "directory"):
            tuning_dir = cfg.tuning.directory
        return KerasTunerImpl(directory=tuning_dir)

    container.register_factory(HyperparameterTuner, create_tuner)

def register_task_services(cfg: Any):
    """
    Register task-specific services.
    """
    # Placeholder for task-specific registrations (e.g. specialized models, evaluators)
    pass
