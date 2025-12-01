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
    # We wrap the static method to match the factory signature expected by Container
    def create_dataloader(cfg: Any = cfg) -> DataLoader:
        return DataLoaderFactory.get_loader(cfg)
    
    container.register_factory(DataLoader, create_dataloader)
    
    # Register Trainer factory
    # StandardTrainer will be refactored to accept dependencies, 
    # but for now we register it as is or with future dependencies
    def create_trainer() -> Trainer:
        return StandardTrainer(TrainingComponentFactory())
        
    container.register_factory(Trainer, create_trainer)
    container.register_singleton(TrainingComponentFactory, TrainingComponentFactory())

def register_task_services(cfg: Any):
    """
    Register task-specific services.
    """
    # Placeholder for task-specific registrations (e.g. specialized models, evaluators)
    pass
