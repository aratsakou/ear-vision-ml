from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable

import tensorflow as tf


class ModelBuilder(ABC):
    @abstractmethod
    def build(self, cfg: Any) -> tf.keras.Model:
        pass

class DataLoader(ABC):
    @abstractmethod
    def load_train(self, cfg: Any) -> tf.data.Dataset:
        pass

    @abstractmethod
    def load_val(self, cfg: Any) -> tf.data.Dataset:
        pass

class Component(ABC):
    """Interface for components with lifecycle management."""
    
    def initialize(self) -> None:
        """Called when the component is resolved/created."""
        pass
        
    def cleanup(self) -> None:
        """Called when the container is shut down."""
        pass

ModelFactoryFn = Callable[[Any], tf.keras.Model]

class Trainer(ABC):
    @abstractmethod
    def train(self, model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, cfg: Any) -> Any:
        pass

class Exporter(ABC):
    @abstractmethod
    def export(self, model: tf.keras.Model, cfg: Any, artifacts_dir: Any) -> dict[str, Any]:
        pass

class Logger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int):
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]):
        pass
