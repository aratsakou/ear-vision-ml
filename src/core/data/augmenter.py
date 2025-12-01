from abc import ABC, abstractmethod
from typing import Any, Optional
import tensorflow as tf

from src.core.data.augmentations import get_augmentation_pipeline

class Augmenter(ABC):
    """
    Interface for data augmentation strategies.
    """
    @abstractmethod
    def augment(self, image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        pass

class NoOpAugmenter(Augmenter):
    """
    Augmenter that does nothing.
    """
    def augment(self, image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return image, label

class ConfigurableAugmenter(Augmenter):
    """
    Augmenter that uses the existing augmentations.py library.
    """
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.aug_fn = None
        
        # Parse config to get augmentation parameters
        if cfg.get("augmentation", {}).get("enabled", False):
            aug_type = cfg.augmentation.get("type", "standard")
            params = cfg.augmentation.get("params", {})
            self.aug_fn = get_augmentation_pipeline(augmentation_type=aug_type, **params)

    def augment(self, image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if self.aug_fn:
            # The existing augmentations.py functions mostly take just the image
            # except MixUp/CutMix which take (image1, label1, image2, label2)
            # For standard/medical/randaugment, we just transform the image
            
            # Check if it's a dual-input augmentation (MixUp/CutMix)
            # This requires a different integration strategy (dataset.zip)
            # For now, we assume single-image augmentation for this class
            # and handle MixUp/CutMix separately or assume the wrapped fn handles it if adapted
            
            # Current augmentations.py 'standard', 'medical', 'randaugment' return image only
            image = self.aug_fn(image)
            
        return image, label
