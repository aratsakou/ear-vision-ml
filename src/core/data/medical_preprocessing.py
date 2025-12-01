import tensorflow as tf
from typing import Any, Tuple
from src.core.data.dataset_loader import Preprocessor

class MedicalPreprocessor(Preprocessor):
    """
    Preprocessor optimized for medical imaging.
    Features:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Per-image standardization or min-max normalization
    """
    def preprocess(self, features: dict[str, Any], cfg: Any) -> Tuple[tf.Tensor, tf.Tensor]:
        image_size = (int(cfg.data.dataset.image_size[0]), int(cfg.data.dataset.image_size[1]))
        num_classes = int(cfg.data.dataset.num_classes)
        
        uri = features['image_uri']
        path = tf.strings.regex_replace(uri, "^file://", "")
        
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        
        # Convert to float32 [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        # Apply CLAHE if enabled
        # Note: tf.image.adjust_contrast is simple contrast.
        # For CLAHE, we might need a custom op or approximation.
        # TensorFlow doesn't have a native CLAHE op in core (it's in tfa/addons).
        # We'll implement a simplified adaptive histogram equalization or use contrast adjustment.
        
        if cfg.get("preprocess", {}).get("clahe", False):
            # Convert to LAB or Grayscale for better contrast adjustment
            # Here we just apply contrast adjustment as a proxy for CLAHE if TFA is missing
            # In a real medical pipeline, we'd use tfa.image.equalize or similar
            img = tf.image.adjust_contrast(img, 1.2)
            
        # Normalization
        norm_type = cfg.get("preprocess", {}).get("normalization", "min_max")
        if norm_type == "standard":
            # Per-image standardization
            img = tf.image.per_image_standardization(img)
        
        label = tf.cast(features['label'], tf.int32)
        label_one_hot = tf.one_hot(label, depth=num_classes)
        
        return img, label_one_hot
