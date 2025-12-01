import logging
import os
from pathlib import Path
from typing import Any, Optional

import tensorflow as tf

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False

log = logging.getLogger(__name__)

class CoreMLExporter:
    """Exports TensorFlow models to Core ML format."""

    def export(self, model: tf.keras.Model, output_dir: Path, cfg: Any) -> Optional[Path]:
        """
        Exports the model to Core ML format.

        Args:
            model: The Keras model to export.
            output_dir: Directory to save the exported model.
            cfg: Configuration object.

        Returns:
            Path to the exported .mlpackage or None if export failed/skipped.
        """
        if not cfg.export.export.coreml.enabled:
            log.info("Core ML export disabled in config.")
            return None

        if not HAS_COREML:
            log.warning("coremltools not installed. Skipping Core ML export.")
            return None

        log.info("Starting Core ML export...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine model type and inputs
        task_name = cfg.task.name
        image_size = (int(cfg.data.dataset.image_size[0]), int(cfg.data.dataset.image_size[1]))
        
        # Define input type
        # Normalize 0-255 images to 0-1 range as expected by the model
        image_input = ct.ImageType(
            name="input_1",
            shape=(1, image_size[0], image_size[1], 3),
            scale=1/255.0, 
        )

        try:
            # Convert model
            # We use the SavedModel path if available, or the Keras model directly
            # Using the Keras model object directly is often easier for shape inference
            
            classifier_config = None
            if task_name == "classification":
                # For classification, we can add class labels if available
                # This is a placeholder - in a real app we'd load labels from a file
                pass

            mlmodel = ct.convert(
                model,
                inputs=[image_input],
                convert_to="mlprogram", # Use modern ML Program format
                compute_precision=ct.precision.FLOAT16 if cfg.export.export.coreml.get("quantize", False) else ct.precision.FLOAT32
            )

            # Add metadata
            mlmodel.author = "Ear Vision ML"
            mlmodel.license = "Proprietary"
            mlmodel.short_description = f"{task_name} model for ear analysis"
            mlmodel.version = "1.0.0"
            
            # Save
            save_path = output_dir / "model.mlpackage"
            mlmodel.save(str(save_path))
            
            log.info(f"Core ML model saved to {save_path}")
            return save_path

        except Exception as e:
            log.error(f"Core ML export failed: {e}")
            # Don't raise, just log error to allow other exports to proceed
            return None
