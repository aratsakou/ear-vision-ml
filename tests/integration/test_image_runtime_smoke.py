"""
Integration test for image inference runtime.

Tests:
- Single image inference
- Batch inference
- Test-time augmentation
- Multiple model formats
- Grad-CAM visualization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.runtimes.image_inference import (
    ImageInferenceRuntime,
    run_image_inference,
)


@pytest.fixture
def dummy_model(tmp_path: Path) -> tuple[Path, tf.keras.Model]:
    """Create a dummy classification model using Functional API."""
    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="probs")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Save as SavedModel for inference runtime
    savedmodel_path = tmp_path / "test_model"
    model.export(savedmodel_path)
    
    # Also save as .keras for Grad-CAM test
    keras_path = tmp_path / "test_model.keras"
    model.save(keras_path)
    
    return savedmodel_path, model


@pytest.fixture
def dummy_images(tmp_path: Path) -> list[Path]:
    """Create dummy test images."""
    import cv2
    
    image_paths = []
    for i in range(3):
        # Create random image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Save as JPEG
        img_path = tmp_path / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        image_paths.append(img_path)
    
    return image_paths


def test_single_image_inference(dummy_model: tuple[Path, tf.keras.Model], dummy_images: list[Path]) -> None:
    """Test single image inference."""
    model_path, _ = dummy_model
    
    runtime = ImageInferenceRuntime(
        model_path=model_path,
        model_format="saved_model",
    )
    
    result = runtime.predict(dummy_images[0])
    
    assert result.image_path == str(dummy_images[0])
    assert result.predictions.shape == (3,)  # 3 classes
    assert result.predicted_class is not None
    assert 0 <= result.predicted_class < 3
    assert 0.0 <= result.confidence <= 1.0
    assert "inference_time_ms" in result.metadata


def test_batch_inference(dummy_model: tuple[Path, tf.keras.Model], dummy_images: list[Path]) -> None:
    """Test batch inference."""
    model_path, _ = dummy_model
    
    runtime = ImageInferenceRuntime(
        model_path=model_path,
        model_format="saved_model",
    )
    
    batch_result = runtime.predict_batch(
        image_paths=dummy_images,
        batch_size=2,
        show_progress=False,
    )
    
    assert len(batch_result.results) == 3
    assert batch_result.batch_size == 2
    assert batch_result.total_time_ms > 0
    assert batch_result.avg_time_per_image_ms > 0
    
    # Check all results
    for result in batch_result.results:
        assert result.predicted_class is not None
        assert 0.0 <= result.confidence <= 1.0


def test_tta_inference(dummy_model: tuple[Path, tf.keras.Model], dummy_images: list[Path]) -> None:
    """Test test-time augmentation."""
    model_path, _ = dummy_model
    
    runtime = ImageInferenceRuntime(
        model_path=model_path,
        model_format="saved_model",
        use_tta=True,
        tta_transforms=5,
    )
    
    result = runtime.predict(dummy_images[0])
    
    assert result.metadata["tta_enabled"] is True
    assert result.predicted_class is not None


def test_run_image_inference(dummy_model: tuple[Path, tf.keras.Model], dummy_images: list[Path], tmp_path: Path) -> None:
    """Test convenience function."""
    model_path, _ = dummy_model
    output_path = tmp_path / "results.json"
    
    run_image_inference(
        model_path=model_path,
        image_paths=dummy_images,
        output_path=output_path,
        model_format="saved_model",
        use_tta=False,
        batch_size=2,
    )
    
    assert output_path.exists()
    
    # Load and verify results
    import json
    results = json.loads(output_path.read_text())
    
    assert results["total_images"] == 3
    assert results["model_format"] == "saved_model"
    assert len(results["results"]) == 3
    
    for result in results["results"]:
        assert "image_path" in result
        assert "predicted_class" in result
        assert "confidence" in result


def test_gradcam(dummy_model: tuple[Path, tf.keras.Model], dummy_images: list[Path], tmp_path: Path) -> None:
    """Test Grad-CAM visualization."""
    _, model = dummy_model
    
    # Ensure model is built and has inputs defined
    if not model.built:
        model.build((None, 64, 64, 3))
        
    # Create a dummy config
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "explainability": {
            "max_samples": 1,
            "classification": {
                "method": "grad_cam",
                "grad_cam": {
                    "layer_name": "conv1" # Use the name from dummy_model fixture
                }
            }
        }
    })
    
    from src.core.explainability.attribution_classification import ClassificationAttributor
    
    # Create dataset from dummy images
    def load_img(path):
        import cv2
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32) / 255.0
        return img
        
    images = [load_img(p) for p in dummy_images]
    ds = tf.data.Dataset.from_tensor_slices((images, np.zeros(len(images)))).batch(1)
    
    attributor = ClassificationAttributor(cfg, tmp_path, model)
    results = attributor.run_attribution({"val": ds})
    
    assert "attribution_summary_json" in results
    assert "overlays_dir" in results
    assert (tmp_path / "overlays" / "heatmap_0.png").exists()
