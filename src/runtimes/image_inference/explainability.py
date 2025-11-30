"""
Explainability utilities for image inference.

Features:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Attention visualization
- Saliency maps
- Feature visualization
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
    """
    
    def __init__(self, model: tf.keras.Model, layer_name: str | None = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            layer_name: Name of convolutional layer to visualize (default: last conv layer)
        """
        self.model = model
        
        # Find last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:  # Conv layer has 4D output
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self) -> tf.keras.Model:
        """Build model for computing gradients."""
        return tf.keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output,
            ],
        )
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            image: Input image (H, W, C) normalized to [0, 1]
            class_idx: Target class index (None = predicted class)
            normalize: Normalize heatmap to [0, 1]
            
        Returns:
            Heatmap array (H, W)
        """
        # Add batch dimension
        img_tensor = tf.expand_dims(image, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of class output w.r.t. conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU to keep only positive contributions
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        if normalize:
            heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, C) in [0, 1]
            heatmap: Grad-CAM heatmap (h, w)
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            Overlayed image (H, W, C) in [0, 1]
        """
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to uint8 and apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Overlay
        overlayed = alpha * heatmap_colored + (1 - alpha) * image
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed


class SaliencyMap:
    """
    Vanilla gradient saliency maps.
    
    Shows which pixels have the most influence on the prediction.
    """
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
    
    def compute(
        self,
        image: np.ndarray,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute saliency map.
        
        Args:
            image: Input image (H, W, C) in [0, 1]
            class_idx: Target class (None = predicted class)
            
        Returns:
            Saliency map (H, W)
        """
        img_tensor = tf.Variable(tf.expand_dims(image, axis=0))
        
        with tf.GradientTape() as tape:
            predictions = self.model(img_tensor)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            target_class = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(target_class, img_tensor)
        grads = tf.abs(grads)
        
        # Take maximum across color channels
        saliency = tf.reduce_max(grads[0], axis=-1)
        
        # Normalize
        saliency = saliency / (tf.reduce_max(saliency) + 1e-10)
        
        return saliency.numpy()


def save_visualization(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    title: str = "Grad-CAM Visualization",
) -> None:
    """
    Save Grad-CAM visualization.
    
    Args:
        original_image: Original image (H, W, C) in [0, 1]
        heatmap: Heatmap (H, W)
        output_path: Path to save visualization
        title: Title for the visualization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    # Overlay
    gradcam = GradCAM(None)  # Dummy for overlay method
    overlayed = gradcam.overlay_heatmap(original_image, heatmap)
    axes[2].imshow(overlayed)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
