from __future__ import annotations

import tensorflow as tf


def classification_loss() -> tf.keras.losses.Loss:
    """Standard categorical cross-entropy for classification."""
    return tf.keras.losses.CategoricalCrossentropy()


def focal_loss(alpha: float = 0.25, gamma: float = 2.0) -> tf.keras.losses.Loss:
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for rare classes
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    def focal_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        focal = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    
    return focal_loss_fn


def label_smoothing_loss(smoothing: float = 0.1) -> tf.keras.losses.Loss:
    """
    Label smoothing to prevent overconfidence.
    
    Converts hard labels [0, 1, 0] to soft labels [0.05, 0.9, 0.05]
    """
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)


def segmentation_loss() -> tf.keras.losses.Loss:
    """Standard categorical cross-entropy for segmentation."""
    return tf.keras.losses.CategoricalCrossentropy()


def dice_loss() -> tf.keras.losses.Loss:
    """
    Dice Loss for segmentation - better for imbalanced classes.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    def dice_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        smooth = 1e-6
        
        # Flatten spatial dimensions
        y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - tf.reduce_mean(dice)
    
    return dice_loss_fn


def combined_dice_ce_loss(dice_weight: float = 0.5) -> tf.keras.losses.Loss:
    """
    Combined Dice + Cross-Entropy loss for segmentation.
    
    Combines the benefits of both losses:
    - CE: Good gradients, class-wise optimization
    - Dice: Handles class imbalance, region-based
    """
    ce_loss = tf.keras.losses.CategoricalCrossentropy()
    dice_fn = dice_loss()
    
    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ce = ce_loss(y_true, y_pred)
        dice = dice_fn(y_true, y_pred)
        return dice_weight * dice + (1 - dice_weight) * ce
    
    return combined_loss


def tversky_loss(alpha: float = 0.7, beta: float = 0.3) -> tf.keras.losses.Loss:
    """
    Tversky Loss - generalization of Dice loss with control over FP/FN.
    
    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
        
    When alpha=beta=0.5, reduces to Dice loss.
    Use alpha > beta to penalize false negatives more (better recall).
    """
    def tversky_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        smooth = 1e-6
        
        y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        true_pos = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        false_neg = tf.reduce_sum(y_true_f * (1 - y_pred_f), axis=0)
        false_pos = tf.reduce_sum((1 - y_true_f) * y_pred_f, axis=0)
        
        tversky = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
        return 1.0 - tf.reduce_mean(tversky)
    
    return tversky_loss_fn


def cropper_loss() -> tf.keras.losses.Loss:
    """
    Smooth L1 loss for bounding box regression.
    
    More robust to outliers than MSE.
    """
    return tf.keras.losses.Huber(delta=1.0)


def iou_loss() -> tf.keras.losses.Loss:
    """
    IoU (Intersection over Union) loss for bbox regression.
    
    Directly optimizes the IoU metric.
    """
    def iou_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # y_true, y_pred shape: [batch, 5] where last dim is [x1, y1, x2, y2, conf]
        # Extract bbox coordinates
        true_bbox = y_true[..., :4]
        pred_bbox = y_pred[..., :4]
        
        # Calculate intersection
        x1_inter = tf.maximum(true_bbox[..., 0], pred_bbox[..., 0])
        y1_inter = tf.maximum(true_bbox[..., 1], pred_bbox[..., 1])
        x2_inter = tf.minimum(true_bbox[..., 2], pred_bbox[..., 2])
        y2_inter = tf.minimum(true_bbox[..., 3], pred_bbox[..., 3])
        
        inter_area = tf.maximum(0.0, x2_inter - x1_inter) * tf.maximum(0.0, y2_inter - y1_inter)
        
        # Calculate union
        true_area = (true_bbox[..., 2] - true_bbox[..., 0]) * (true_bbox[..., 3] - true_bbox[..., 1])
        pred_area = (pred_bbox[..., 2] - pred_bbox[..., 0]) * (pred_bbox[..., 3] - pred_bbox[..., 1])
        union_area = true_area + pred_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        
        # IoU loss
        return 1.0 - tf.reduce_mean(iou)
    
    return iou_loss_fn


def get_loss(loss_name: str, **kwargs) -> tf.keras.losses.Loss:
    """
    Factory function to get loss by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss
        
    Returns:
        Loss function
    """
    losses = {
        "categorical_crossentropy": classification_loss,
        "focal": lambda: focal_loss(**kwargs),
        "label_smoothing": lambda: label_smoothing_loss(**kwargs),
        "dice": dice_loss,
        "dice_ce": lambda: combined_dice_ce_loss(**kwargs),
        "tversky": lambda: tversky_loss(**kwargs),
        "huber": cropper_loss,
        "iou": iou_loss,
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name]()
