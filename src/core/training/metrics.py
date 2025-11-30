from __future__ import annotations

import tensorflow as tf


def classification_metrics() -> list[tf.keras.metrics.Metric]:
    """Standard classification metrics."""
    return [
        tf.keras.metrics.CategoricalAccuracy(name="acc"),
        tf.keras.metrics.AUC(name="auc", multi_label=True),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]


class F1Score(tf.keras.metrics.Metric):
    """F1 Score metric (harmonic mean of precision and recall)."""
    
    def __init__(self, name: str = "f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


class DiceCoefficient(tf.keras.metrics.Metric):
    """Dice coefficient for segmentation (1 - dice_loss)."""
    
    def __init__(self, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        smooth = 1e-6
        
        y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        self.dice_sum.assign_add(tf.reduce_mean(dice))
        self.count.assign_add(1.0)
    
    def result(self):
        return self.dice_sum / (self.count + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)


class IoU(tf.keras.metrics.Metric):
    """Intersection over Union (Jaccard Index) for segmentation."""
    
    def __init__(self, name: str = "iou", num_classes: int = 2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou_sum = self.add_weight(name="iou_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        smooth = 1e-6
        
        y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        self.iou_sum.assign_add(tf.reduce_mean(iou))
        self.count.assign_add(1.0)
    
    def result(self):
        return self.iou_sum / (self.count + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)


class BBoxIoU(tf.keras.metrics.Metric):
    """IoU metric for bounding box regression."""
    
    def __init__(self, name: str = "bbox_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(name="iou_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract bbox coordinates [x1, y1, x2, y2, conf]
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
        
        self.iou_sum.assign_add(tf.reduce_mean(iou))
        self.count.assign_add(1.0)
    
    def result(self):
        return self.iou_sum / (self.count + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)


def segmentation_metrics() -> list[tf.keras.metrics.Metric]:
    """
    Comprehensive segmentation metrics.
    
    Returns:
        - Pixel accuracy (standard)
        - Dice coefficient (region overlap)
        - IoU / Jaccard index (region overlap)
    """
    return [
        tf.keras.metrics.CategoricalAccuracy(name="pixel_acc"),
        DiceCoefficient(name="dice"),
        IoU(name="iou"),
    ]


def cropper_metrics() -> list[tf.keras.metrics.Metric]:
    """
    Metrics for bounding box regression.
    
    Returns:
        - MAE (mean absolute error)
        - BBox IoU (intersection over union)
    """
    return [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        BBoxIoU(name="bbox_iou"),
    ]


def get_metrics(task: str, **kwargs) -> list[tf.keras.metrics.Metric]:
    """
    Factory function to get metrics by task.
    
    Args:
        task: Task name (classification, segmentation, cropper)
        **kwargs: Additional parameters
        
    Returns:
        List of metrics
    """
    metrics_map = {
        "classification": classification_metrics,
        "segmentation": segmentation_metrics,
        "cropper": cropper_metrics,
    }
    
    if task not in metrics_map:
        raise ValueError(f"Unknown task: {task}. Available: {list(metrics_map.keys())}")
    
    return metrics_map[task]()
