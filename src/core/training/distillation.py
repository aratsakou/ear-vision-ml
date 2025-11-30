import tensorflow as tf
from typing import Any, Tuple

class DistillationLoss(tf.keras.losses.Loss):
    """
    Computes the distillation loss:
    L = alpha * student_loss + (1 - alpha) * distillation_loss
    
    where distillation_loss is KL divergence between soft targets of teacher and student.
    """
    def __init__(self, 
                 student_loss_fn: tf.keras.losses.Loss,
                 alpha: float = 0.1, 
                 temperature: float = 3.0,
                 name: str = "distillation_loss"):
        super().__init__(name=name)
        self.student_loss_fn = student_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.kl_divergence = tf.keras.losses.KLDivergence()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # This standard call signature doesn't easily support the teacher logits 
        # unless y_true is packed or we use a custom training loop.
        # For a custom training loop, we might call this differently.
        # However, to fit into standard Keras compile/fit, we often need a custom model class.
        # Here we assume y_true contains just the ground truth labels.
        # The teacher predictions must be passed separately or computed within a custom model.
        
        # NOTE: This class is intended to be used within a Distiller model wrapper
        # where we have access to teacher logits.
        return self.student_loss_fn(y_true, y_pred)

    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, teacher_logits: tf.Tensor) -> tf.Tensor:
        """
        Compute the full distillation loss.
        """
        student_loss = self.student_loss_fn(y_true, y_pred)
        
        # Soften probabilities
        teacher_probs = tf.nn.softmax(teacher_logits / self.temperature, axis=1)
        student_probs = tf.nn.softmax(y_pred / self.temperature, axis=1)
        
        distillation_loss = self.kl_divergence(teacher_probs, student_probs)
        
        # Scale distillation loss by temperature^2 as per Hinton et al.
        distillation_loss = distillation_loss * (self.temperature ** 2)
        
        return self.alpha * student_loss + (1 - self.alpha) * distillation_loss


class Distiller(tf.keras.Model):
    """
    Custom Keras Model for Knowledge Distillation.
    Wraps a student model and a teacher model.
    """
    def __init__(self, student: tf.keras.Model, teacher: tf.keras.Model, cfg: Any):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.cfg = cfg
        
        # Distillation params
        self.alpha = cfg.training.distillation.alpha
        self.temperature = cfg.training.distillation.temperature
        
        self.distillation_loss_fn = DistillationLoss(
            student_loss_fn=tf.keras.losses.get(cfg.training.loss.name), # Simplified retrieval
            alpha=self.alpha,
            temperature=self.temperature
        )

    def compile(self, optimizer, metrics, student_loss_fn):
        """
        Compile with specific student loss function.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = DistillationLoss(
            student_loss_fn=student_loss_fn,
            alpha=self.alpha,
            temperature=self.temperature
        )
        self.student_loss_metric = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_metric = tf.keras.metrics.Mean(name="distillation_loss")

    def train_step(self, data):
        # Unpack data
        x, y = data
        
        # Forward pass of teacher
        teacher_logits = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_logits = self.student(x, training=True)
            
            # Compute loss
            loss = self.distillation_loss_fn.compute_loss(y, student_logits, teacher_logits)
            
        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.student_loss_metric.update_state(self.distillation_loss_fn.student_loss_fn(y, student_logits))
        # Re-compute dist loss for logging (could be optimized)
        dist_loss_val = loss - (self.alpha * self.student_loss_metric.result())
        self.distillation_loss_metric.update_state(dist_loss_val)
        
        # Update compiled metrics (e.g. accuracy)
        self.compiled_metrics.update_state(y, student_logits)
        
        results = {}
        for m in self.metrics:
            res = m.result()
            if isinstance(res, dict):
                results.update(res)
            else:
                results[m.name] = res

        results.update({
            "student_loss": self.student_loss_metric.result(),
            "distillation_loss": self.distillation_loss_metric.result(),
            "loss": loss
        })
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        
        # For validation, we typically just evaluate the student performance
        # But we can also track distillation loss if desired.
        student_loss = self.distillation_loss_fn.student_loss_fn(y, y_pred)
        
        self.compiled_metrics.update_state(y, y_pred)
        
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": student_loss})
        return results
    
    def call(self, inputs, training=None, mask=None):
        # By default, act as the student model
        return self.student(inputs, training=training, mask=mask)
