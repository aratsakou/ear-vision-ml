"""
Advanced model equivalence testing and validation.

Features:
- Numerical equivalence testing (Keras vs TFLite)
- Output distribution analysis
- Quantization error analysis
- Performance regression detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class EquivalenceResult:
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    ok: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class QuantizationAnalysis:
    """Analysis of quantization effects on model outputs."""
    snr_db: float  # Signal-to-noise ratio
    psnr_db: float  # Peak signal-to-noise ratio
    cosine_similarity: float
    correlation: float
    max_error: float
    mean_error: float


def compute_quantization_metrics(
    original: np.ndarray,
    quantized: np.ndarray,
) -> QuantizationAnalysis:
    """
    Compute comprehensive quantization quality metrics.
    
    Args:
        original: Original model outputs
        quantized: Quantized model outputs
        
    Returns:
        QuantizationAnalysis with quality metrics
    """
    # Flatten arrays
    orig_flat = original.flatten()
    quant_flat = quantized.flatten()
    
    # Error metrics
    error = orig_flat - quant_flat
    max_error = float(np.max(np.abs(error)))
    mean_error = float(np.mean(np.abs(error)))
    
    # Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(orig_flat ** 2)
    noise_power = np.mean(error ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_val = np.max(np.abs(orig_flat))
    psnr_db = 20 * np.log10(max_val / (np.sqrt(noise_power) + 1e-10))
    
    # Cosine similarity
    cosine_sim = np.dot(orig_flat, quant_flat) / (
        np.linalg.norm(orig_flat) * np.linalg.norm(quant_flat) + 1e-10
    )
    
    # Correlation
    correlation = np.corrcoef(orig_flat, quant_flat)[0, 1]
    
    return QuantizationAnalysis(
        snr_db=float(snr_db),
        psnr_db=float(psnr_db),
        cosine_similarity=float(cosine_sim),
        correlation=float(correlation),
        max_error=max_error,
        mean_error=mean_error,
    )


def compare_keras_vs_tflite(
    keras_model: tf.keras.Model,
    tflite_path: str,
    input_shape: tuple[int, int, int],
    atol: float = 1e-2,
    num_samples: int = 100,
) -> EquivalenceResult:
    """
    Advanced equivalence testing between Keras and TFLite models.
    
    Tests multiple random inputs and computes comprehensive statistics.
    
    Args:
        keras_model: Original Keras model
        tflite_path: Path to TFLite model
        input_shape: Input shape (H, W, C)
        atol: Absolute tolerance for equivalence
        num_samples: Number of random inputs to test
        
    Returns:
        EquivalenceResult with detailed metrics
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    
    # Check if model uses quantized inputs
    is_quantized_input = in_details['dtype'] == np.uint8
    
    max_diffs = []
    mean_diffs = []
    rel_diffs = []
    
    all_keras_outputs = []
    all_tflite_outputs = []
    
    for i in range(num_samples):
        # Create test input
        x = np.random.RandomState(i).rand(1, *input_shape).astype(np.float32)
        
        # Keras prediction
        y_keras = keras_model.predict(x, verbose=0)
        y_keras = np.asarray(y_keras)
        
        # TFLite prediction
        if is_quantized_input:
            # Quantize input to uint8
            x_quant = (x * 255).astype(np.uint8)
            interpreter.set_tensor(in_details["index"], x_quant)
        else:
            interpreter.set_tensor(in_details["index"], x)
        
        interpreter.invoke()
        y_tflite = interpreter.get_tensor(out_details["index"])
        
        # Dequantize output if needed
        if out_details['dtype'] == np.uint8:
            scale, zero_point = out_details['quantization']
            y_tflite = (y_tflite.astype(np.float32) - zero_point) * scale
        
        # Compute differences
        abs_diff = np.abs(y_keras - y_tflite)
        max_diffs.append(np.max(abs_diff))
        mean_diffs.append(np.mean(abs_diff))
        
        # Relative difference (avoid division by zero)
        rel_diff = abs_diff / (np.abs(y_keras) + 1e-10)
        rel_diffs.append(np.max(rel_diff))
        
        all_keras_outputs.append(y_keras)
        all_tflite_outputs.append(y_tflite)
    
    # Aggregate statistics
    max_abs_diff = float(np.max(max_diffs))
    mean_abs_diff = float(np.mean(mean_diffs))
    max_rel_diff = float(np.max(rel_diffs))
    
    # Compute quantization analysis
    keras_concat = np.concatenate(all_keras_outputs, axis=0)
    tflite_concat = np.concatenate(all_tflite_outputs, axis=0)
    quant_analysis = compute_quantization_metrics(keras_concat, tflite_concat)
    
    # Determine if models are equivalent
    ok = max_abs_diff <= atol
    
    details = {
        "num_samples_tested": num_samples,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "std_abs_diff": float(np.std(max_diffs)),
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": float(np.mean(rel_diffs)),
        "quantization_analysis": {
            "snr_db": quant_analysis.snr_db,
            "psnr_db": quant_analysis.psnr_db,
            "cosine_similarity": quant_analysis.cosine_similarity,
            "correlation": quant_analysis.correlation,
        },
        "is_quantized_input": is_quantized_input,
        "is_quantized_output": out_details['dtype'] == np.uint8,
    }
    
    return EquivalenceResult(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_rel_diff=max_rel_diff,
        ok=ok,
        details=details,
    )


def validate_model_outputs(
    model: tf.keras.Model,
    test_inputs: np.ndarray,
    expected_output_range: tuple[float, float] = (0.0, 1.0),
    check_nan: bool = True,
    check_inf: bool = True,
) -> dict[str, Any]:
    """
    Validate model outputs for common issues.
    
    Args:
        model: Model to validate
        test_inputs: Test input data
        expected_output_range: Expected range of outputs
        check_nan: Check for NaN values
        check_inf: Check for Inf values
        
    Returns:
        Validation results dictionary
    """
    outputs = model.predict(test_inputs, verbose=0)
    
    issues = []
    
    # Check for NaN
    if check_nan and np.any(np.isnan(outputs)):
        issues.append("NaN values detected in outputs")
    
    # Check for Inf
    if check_inf and np.any(np.isinf(outputs)):
        issues.append("Inf values detected in outputs")
    
    # Check output range
    min_val, max_val = np.min(outputs), np.max(outputs)
    if min_val < expected_output_range[0] or max_val > expected_output_range[1]:
        issues.append(
            f"Output range [{min_val:.4f}, {max_val:.4f}] outside expected "
            f"[{expected_output_range[0]}, {expected_output_range[1]}]"
        )
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "output_stats": {
            "min": float(min_val),
            "max": float(max_val),
            "mean": float(np.mean(outputs)),
            "std": float(np.std(outputs)),
        },
    }
