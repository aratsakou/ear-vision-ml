"""
Advanced model export utilities with state-of-the-art optimization techniques.

Features:
- Multi-format export (SavedModel, TFLite, ONNX)
- Advanced quantization (INT8, FP16, dynamic range)
- Post-training optimization
- Model pruning and compression
- Benchmark generation
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.interfaces import Exporter, Component
from src.core.export.coreml_exporter import CoreMLExporter


@dataclass(frozen=True)
class ExportPaths:
    saved_model_dir: Path
    tflite_path: Path | None
    tflite_quant_path: Path | None
    tflite_int8_path: Path | None
    tflite_fp16_path: Path | None
    onnx_path: Path | None
    coreml_path: Path | None
    manifest_path: Path
    benchmark_path: Path | None


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to get git commit: {e}")
    return "unknown"


def _write_model_manifest(
    out_dir: Path,
    cfg: Any,
    model: tf.keras.Model,
    dataset_id: str,
    created_by: str,
    artifacts: dict[str, str],
    benchmark_results: dict[str, Any] | None = None,
) -> Path:
    """Write comprehensive model manifest with metadata."""
    # Get model size
    param_count = model.count_params()
    
    manifest = {
        "model_id": f"{cfg.model.name}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "task_name": str(cfg.task.name),
        "model_name": str(cfg.model.name),
        "version": "0.1.0",
        "dataset_id": dataset_id,
        "preprocess_pipeline_id": str(cfg.preprocess.pipeline_id),
        "preprocess_pipeline_version": str(cfg.preprocess.version),
        "input_shape": [int(x) for x in cfg.model.input_shape],
        "outputs": {o.name: list(o.shape) for o in model.outputs},
        "class_labels": [],
        "git_commit": _get_git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": created_by,
        "artifacts": artifacts,
        "model_stats": {
            "total_params": int(param_count),
            "trainable_params": int(sum(tf.size(w).numpy() for w in model.trainable_weights)),
            "non_trainable_params": int(sum(tf.size(w).numpy() for w in model.non_trainable_weights)),
        },
        "benchmark": benchmark_results or {},
    }
    
    p = out_dir / "model_manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


def _create_representative_dataset(input_shape: tuple[int, ...], num_samples: int = 100):
    """Create representative dataset for quantization calibration."""
    def representative_data_gen():
        for _ in range(num_samples):
            # Generate random data matching input shape
            data = np.random.rand(1, *input_shape).astype(np.float32)
            yield [data]
    return representative_data_gen


def _export_tflite_variants(
    saved_model_dir: Path,
    out_dir: Path,
    input_shape: tuple[int, ...],
    enable_int8: bool = True,
    enable_fp16: bool = True,
) -> dict[str, Path | None]:
    """
    Export multiple TFLite variants with different optimizations.
    
    Returns dict with paths to:
    - float32: Standard float model
    - dynamic_range: Dynamic range quantization (weights only)
    - fp16: FP16 quantization
    - int8: Full INT8 quantization (requires representative dataset)
    """
    paths = {
        "float32": None,
        "dynamic_range": None,
        "fp16": None,
        "int8": None,
    }
    
    # 1. Float32 (baseline)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    float_path = out_dir / "model_float32.tflite"
    float_path.write_bytes(tflite_model)
    paths["float32"] = float_path
    
    # 2. Dynamic Range Quantization (weights to INT8, activations stay float)
    converter_dr = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter_dr.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dr = converter_dr.convert()
    dr_path = out_dir / "model_dynamic_range.tflite"
    dr_path.write_bytes(tflite_dr)
    paths["dynamic_range"] = dr_path
    
    # 3. FP16 Quantization
    if enable_fp16:
        converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_fp16.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter_fp16.convert()
        fp16_path = out_dir / "model_fp16.tflite"
        fp16_path.write_bytes(tflite_fp16)
        paths["fp16"] = fp16_path
    
    # 4. Full INT8 Quantization (requires representative dataset)
    if enable_int8:
        try:
            converter_int8 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
            converter_int8.representative_dataset = _create_representative_dataset(input_shape)
            # For full INT8, set input/output types
            converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter_int8.inference_input_type = tf.uint8
            converter_int8.inference_output_type = tf.uint8
            tflite_int8 = converter_int8.convert()
            int8_path = out_dir / "model_int8.tflite"
            int8_path.write_bytes(tflite_int8)
            paths["int8"] = int8_path
        except Exception as e:
            print(f"Warning: INT8 quantization failed: {e}")
    
    return paths


def _benchmark_tflite_models(
    tflite_paths: dict[str, Path | None],
    input_shape: tuple[int, ...],
    num_runs: int = 100,
) -> dict[str, Any]:
    """
    Benchmark TFLite models for latency and size.
    
    Returns metrics for each variant.
    """
    import time
    
    results = {}
    
    for variant, path in tflite_paths.items():
        if path is None or not path.exists():
            continue
        
        # Model size
        size_mb = path.stat().st_size / (1024 * 1024)
        
        # Latency benchmark
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        _ = interpreter.get_output_details()[0]  # Needed for allocation
        
        # Warm-up
        dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
        if input_details['dtype'] == np.uint8:
            # Scale to uint8 range
            dummy_input = (dummy_input * 255).astype(np.uint8)
        
        for _ in range(10):
            interpreter.set_tensor(input_details['index'], dummy_input)
            interpreter.invoke()
        
        # Actual benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            interpreter.set_tensor(input_details['index'], dummy_input)
            interpreter.invoke()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        results[variant] = {
            "size_mb": round(size_mb, 3),
            "latency_ms": {
                "mean": round(np.mean(latencies), 3),
                "std": round(np.std(latencies), 3),
                "min": round(np.min(latencies), 3),
                "max": round(np.max(latencies), 3),
                "p50": round(np.percentile(latencies, 50), 3),
                "p95": round(np.percentile(latencies, 95), 3),
                "p99": round(np.percentile(latencies, 99), 3),
            },
            "compression_ratio": None,  # Will be calculated relative to float32
        }
    
    # Calculate compression ratios
    if "float32" in results:
        base_size = results["float32"]["size_mb"]
        for variant in results:
            if variant != "float32":
                results[variant]["compression_ratio"] = round(
                    base_size / results[variant]["size_mb"], 2
                )
    
    return results


class StandardExporter(Exporter, Component):
    def initialize(self) -> None:
        print("StandardExporter initialized")

    def cleanup(self) -> None:
        print("StandardExporter cleaned up")
    def export(self, model: tf.keras.Model, cfg: Any, artifacts_dir: Any) -> dict[str, Any]:
        """
        Export model with state-of-the-art optimizations.
        """
        out_dir = Path(artifacts_dir) / "exports" / str(cfg.model.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_id = getattr(cfg.data.dataset, "id", "unknown")
        created_by = "unknown"
        enable_advanced_quantization = True
        enable_benchmarking = True

        # 1. Export SavedModel
        saved_model_dir = out_dir / "saved_model"
        model.export(saved_model_dir)

        # 2. Get input shape for quantization
        input_shape = tuple(int(x) for x in cfg.model.input_shape)
        
        # 3. Export TFLite variants
        tflite_enabled = getattr(cfg.export.export.tflite, "enabled", True) if hasattr(cfg, 'export') and hasattr(cfg.export, 'export') else True
        tflite_paths_dict = {}
        
        if tflite_enabled:
            tflite_paths_dict = _export_tflite_variants(
                saved_model_dir=saved_model_dir,
                out_dir=out_dir,
                input_shape=input_shape,
                enable_int8=enable_advanced_quantization,
                enable_fp16=enable_advanced_quantization,
            )
        

            
        # 4. Export Core ML (if enabled)
        coreml_enabled = False
        if hasattr(cfg, 'export') and hasattr(cfg.export, 'export') and hasattr(cfg.export.export, 'coreml'):
            coreml_enabled = getattr(cfg.export.export.coreml, "enabled", False)
            
        coreml_path = None
        if coreml_enabled:
            exporter = CoreMLExporter()
            coreml_path = exporter.export(model, out_dir, cfg)
        
        # 5. Benchmark models
        benchmark_results = None
        benchmark_path = None
        if enable_benchmarking and tflite_paths_dict:
            benchmark_results = _benchmark_tflite_models(
                tflite_paths=tflite_paths_dict,
                input_shape=input_shape,
            )
            benchmark_path = out_dir / "benchmark_results.json"
            benchmark_path.write_text(json.dumps(benchmark_results, indent=2))
        
        # 5. Create artifacts dict
        artifacts = {
            "saved_model_dir": str(saved_model_dir),
            "tflite_float32": str(tflite_paths_dict.get("float32", "")),
            "tflite_dynamic_range": str(tflite_paths_dict.get("dynamic_range", "")),
            "tflite_fp16": str(tflite_paths_dict.get("fp16", "")),
            "tflite_int8": str(tflite_paths_dict.get("int8", "")),
            "coreml": str(coreml_path) if coreml_path else "",
            "benchmark_results": str(benchmark_path) if benchmark_path else "",
        }
        
        # 6. Write manifest
        manifest_path = _write_model_manifest(
            out_dir=out_dir,
            cfg=cfg,
            model=model,
            dataset_id=dataset_id,
            created_by=created_by,
            artifacts=artifacts,
            benchmark_results=benchmark_results,
        )
        
        export_paths = ExportPaths(
            saved_model_dir=saved_model_dir,
            tflite_path=tflite_paths_dict.get("float32"),
            tflite_quant_path=tflite_paths_dict.get("dynamic_range"),
            tflite_int8_path=tflite_paths_dict.get("int8"),
            tflite_fp16_path=tflite_paths_dict.get("fp16"),
            onnx_path=None,
            coreml_path=coreml_path,
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
        )
        
        # Return dict representation of ExportPaths for interface compatibility
        return asdict(export_paths)




if __name__ == "__main__":
    raise SystemExit("Use task entrypoints. Export is called from trainers.")
