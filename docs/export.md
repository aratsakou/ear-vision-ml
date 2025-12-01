# Model Export

Models are automatically exported at the end of training if configured.

## Supported Formats

1.  **SavedModel**: Standard TensorFlow format.
2.  **TFLite**: TensorFlow Lite format for mobile/edge.
    - **Float32**: Standard TFLite.
    - **Quantized**: Int8 quantization for reduced size and faster inference.
3.  **CoreML**: Apple CoreML format (requires `coremltools`).

## Configuration

Export settings are in `configs/export/`.

```yaml
export:
  tflite:
    enabled: true
    quantize: true
  coreml:
    enabled: false # Set to true to enable
  saved_model:
    enabled: true
```

## Manual Export

You can manually export a trained model using the exporter script (if available) or by running a training job with `training.epochs=0` and loading weights (workaround).
Future versions will provide a dedicated export CLI.
