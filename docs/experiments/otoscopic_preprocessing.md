# Otoscopic Image Preprocessing Strategy

## Analysis Summary

**Dataset**: 600 images (subset), 5 classes, 500x500 pixels average

### Image Characteristics
- **Dimensions**: 500x500 pixels (consistent across dataset)
- **Aspect Ratio**: ~1.0 (square images)
- **Color Space**: RGB, moderate brightness variation (std: 16.0)
- **File Format**: JPEG

### Recommended Input Size: **512x512**

**Rationale**:
- Original images are 500x500, so 512x512 requires minimal upscaling
- Maintains image quality and detail
- Standard size for modern CNNs
- Balances quality vs. training speed

**Alternatives Considered**:
- 224x224: Too much downscaling, loss of detail
- 256x256: Better than 224 but still significant downscaling
- 384x384: Good middle ground
- **512x512**: ✓ Best preserves original resolution

### Augmentation Strategy

Medical imaging requires careful augmentation to preserve diagnostic features:

| Augmentation | Enabled | Parameters | Rationale |
|--------------|---------|------------|-----------|
| **Rotation** | ✓ | ±15° | Slight orientation variations acceptable |
| **Horizontal Flip** | ✓ | 50% probability | Left/right ear symmetry |
| **Vertical Flip** | ✗ | N/A | Medically significant orientation |
| **Brightness** | ✓ | ±20% | Lighting conditions vary |
| **Contrast** | ✓ | ±15% | Camera/lighting differences |
| **Zoom** | ✓ | 0.9-1.1 | Different camera distances |
| **Color Jitter** | ✓ | Subtle | Slight color variations |

### Preprocessing Pipeline

```python
1. Load image from URI
2. Resize to 512x512 (bilinear interpolation)
3. Normalize to [0, 1] range
4. Apply augmentations (training only):
   - Random rotation (±15°)
   - Random horizontal flip
   - Random brightness (0.8-1.2)
   - Random contrast (0.85-1.15)
   - Random zoom (0.9-1.1)
5. Convert to tensor
```

### ROI Detection

**Assessment**: Not applicable for this dataset
- Images are already cropped to ear canal
- No significant background or irrelevant regions
- Full frame preprocessing is appropriate

### Configuration

```yaml
data:
  dataset:
    image_size: [512, 512]
    num_classes: 5
    
preprocess:
  pipeline_id: full_frame_v1
  normalisation: '0_1'
  
# Augmentation (if using data augmentation config)
augmentation:
  rotation_range: 15
  horizontal_flip: true
  vertical_flip: false
  brightness_range: [0.8, 1.2]
  contrast_range: [0.85, 1.15]
  zoom_range: [0.9, 1.1]
```

## Validation

Analysis performed on 100 sample images from training set:
- Dimension consistency: ✓ All images 500x500
- Color space: ✓ RGB, no grayscale
- Quality: ✓ No corrupted images detected
- Balance: ✓ Equal samples per class

## Next Steps

1. Create baseline training configurations with 512x512 input
2. Train models with and without augmentation
3. Compare results to validate preprocessing strategy
4. Document any adjustments needed based on training results
