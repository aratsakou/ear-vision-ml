#!/usr/bin/env python3
"""
Otoscopic Image Preprocessing Analysis

Analyzes otoscopic images to determine optimal preprocessing strategy:
- Image size distribution and recommendations
- Aspect ratio analysis
- Color space characteristics
- ROI detection feasibility
- Augmentation recommendations
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def analyze_image_dimensions(manifest_path: Path, sample_size: int = 100) -> Dict:
    """Analyze image dimensions from dataset."""
    print(f"Analyzing image dimensions (sample size: {sample_size})...")
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Load a sample of images
    data_dir = manifest_path.parent / "data"
    train_parquet = manifest_path.parent / manifest["splits"]["train"][0]
    df = pd.read_parquet(train_parquet)
    
    # Sample images
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    dimensions = []
    aspect_ratios = []
    file_sizes = []
    
    for _, row in sample_df.iterrows():
        img_uri = row["image_uri"].replace("file://", "")
        try:
            with Image.open(img_uri) as img:
                w, h = img.size
                dimensions.append((w, h))
                aspect_ratios.append(w / h)
                file_sizes.append(Path(img_uri).stat().st_size / 1024)  # KB
        except Exception as e:
            print(f"Error loading {img_uri}: {e}")
    
    widths, heights = zip(*dimensions) if dimensions else ([], [])
    
    return {
        "num_samples": len(dimensions),
        "widths": {
            "min": min(widths) if widths else 0,
            "max": max(widths) if widths else 0,
            "mean": np.mean(widths) if widths else 0,
            "std": np.std(widths) if widths else 0,
        },
        "heights": {
            "min": min(heights) if heights else 0,
            "max": max(heights) if heights else 0,
            "mean": np.mean(heights) if heights else 0,
            "std": np.std(heights) if heights else 0,
        },
        "aspect_ratios": {
            "min": min(aspect_ratios) if aspect_ratios else 0,
            "max": max(aspect_ratios) if aspect_ratios else 0,
            "mean": np.mean(aspect_ratios) if aspect_ratios else 0,
            "std": np.std(aspect_ratios) if aspect_ratios else 0,
        },
        "file_sizes_kb": {
            "min": min(file_sizes) if file_sizes else 0,
            "max": max(file_sizes) if file_sizes else 0,
            "mean": np.mean(file_sizes) if file_sizes else 0,
        },
        "raw_dimensions": dimensions[:20],  # Store first 20 for reference
    }


def analyze_color_characteristics(manifest_path: Path, sample_size: int = 50) -> Dict:
    """Analyze color space characteristics."""
    print(f"Analyzing color characteristics (sample size: {sample_size})...")
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    train_parquet = manifest_path.parent / manifest["splits"]["train"][0]
    df = pd.read_parquet(train_parquet)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    brightness_values = []
    contrast_values = []
    saturation_values = []
    
    for _, row in sample_df.iterrows():
        img_uri = row["image_uri"].replace("file://", "")
        try:
            img = cv2.imread(img_uri)
            if img is None:
                continue
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Brightness (V channel in HSV)
            brightness_values.append(np.mean(hsv[:, :, 2]))
            
            # Contrast (std of grayscale)
            contrast_values.append(np.std(gray))
            
            # Saturation (S channel in HSV)
            saturation_values.append(np.mean(hsv[:, :, 1]))
            
        except Exception as e:
            print(f"Error analyzing {img_uri}: {e}")
    
    return {
        "brightness": {
            "mean": np.mean(brightness_values) if brightness_values else 0,
            "std": np.std(brightness_values) if brightness_values else 0,
            "min": min(brightness_values) if brightness_values else 0,
            "max": max(brightness_values) if brightness_values else 0,
        },
        "contrast": {
            "mean": np.mean(contrast_values) if contrast_values else 0,
            "std": np.std(contrast_values) if contrast_values else 0,
        },
        "saturation": {
            "mean": np.mean(saturation_values) if saturation_values else 0,
            "std": np.std(saturation_values) if saturation_values else 0,
        },
    }


def recommend_input_size(dimension_stats: Dict) -> Dict:
    """Recommend optimal input size based on image dimensions."""
    mean_width = dimension_stats["widths"]["mean"]
    mean_height = dimension_stats["heights"]["mean"]
    
    # Common input sizes
    sizes = [224, 256, 384, 512]
    
    # Find closest size that doesn't require too much upscaling
    avg_dim = (mean_width + mean_height) / 2
    
    recommended = None
    for size in sizes:
        if size >= avg_dim * 0.8:  # Allow some downscaling
            recommended = size
            break
    
    if recommended is None:
        recommended = sizes[-1]  # Use largest if images are very big
    
    return {
        "recommended_size": recommended,
        "rationale": f"Images average {mean_width:.0f}x{mean_height:.0f}. "
                    f"Recommended {recommended}x{recommended} to minimize information loss.",
        "alternatives": {
            224: "Fastest training, good for MobileNet",
            256: "Balanced speed/quality",
            384: "Higher quality, slower training",
            512: "Maximum quality, slowest training",
        },
    }


def recommend_augmentation(color_stats: Dict, dimension_stats: Dict) -> Dict:
    """Recommend augmentation strategy."""
    recommendations = {
        "rotation": {
            "enabled": True,
            "range": "±15°",
            "rationale": "Medical images can have slight orientation variations",
        },
        "horizontal_flip": {
            "enabled": True,
            "rationale": "Ear images can be from left or right ear",
        },
        "vertical_flip": {
            "enabled": False,
            "rationale": "Vertical orientation is medically significant",
        },
        "brightness": {
            "enabled": True,
            "range": "±20%",
            "rationale": f"Brightness varies (std: {color_stats['brightness']['std']:.1f})",
        },
        "contrast": {
            "enabled": True,
            "range": "±15%",
            "rationale": "Lighting conditions vary across images",
        },
        "zoom": {
            "enabled": True,
            "range": "0.9-1.1",
            "rationale": "Simulate different camera distances",
        },
    }
    
    return recommendations


def create_visualizations(
    dimension_stats: Dict,
    color_stats: Dict,
    output_dir: Path,
) -> None:
    """Create visualization plots."""
    print("Creating visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Otoscopic Dataset Analysis", fontsize=16)
    
    # Dimension distribution
    ax = axes[0, 0]
    dims = dimension_stats["raw_dimensions"]
    if dims:
        widths, heights = zip(*dims)
        ax.scatter(widths, heights, alpha=0.6)
        ax.set_xlabel("Width (pixels)")
        ax.set_ylabel("Height (pixels)")
        ax.set_title("Image Dimensions")
        ax.grid(True, alpha=0.3)
    
    # Aspect ratio
    ax = axes[0, 1]
    ar_stats = dimension_stats["aspect_ratios"]
    ax.bar(["Min", "Mean", "Max"], [ar_stats["min"], ar_stats["mean"], ar_stats["max"]])
    ax.set_ylabel("Aspect Ratio")
    ax.set_title("Aspect Ratio Distribution")
    ax.axhline(y=1.0, color='r', linestyle='--', label='Square (1:1)')
    ax.legend()
    
    # Color characteristics
    ax = axes[1, 0]
    metrics = ["Brightness", "Contrast", "Saturation"]
    values = [
        color_stats["brightness"]["mean"],
        color_stats["contrast"]["mean"],
        color_stats["saturation"]["mean"],
    ]
    ax.bar(metrics, values)
    ax.set_ylabel("Mean Value")
    ax.set_title("Color Characteristics")
    
    # Size recommendation
    ax = axes[1, 1]
    sizes = [224, 256, 384, 512]
    mean_dim = (dimension_stats["widths"]["mean"] + dimension_stats["heights"]["mean"]) / 2
    scaling_factors = [mean_dim / s for s in sizes]
    ax.bar([str(s) for s in sizes], scaling_factors)
    ax.set_xlabel("Input Size")
    ax.set_ylabel("Scaling Factor")
    ax.set_title("Input Size vs. Original Resolution")
    ax.axhline(y=1.0, color='r', linestyle='--', label='No scaling')
    ax.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "preprocessing_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze otoscopic images for preprocessing")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="data/otoscopic/subset/manifest.json",
        help="Path to dataset manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/experiments",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of images to sample for analysis",
    )
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    
    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing dataset: {manifest_path}")
    
    # Run analyses
    dimension_stats = analyze_image_dimensions(manifest_path, args.sample_size)
    color_stats = analyze_color_characteristics(manifest_path, min(args.sample_size, 50))
    
    # Generate recommendations
    input_size_rec = recommend_input_size(dimension_stats)
    augmentation_rec = recommend_augmentation(color_stats, dimension_stats)
    
    # Compile results
    results = {
        "dataset": str(manifest_path),
        "analysis_date": pd.Timestamp.now().isoformat(),
        "dimension_statistics": dimension_stats,
        "color_statistics": color_stats,
        "recommendations": {
            "input_size": input_size_rec,
            "augmentation": augmentation_rec,
        },
    }
    
    # Save results
    results_path = output_dir / "otoscopic_preprocessing_analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Analysis saved to {results_path}")
    
    # Create visualizations
    create_visualizations(dimension_stats, color_stats, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING RECOMMENDATIONS")
    print("="*60)
    print(f"\nInput Size: {input_size_rec['recommended_size']}x{input_size_rec['recommended_size']}")
    print(f"Rationale: {input_size_rec['rationale']}")
    print("\nAugmentation Strategy:")
    for aug_name, aug_config in augmentation_rec.items():
        if aug_config["enabled"]:
            range_info = f" ({aug_config.get('range', '')})" if 'range' in aug_config else ""
            print(f"  ✓ {aug_name.replace('_', ' ').title()}{range_info}")
            print(f"    → {aug_config['rationale']}")
    print("="*60)


if __name__ == "__main__":
    main()
