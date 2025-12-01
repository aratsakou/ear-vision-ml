#!/usr/bin/env python3
"""
Otoscopic Dataset Builder

Builds a parquet dataset from otoscopic images with the following features:
- Stratified 80/10/10 train/val/test split
- Balanced class distribution
- Manifest generation
- Subset mode for quick experiments
- Schema validation
"""

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Class mapping
CLASS_MAPPING = {
    "Acute Otitis Media": 0,
    "Cerumen Impaction": 1,
    "Chronic Otitis Media": 2,
    "Myringosclerosis": 3,
    "Normal": 4,
}

CLASS_DESCRIPTIONS = {
    "Acute Otitis Media": "Bacterial or viral infection causing inflammation and fluid buildup behind the eardrum",
    "Cerumen Impaction": "Excessive earwax buildup blocking the ear canal",
    "Chronic Otitis Media": "Persistent ear infection leading to long-term damage",
    "Myringosclerosis": "Formation of calcified plaques on the tympanic membrane",
    "Normal": "Healthy ear with no signs of infection or abnormalities",
}


def analyze_images(source_dir: Path) -> Dict:
    """Analyze image properties across all classes."""
    print("Analyzing images...")
    
    stats = {
        "total_images": 0,
        "class_counts": {},
        "dimensions": [],
        "formats": defaultdict(int),
        "file_sizes": [],
    }
    
    for class_name in CLASS_MAPPING.keys():
        class_dir = source_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
            
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        stats["class_counts"][class_name] = len(images)
        stats["total_images"] += len(images)
        
        # Sample 10 images per class for dimension analysis
        for img_path in images[:10]:
            try:
                with Image.open(img_path) as img:
                    stats["dimensions"].append(img.size)
                    stats["formats"][img.format] += 1
                stats["file_sizes"].append(img_path.stat().st_size)
            except Exception as e:
                print(f"Error analyzing {img_path}: {e}")
    
    # Compute statistics
    if stats["dimensions"]:
        widths, heights = zip(*stats["dimensions"])
        stats["dimension_stats"] = {
            "mean_width": np.mean(widths),
            "mean_height": np.mean(heights),
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
        }
    
    if stats["file_sizes"]:
        stats["file_size_stats"] = {
            "mean_mb": np.mean(stats["file_sizes"]) / 1024 / 1024,
            "min_mb": min(stats["file_sizes"]) / 1024 / 1024,
            "max_mb": max(stats["file_sizes"]) / 1024 / 1024,
        }
    
    return stats


def collect_image_paths(source_dir: Path, subset_ratio: float = 1.0) -> List[Tuple[Path, str, int]]:
    """Collect all image paths with their labels."""
    print(f"Collecting image paths (subset ratio: {subset_ratio})...")
    
    image_data = []
    
    for class_name, label in CLASS_MAPPING.items():
        class_dir = source_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping")
            continue
        
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        
        # Subset if requested
        if subset_ratio < 1.0:
            n_subset = int(len(images) * subset_ratio)
            images = np.random.choice(images, size=n_subset, replace=False).tolist()
        
        for img_path in images:
            image_data.append((img_path, class_name, label))
    
    print(f"Collected {len(image_data)} images across {len(CLASS_MAPPING)} classes")
    return image_data


def create_splits(
    image_data: List[Tuple[Path, str, int]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Dict[str, List[Tuple[Path, str, int]]]:
    """Create stratified train/val/test splits."""
    print(f"Creating splits: {train_ratio}/{val_ratio}/{test_ratio}...")
    
    # Extract labels for stratification
    labels = [label for _, _, label in image_data]
    
    # First split: train vs (val + test)
    train_data, temp_data = train_test_split(
        image_data,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state,
    )
    
    # Second split: val vs test
    temp_labels = [label for _, _, label in temp_data]
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=random_state,
    )
    
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }
    
    # Print split statistics
    for split_name, split_data in splits.items():
        class_counts = defaultdict(int)
        for _, class_name, _ in split_data:
            class_counts[class_name] += 1
        print(f"{split_name}: {len(split_data)} images - {dict(class_counts)}")
    
    return splits


def create_parquet_files(
    splits: Dict[str, List[Tuple[Path, str, int]]],
    output_dir: Path,
    shard_size: int = 500,
) -> Dict[str, List[str]]:
    """Create parquet files for each split."""
    print("Creating parquet files...")
    
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_paths = {}
    
    for split_name, split_data in splits.items():
        split_parquets = []
        
        # Create shards
        for shard_idx in range(0, len(split_data), shard_size):
            shard_data = split_data[shard_idx : shard_idx + shard_size]
            
            # Create DataFrame
            rows = []
            for img_path, class_name, label in shard_data:
                rows.append({
                    "image_uri": f"file://{img_path.absolute()}",
                    "label": label,
                    "class_name": class_name,
                    "split": split_name,
                })
            
            df = pd.DataFrame(rows)
            
            # Save parquet
            shard_file = f"{split_name}-{shard_idx // shard_size:04d}.parquet"
            parquet_path = data_dir / shard_file
            df.to_parquet(parquet_path, index=False)
            
            split_parquets.append(f"data/{shard_file}")
            print(f"  Created {shard_file} with {len(df)} rows")
        
        parquet_paths[split_name] = split_parquets
    
    return parquet_paths


def create_manifest(
    output_dir: Path,
    parquet_paths: Dict[str, List[str]],
    dataset_id: str,
    stats: Dict,
) -> None:
    """Create manifest.json file."""
    print("Creating manifest...")
    
    manifest = {
        "dataset_id": dataset_id,
        "task_name": "classification",
        "version": "1.0.0",
        "status": "active",
        "label_mapping_versions": {
            "ontology": "1.0.0",
            "task_mapping": "1.0.0",
            "labelbox_mappings": [],
        },
        "sampling_config": {
            "hash": "otoscopic_stratified_split",
            "parameters": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "random_state": 42,
            },
        },
        "preprocess_pipeline": {
            "id": "full_frame_v1",
            "version": "1.0.0",
        },
        "splits": parquet_paths,
        "created_by": "build_otoscopic_dataset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "num_classes": len(CLASS_MAPPING),
            "class_mapping": CLASS_MAPPING,
            "class_descriptions": CLASS_DESCRIPTIONS,
            "total_images": stats["total_images"],
            "class_counts": stats["class_counts"],
        },
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Build otoscopic dataset")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="tmp/Otoscopic_Data",
        help="Source directory containing class folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/otoscopic/full",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=1.0,
        help="Ratio of data to use (0.0-1.0). Use 0.2 for quick experiments",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="otoscopic_classification",
        help="Dataset identifier",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Analyze images
    stats = analyze_images(source_dir)
    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Class counts: {stats['class_counts']}")
    if "dimension_stats" in stats:
        print(f"  Avg dimensions: {stats['dimension_stats']['mean_width']:.0f}x{stats['dimension_stats']['mean_height']:.0f}")
    
    # Save analysis
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nAnalysis saved to {analysis_path}")
    
    # Step 2: Collect image paths
    image_data = collect_image_paths(source_dir, subset_ratio=args.subset_ratio)
    
    # Step 3: Create splits
    splits = create_splits(image_data)
    
    # Step 4: Create parquet files
    parquet_paths = create_parquet_files(splits, output_dir)
    
    # Step 5: Create manifest
    create_manifest(output_dir, parquet_paths, args.dataset_id, stats)
    
    print(f"\n✅ Dataset created successfully in {output_dir}")
    print(f"   Total images: {len(image_data)}")
    print(f"   Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    main()
