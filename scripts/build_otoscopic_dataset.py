#!/usr/bin/env python3
"""
Otoscopic Dataset Builder

Builds a parquet dataset from otoscopic images using the shared dataset builder.
Features:
- Stratified 80/10/10 train/val/test split
- Balanced class distribution
- Manifest generation
- Subset mode for quick experiments
- Schema validation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.core.data.dataset_builder import (
    BuildSpec,
    build_parquet_dataset,
    stratified_split,
)

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


def collect_image_paths(source_dir: Path, subset_ratio: float = 1.0) -> pd.DataFrame:
    """Collect all image paths and return as DataFrame."""
    print(f"Collecting image paths (subset ratio: {subset_ratio})...")
    
    data = []
    
    for class_name, label in CLASS_MAPPING.items():
        class_dir = source_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping")
            continue
        
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        
        # Subset if requested
        if subset_ratio < 1.0:
            n_subset = int(len(images) * subset_ratio)
            if n_subset > 0:
                images = np.random.choice(images, size=n_subset, replace=False).tolist()
        
        for img_path in images:
            if img_path.stat().st_size == 0:
                print(f"Warning: Skipping empty file {img_path}")
                continue
                
            data.append({
                "image_uri": f"file://{img_path.absolute()}",
                "image_path": str(img_path),
                "class_name": class_name,
                "label": label
            })
    
    df = pd.DataFrame(data)
    print(f"Collected {len(df)} images across {len(CLASS_MAPPING)} classes")
    return df


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
        # For testing purposes, if source dir doesn't exist, we might want to fail gracefully or mock
        # But for this script, failing is appropriate
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Step 1: Collect data
    df = collect_image_paths(source_dir, subset_ratio=args.subset_ratio)
    
    if len(df) == 0:
        print("No images found. Exiting.")
        return

    # Step 2: Create splits
    splits = stratified_split(
        df,
        label_column="label",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=args.random_seed
    )
    
    # Step 3: Build dataset
    spec = BuildSpec(
        dataset_id=args.dataset_id,
        task_name="classification",
        dataset_version="1.0.0",
        ontology_version="1.0.0",
        task_mapping_version="1.0.0",
        preprocess_pipeline_id="full_frame_v1",
        preprocess_pipeline_version="1.0.0",
        created_by="build_otoscopic_dataset",
        status="active"
    )
    
    print(f"Building dataset in {output_dir}...")
    build_parquet_dataset(
        out_dir=output_dir,
        spec=spec,
        splits=splits,
        validate=True,
        compute_stats=True
    )
    
    # Step 4: Add custom metadata to manifest
    # The builder creates a standard manifest, we append our specific metadata
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        manifest["metadata"] = {
            "num_classes": len(CLASS_MAPPING),
            "class_mapping": CLASS_MAPPING,
            "class_descriptions": CLASS_DESCRIPTIONS,
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print("Updated manifest with custom metadata")

    print(f"\n✅ Dataset created successfully in {output_dir}")


if __name__ == "__main__":
    main()
