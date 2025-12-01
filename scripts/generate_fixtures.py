from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd


def create_dummy_data(base_dir: Path, num_rows=10, split="train"):
    """Create dummy dataset with images and masks."""
    data_dir = base_dir / "data"
    images_dir = base_dir / "images"
    masks_dir = base_dir / "masks"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    image_uris = []
    mask_uris = []
    
    for i in range(num_rows):
        # Create a dummy image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_path = images_dir / f"img_{split}_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        image_uris.append(f"file://{img_path.absolute()}")
        
        # Create a dummy mask (grayscale, values 0..num_classes)
        mask = np.random.randint(0, 3, (224, 224), dtype=np.uint8)
        mask_path = masks_dir / f"mask_{split}_{i}.png"
        cv2.imwrite(str(mask_path), mask)
        mask_uris.append(f"file://{mask_path.absolute()}")

    df = pd.DataFrame({
        'image_uri': image_uris,
        'mask_uri': mask_uris,
        'timestamp_ms': [i * 1000 for i in range(num_rows)],
        'label': np.random.randint(0, 3, size=num_rows),
        'split': [split] * num_rows
    })
    return df


def create_manifest(base_dir: Path, dataset_id: str):
    """Create a minimal manifest.json."""
    manifest = {
        "dataset_id": dataset_id,
        "task_name": "classification",
        "version": "1.0.0",
        "status": "active",
        "label_mapping_versions": {
            "ontology": "1.0.0",
            "task_mapping": "1.0.0",
            "labelbox_mappings": []
        },
        "sampling_config": {
            "hash": "dummy_hash",
            "parameters": {}
        },
        "preprocess_pipeline": {
            "id": "full_frame_v1",
            "version": "1.0.0"
        },
        "splits": {
            "train": ["data/train-0000.parquet"],
            "val": ["data/val-0000.parquet"],
            "test": ["data/test-0000.parquet"]
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "generate_fixtures",
    }
    (base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy fixtures for testing")
    parser.add_argument("--output-dir", type=str, default="tests/fixtures/manifests/local_smoke",
                        help="Output directory for fixtures")
    parser.add_argument("--num-images", type=int, default=20,
                        help="Number of images per split")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of classes")
    
    args = parser.parse_args()
    
    base_path = Path(args.output_dir)
    
    # Create classification dataset
    cls_path = base_path / "classification"
    df_train = create_dummy_data(cls_path, num_rows=args.num_images, split="train")
    df_train.to_parquet(cls_path / "data/train-0000.parquet", index=False)
    
    df_val = create_dummy_data(cls_path, num_rows=args.num_images // 2, split="val")
    df_val.to_parquet(cls_path / "data/val-0000.parquet", index=False)
    
    df_test = create_dummy_data(cls_path, num_rows=args.num_images // 2, split="test")
    df_test.to_parquet(cls_path / "data/test-0000.parquet", index=False)
    
    create_manifest(cls_path, "synthetic_classification")
    
    # Create segmentation dataset
    seg_path = base_path / "segmentation"
    df_train = create_dummy_data(seg_path, num_rows=args.num_images, split="train")
    df_train.to_parquet(seg_path / "data/train-0000.parquet", index=False)
    
    df_val = create_dummy_data(seg_path, num_rows=args.num_images // 2, split="val")
    df_val.to_parquet(seg_path / "data/val-0000.parquet", index=False)
    
    df_test = create_dummy_data(seg_path, num_rows=args.num_images // 2, split="test")
    df_test.to_parquet(seg_path / "data/test-0000.parquet", index=False)
    
    create_manifest(seg_path, "synthetic_segmentation")
    
    print(f"Created dummy datasets in {base_path}")
    print(f"  - Classification: {cls_path}")
    print(f"  - Segmentation: {seg_path}")

