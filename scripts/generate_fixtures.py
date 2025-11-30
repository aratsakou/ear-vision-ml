from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def create_dummy_data(base_dir: Path, num_rows=10):
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
        img_path = images_dir / f"img_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        image_uris.append(f"file://{img_path.absolute()}")
        
        # Create a dummy mask (grayscale, values 0..2)
        mask = np.random.randint(0, 3, (224, 224), dtype=np.uint8)
        mask_path = masks_dir / f"mask_{i}.png"
        cv2.imwrite(str(mask_path), mask)
        mask_uris.append(f"file://{mask_path.absolute()}")

    df = pd.DataFrame({
        'image_uri': image_uris,
        'mask_uri': mask_uris,
        'timestamp_ms': [i * 1000 for i in range(num_rows)],
        'label': np.random.randint(0, 3, size=num_rows),
        'split': ['train'] * num_rows
    })
    return df

base_path = Path("tests/fixtures/manifests/local_smoke")
df_train = create_dummy_data(base_path, num_rows=10)
df_train.to_parquet(base_path / "data/train-0000.parquet")

df_val = create_dummy_data(base_path, num_rows=5)
df_val.to_parquet(base_path / "data/val-0000.parquet")

df_test = create_dummy_data(base_path, num_rows=5)
df_test.to_parquet(base_path / "data/test-0000.parquet")

print("Created dummy parquet files, images, and masks.")
