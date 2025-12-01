import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional

import jsonschema
import pandas as pd
import tensorflow as tf

from src.core.interfaces import DataLoader
from src.core.data.augmenter import Augmenter, NoOpAugmenter, ConfigurableAugmenter


log = logging.getLogger(__name__)

def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Loads and validates the dataset manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Load schema
    schema_path = Path(__file__).parents[1] / "contracts" / "dataset_manifest_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
        
    jsonschema.validate(instance=manifest, schema=schema)
    return manifest

def load_dataset_from_manifest_dir(
    manifest_dir: Path | str, 
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Loads a dataset from a manifest directory.
    
    Args:
        manifest_dir: Path to the directory containing manifest.json.
        split: Dataset split to load ('train', 'val', 'test').
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        
    Returns:
        A tf.data.Dataset yielding dictionaries of features.
    """
    manifest_dir = Path(manifest_dir)
    manifest_path = manifest_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
    manifest = load_manifest(manifest_path)
    
    if split not in manifest["splits"]:
        raise ValueError(f"Split '{split}' not found in manifest. Available: {list(manifest['splits'].keys())}")
        
    parquet_files = manifest["splits"][split]
    full_paths = [str(manifest_dir / p) for p in parquet_files]
    
    def generator() -> Generator[dict[str, Any], None, None]:
        for p_path in full_paths:
            df = pd.read_parquet(p_path)
            for _, row in df.iterrows():
                yield row.to_dict()

    # Determine output signature from the first file
    first_df = pd.read_parquet(full_paths[0])
    output_signature = {
        k: tf.TensorSpec(shape=(), dtype=tf.string if v == 'object' else tf.int64 if v == 'int64' else tf.float32)
        for k, v in first_df.dtypes.items()
    }
    
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    
    ds = ds.batch(batch_size)
    return ds

class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, features: dict[str, Any], cfg: Any) -> tuple[tf.Tensor, tf.Tensor]:
        pass

class ClassificationPreprocessor(Preprocessor):
    def preprocess(self, features: dict[str, Any], cfg: Any) -> tuple[tf.Tensor, tf.Tensor]:
        image_size = (int(cfg.data.dataset.image_size[0]), int(cfg.data.dataset.image_size[1]))
        num_classes = int(cfg.data.dataset.num_classes)
        
        uri = features['image_uri']
        path = tf.strings.regex_replace(uri, "^file://", "")
        
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0 
        
        label = tf.cast(features['label'], tf.int32)
        label_one_hot = tf.one_hot(label, depth=num_classes)
        
        return img, label_one_hot

class SegmentationPreprocessor(Preprocessor):
    def preprocess(self, features: dict[str, Any], cfg: Any) -> tuple[tf.Tensor, tf.Tensor]:
        image_size = (int(cfg.data.dataset.image_size[0]), int(cfg.data.dataset.image_size[1]))
        num_classes = int(cfg.data.dataset.num_classes)
        
        # Image
        img_uri = features['image_uri']
        img_path = tf.strings.regex_replace(img_uri, "^file://", "")
        img_bytes = tf.io.read_file(img_path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0

        # Mask
        mask_uri = features['mask_uri']
        mask_path = tf.strings.regex_replace(mask_uri, "^file://", "")
        mask_bytes = tf.io.read_file(mask_path)
        mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, image_size, method="nearest")
        mask = tf.cast(mask, tf.int32)
        mask = tf.squeeze(mask, axis=-1) # [H, W]
        mask_one_hot = tf.one_hot(mask, depth=num_classes) # [H, W, C]
        
        return img, mask_one_hot

# Imports are handled at the top of the file or locally to avoid circular deps if needed
# But here we are at module level.
# The previous tool call added the import before ManifestDataLoader definition.
# We need to make sure we don't have duplicate imports or missing ones.
# Let's just fix the previous import line.

class ManifestDataLoader(DataLoader):
    def __init__(self, preprocessor: Preprocessor, augmenter: Optional[Augmenter] = None):
        self.preprocessor = preprocessor
        self.augmenter = augmenter or NoOpAugmenter()

    def _load(self, cfg: Any, split: str) -> tf.data.Dataset:
        batch_size = int(cfg.data.dataset.batch_size)

        ds_raw = load_dataset_from_manifest_dir(
            manifest_dir=cfg.data.dataset.manifest_path,
            split=split,
            batch_size=batch_size,
        )
        
        # Preprocess (unbatch first to process individual items)
        ds = ds_raw.unbatch().map(
            lambda x: self.preprocessor.preprocess(x, cfg),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation if training
        if split == "train":
            ds = ds.map(
                lambda x, y: self.augmenter.augment(x, y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
        # Batch and prefetch
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def load_train(self, cfg: Any) -> tf.data.Dataset:
        return self._load(cfg, "train")

    def load_val(self, cfg: Any) -> tf.data.Dataset:
        return self._load(cfg, "val")

class SyntheticDataLoader(DataLoader):
    def _make_synthetic_tfdata(self, cfg: Any) -> tf.data.Dataset:
        image_size = (int(cfg.data.dataset.image_size[0]), int(cfg.data.dataset.image_size[1]))
        num_classes = int(cfg.data.dataset.num_classes)
        batch_size = int(cfg.data.dataset.batch_size)
        
        h, w = image_size
        x = tf.random.uniform((batch_size * 4, h, w, 3), dtype=tf.float32)
        
        if cfg.task.name == "segmentation":
             y = tf.one_hot(
                tf.random.uniform((batch_size * 4, h, w), maxval=num_classes, dtype=tf.int32),
                depth=num_classes,
            )
        elif cfg.task.name == "cropper":
            # fake bbox+conf targets [B, 5]
            y = tf.random.uniform((batch_size * 4, 5), dtype=tf.float32)
        else:
            y = tf.one_hot(tf.random.uniform((batch_size * 4,), maxval=num_classes, dtype=tf.int32), depth=num_classes)
            
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).prefetch(2)
        return ds

    def load_train(self, cfg: Any) -> tf.data.Dataset:
        return self._make_synthetic_tfdata(cfg)

    def load_val(self, cfg: Any) -> tf.data.Dataset:
        return self._make_synthetic_tfdata(cfg)



class DataLoaderFactory:
    @staticmethod
    def get_loader(cfg: Any) -> DataLoader:
        if cfg.data.dataset.mode == "manifest":
            task_name = str(cfg.task.name).lower()
            augmenter = ConfigurableAugmenter(cfg)
            
            if task_name == "segmentation":
                return ManifestDataLoader(SegmentationPreprocessor(), augmenter)
            else:
                # Check if medical preprocessing is requested
                preprocess_type = cfg.get("preprocess", {}).get("type", "standard")
                if preprocess_type == "medical":
                    from src.core.data.medical_preprocessing import MedicalPreprocessor
                    return ManifestDataLoader(MedicalPreprocessor(), augmenter)
                else:
                    return ManifestDataLoader(ClassificationPreprocessor(), augmenter)
        else:
            return SyntheticDataLoader()
