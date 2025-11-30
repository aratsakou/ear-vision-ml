"""
Unit tests for Data Loader Strategy Pattern.
"""
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from src.core.data.dataset_loader import (
    Preprocessor,
    ClassificationPreprocessor,
    SegmentationPreprocessor,
    ManifestDataLoader,
    SyntheticDataLoader,
    DataLoaderFactory
)


class TestPreprocessorStrategy:
    """Test the preprocessor strategy pattern."""
    
    def test_preprocessor_is_abstract(self):
        """Test that Preprocessor cannot be instantiated."""
        with pytest.raises(TypeError):
            Preprocessor()
    
    def test_classification_preprocessor_exists(self):
        """Test that ClassificationPreprocessor implements Preprocessor."""
        preprocessor = ClassificationPreprocessor()
        assert isinstance(preprocessor, Preprocessor)
        assert hasattr(preprocessor, 'preprocess')
        assert callable(preprocessor.preprocess)
    
    def test_segmentation_preprocessor_exists(self):
        """Test that SegmentationPreprocessor implements Preprocessor."""
        preprocessor = SegmentationPreprocessor()
        assert isinstance(preprocessor, Preprocessor)
        assert hasattr(preprocessor, 'preprocess')
        assert callable(preprocessor.preprocess)


class TestSyntheticDataLoader:
    """Test the synthetic data loader."""
    
    def test_synthetic_loader_classification(self):
        """Test synthetic data loader for classification task."""
        cfg = OmegaConf.create({
            "task": {"name": "classification"},
            "data": {
                "dataset": {
                    "image_size": [224, 224],
                    "num_classes": 3,
                    "batch_size": 4
                }
            }
        })
        
        loader = SyntheticDataLoader()
        ds_train = loader.load_train(cfg)
        ds_val = loader.load_val(cfg)
        
        # Check that datasets are created
        assert isinstance(ds_train, tf.data.Dataset)
        assert isinstance(ds_val, tf.data.Dataset)
        
        # Check batch shape
        for images, labels in ds_train.take(1):
            assert images.shape == (4, 224, 224, 3)
            assert labels.shape == (4, 3)  # One-hot encoded
    
    def test_synthetic_loader_segmentation(self):
        """Test synthetic data loader for segmentation task."""
        cfg = OmegaConf.create({
            "task": {"name": "segmentation"},
            "data": {
                "dataset": {
                    "image_size": [256, 256],
                    "num_classes": 2,
                    "batch_size": 2
                }
            }
        })
        
        loader = SyntheticDataLoader()
        ds_train = loader.load_train(cfg)
        
        # Check batch shape for segmentation
        for images, masks in ds_train.take(1):
            assert images.shape == (2, 256, 256, 3)
            assert masks.shape == (2, 256, 256, 2)  # Pixel-wise one-hot
    
    def test_synthetic_loader_cropper(self):
        """Test synthetic data loader for cropper task."""
        cfg = OmegaConf.create({
            "task": {"name": "cropper"},
            "data": {
                "dataset": {
                    "image_size": [224, 224],
                    "num_classes": 2,  # Not used for cropper
                    "batch_size": 4
                }
            }
        })
        
        loader = SyntheticDataLoader()
        ds_train = loader.load_train(cfg)
        
        # Check batch shape for cropper
        for images, bboxes in ds_train.take(1):
            assert images.shape == (4, 224, 224, 3)
            assert bboxes.shape == (4, 5)  # [x1, y1, x2, y2, conf]


class TestDataLoaderFactory:
    """Test the data loader factory."""
    
    def test_factory_returns_synthetic_loader(self):
        """Test that factory returns SyntheticDataLoader for synthetic mode."""
        cfg = OmegaConf.create({
            "task": {"name": "classification"},
            "data": {
                "dataset": {
                    "mode": "synthetic",
                    "image_size": [224, 224],
                    "num_classes": 3,
                    "batch_size": 4
                }
            }
        })
        
        loader = DataLoaderFactory.get_loader(cfg)
        assert isinstance(loader, SyntheticDataLoader)
    
    def test_factory_returns_manifest_loader_for_classification(self):
        """Test that factory returns ManifestDataLoader with ClassificationPreprocessor."""
        cfg = OmegaConf.create({
            "task": {"name": "classification"},
            "data": {
                "dataset": {
                    "mode": "manifest",
                    "manifest_path": "/tmp/test",
                    "image_size": [224, 224],
                    "num_classes": 3,
                    "batch_size": 4
                }
            }
        })
        
        loader = DataLoaderFactory.get_loader(cfg)
        assert isinstance(loader, ManifestDataLoader)
        assert isinstance(loader.preprocessor, ClassificationPreprocessor)
    
    def test_factory_returns_manifest_loader_for_segmentation(self):
        """Test that factory returns ManifestDataLoader with SegmentationPreprocessor."""
        cfg = OmegaConf.create({
            "task": {"name": "segmentation"},
            "data": {
                "dataset": {
                    "mode": "manifest",
                    "manifest_path": "/tmp/test",
                    "image_size": [256, 256],
                    "num_classes": 2,
                    "batch_size": 2
                }
            }
        })
        
        loader = DataLoaderFactory.get_loader(cfg)
        assert isinstance(loader, ManifestDataLoader)
        assert isinstance(loader.preprocessor, SegmentationPreprocessor)
    
    def test_loader_implements_interface(self):
        """Test that all loaders implement DataLoader interface."""
        from src.core.interfaces import DataLoader
        
        synthetic = SyntheticDataLoader()
        manifest = ManifestDataLoader(ClassificationPreprocessor())
        
        assert isinstance(synthetic, DataLoader)
        assert isinstance(manifest, DataLoader)


class TestDataLoaderConsistency:
    """Test that data loaders produce consistent outputs."""
    
    def test_train_and_val_have_same_structure(self):
        """Test that train and val datasets have the same structure."""
        cfg = OmegaConf.create({
            "task": {"name": "classification"},
            "data": {
                "dataset": {
                    "mode": "synthetic",
                    "image_size": [224, 224],
                    "num_classes": 3,
                    "batch_size": 4
                }
            }
        })
        
        loader = SyntheticDataLoader()
        ds_train = loader.load_train(cfg)
        ds_val = loader.load_val(cfg)
        
        # Get one batch from each
        train_batch = next(iter(ds_train))
        val_batch = next(iter(ds_val))
        
        # Check shapes match
        assert train_batch[0].shape == val_batch[0].shape
        assert train_batch[1].shape == val_batch[1].shape
