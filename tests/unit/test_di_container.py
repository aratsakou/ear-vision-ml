"""
Unit tests for Dependency Injection container and interfaces.
"""
import pytest
from unittest.mock import Mock
import tensorflow as tf

from src.core.di import Container, get_container
from src.core.interfaces import ModelBuilder, DataLoader, Trainer, Exporter


class TestDIContainer:
    """Test the dependency injection container."""
    
    def test_register_and_resolve_singleton(self):
        """Test registering and resolving singleton instances."""
        container = Container()
        
        # Create a mock service
        mock_builder = Mock(spec=ModelBuilder)
        
        # Register singleton
        container.register_singleton(ModelBuilder, mock_builder)
        
        # Resolve should return the same instance
        resolved1 = container.resolve(ModelBuilder)
        resolved2 = container.resolve(ModelBuilder)
        
        assert resolved1 is mock_builder
        assert resolved2 is mock_builder
        assert resolved1 is resolved2
    
    def test_register_and_resolve_factory(self):
        """Test registering and resolving factory functions."""
        container = Container()
        
        # Create a factory that returns new instances
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return Mock(spec=ModelBuilder, id=call_count)
        
        # Register factory
        container.register_factory(ModelBuilder, factory)
        
        # Each resolve should call the factory
        resolved1 = container.resolve(ModelBuilder)
        resolved2 = container.resolve(ModelBuilder)
        
        assert resolved1.id == 1
        assert resolved2.id == 2
        assert resolved1 is not resolved2
    
    def test_resolve_unregistered_service_raises_error(self):
        """Test that resolving unregistered service raises ValueError."""
        container = Container()
        
        with pytest.raises(ValueError, match="Service .* not registered"):
            container.resolve(ModelBuilder)
    
    def test_multiple_service_types(self):
        """Test registering multiple different service types."""
        container = Container()
        
        mock_builder = Mock(spec=ModelBuilder)
        mock_trainer = Mock(spec=Trainer)
        mock_exporter = Mock(spec=Exporter)
        
        container.register_singleton(ModelBuilder, mock_builder)
        container.register_singleton(Trainer, mock_trainer)
        container.register_singleton(Exporter, mock_exporter)
        
        assert container.resolve(ModelBuilder) is mock_builder
        assert container.resolve(Trainer) is mock_trainer
        assert container.resolve(Exporter) is mock_exporter
    
    def test_get_container_returns_global_instance(self):
        """Test that get_container returns the global container."""
        container1 = get_container()
        container2 = get_container()
        
        assert container1 is container2


class TestInterfaces:
    """Test that interfaces are properly defined."""
    
    def test_model_builder_interface(self):
        """Test ModelBuilder interface definition."""
        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            ModelBuilder()
        
        # Mock implementation should work
        mock_builder = Mock(spec=ModelBuilder)
        assert hasattr(mock_builder, 'build')
    
    def test_data_loader_interface(self):
        """Test DataLoader interface definition."""
        with pytest.raises(TypeError):
            DataLoader()
        
        mock_loader = Mock(spec=DataLoader)
        assert hasattr(mock_loader, 'load_train')
        assert hasattr(mock_loader, 'load_val')
    
    def test_trainer_interface(self):
        """Test Trainer interface definition."""
        with pytest.raises(TypeError):
            Trainer()
        
        mock_trainer = Mock(spec=Trainer)
        assert hasattr(mock_trainer, 'train')
    
    def test_exporter_interface(self):
        """Test Exporter interface definition."""
        with pytest.raises(TypeError):
            Exporter()
        
        mock_exporter = Mock(spec=Exporter)
        assert hasattr(mock_exporter, 'export')


class TestInterfaceContracts:
    """Test that implementations properly implement interfaces."""
    
    def test_model_builder_contract(self):
        """Test that ModelBuilder implementations have correct signature."""
        from src.core.models.factories.model_factory import RegistryModelBuilder
        
        builder = RegistryModelBuilder()
        assert isinstance(builder, ModelBuilder)
        assert callable(builder.build)
    
    def test_trainer_contract(self):
        """Test that Trainer implementations have correct signature."""
        from src.core.training.standard_trainer import StandardTrainer
        
        trainer = StandardTrainer()
        assert isinstance(trainer, Trainer)
        assert callable(trainer.train)
    
    def test_exporter_contract(self):
        """Test that Exporter implementations have correct signature."""
        from src.core.export.exporter import StandardExporter
        
        exporter = StandardExporter()
        assert isinstance(exporter, Exporter)
        assert callable(exporter.export)
    
    def test_data_loader_contract(self):
        """Test that DataLoader implementations have correct signature."""
        from src.core.data.dataset_loader import SyntheticDataLoader, ManifestDataLoader
        from src.core.data.dataset_loader import ClassificationPreprocessor
        
        synthetic_loader = SyntheticDataLoader()
        assert isinstance(synthetic_loader, DataLoader)
        assert callable(synthetic_loader.load_train)
        assert callable(synthetic_loader.load_val)
        
        manifest_loader = ManifestDataLoader(ClassificationPreprocessor())
        assert isinstance(manifest_loader, DataLoader)
        assert callable(manifest_loader.load_train)
        assert callable(manifest_loader.load_val)
