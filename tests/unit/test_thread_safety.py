"""
Thread safety tests for model registry and DI container request scope.
"""
import pytest
import threading
import time
from unittest.mock import Mock
from omegaconf import OmegaConf

from src.core.di import Container, get_container, Scope
from src.core.models.factories.model_factory import RegistryModelBuilder, register_model
import tensorflow as tf


def test_concurrent_model_registration():
    """Test that concurrent model registration is thread-safe."""
    builder = RegistryModelBuilder()
    errors = []
    
    def register_models(thread_id):
        """Register models from multiple threads."""
        try:
            for i in range(10):
                model_name = f"test_model_{thread_id}_{i}"
                
                def make_builder(name=model_name):
                    def builder_fn(cfg):
                        return Mock(name=name)
                    return builder_fn
                
                builder.register(model_name, make_builder())
        except Exception as e:
            errors.append(str(e))
    
    # Create multiple threads registering models
    threads = []
    for i in range(5):
        t = threading.Thread(target=register_models, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Should not have errors
    assert len(errors) == 0, f"Errors during concurrent registration: {errors}"


def test_concurrent_model_building():
    """Test that concurrent model building is thread-safe."""
    builder = RegistryModelBuilder()
    results = {}
    errors = []
    
    # Register a test model
    def test_model_builder(cfg):
        # Simulate some work
        time.sleep(0.01)
        return Mock(name=f"model_{cfg.model.name}")
    
    builder.register("test_model", test_model_builder)
    
    def build_model(thread_id):
        """Build model from multiple threads."""
        try:
            cfg = OmegaConf.create({
                "model": {
                    "name": "test_model"
                }
            })
            model = builder.build(cfg)
            results[thread_id] = model.name
        except Exception as e:
            errors.append(str(e))
    
    #Create multiple threads building models
    threads = []
    for i in range(10):
        t = threading.Thread(target=build_model, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Should not have errors
    assert len(errors) == 0, f"Errors during concurrent building: {errors}"
    # All threads should have built successfully
    assert len(results) == 10


def test_request_scope_thread_safety():
    """Test that request scope operations don't cause race conditions."""
    container = Container()
    counter = {'value': 0}
    errors = []
    
    class RequestScopedService:
        def __init__(self):
            counter['value'] += 1
            self.instance_id = counter['value']
    
    container.register(RequestScopedService, RequestScopedService, Scope.REQUEST)
    
    def use_request_scope():
        """Use request scope from multiple threads."""
        try:
            # Each thread does multiple begin/end cycles
            for _ in range(10):
                container.begin_request()
                try:
                    # Resolve service - should be cached within request
                    service1 = container.resolve(RequestScopedService)
                    service2 = container.resolve(RequestScopedService)
                    # Within same request, should be same instance
                    assert service1 is service2
                finally:
                    container.end_request()
        except Exception as e:
            errors.append(str(e))
    
    # Create multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=use_request_scope)
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Should not have errors (no race conditions)
    assert len(errors) == 0, f"Errors during request scope ops: {errors}"


def test_concurrent_request_scope_clear():
    """Test that concurrent clear operations don't cause race conditions."""
    container = Container()
    errors = []
    
    class TestService:
        pass
    
    container.register(TestService, TestService, Scope.REQUEST)
    
    def stress_test_request_scope():
        """Stress test request scope with many operations."""
        try:
            for _ in range(100):
                container.begin_request()
                try:
                    container.resolve(TestService)
                finally:
                    container.end_request()
        except Exception as e:
            errors.append(str(e))
    
    # Create many threads
    threads = []
    for _ in range(10):
        t = threading.Thread(target=stress_test_request_scope)
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Should not have errors
    assert len(errors) == 0, f"Errors during stress test: {errors}"


def test_model_registry_isolation():
    """Test that registry operations don't interfere across threads."""
    builder = RegistryModelBuilder()
    
    # Pre-register some models
    for i in range(5):
        builder.register(f"base_model_{i}", lambda cfg, i=i: Mock(name=f"base_{i}"))
    
    errors = []
    
    def concurrent_operations(thread_id):
        """Mix registration and building."""
        try:
            # Register new model
            builder.register(f"thread_model_{thread_id}", 
                           lambda cfg, tid=thread_id: Mock(name=f"thread_{tid}"))
            
            # Build existing model
            cfg = OmegaConf.create({"model": {"name": f"base_model_0"}})
            builder.build(cfg)
            
            # Build newly registered model
            cfg2 = OmegaConf.create({"model": {"name": f"thread_model_{thread_id}"}})
            builder.build(cfg2)
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    # Run concurrent operations
    threads = []
    for i in range(10):
        t = threading.Thread(target=concurrent_operations, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Should not have errors
    assert len(errors) == 0, f"Errors: {errors}"
