"""
Unit tests for advanced DI container features:
- Circular dependency detection
- Resource cleanup
- Thread-local isolation
- Context manager support
"""
import pytest
import threading
import time
from unittest.mock import Mock

from src.core.di import Container, ContainerContext, Scope
from src.core.interfaces import Component


class MockComponent(Component):
    """Mock component for testing lifecycle."""
    def __init__(self):
        self.initialized = False
        self.cleaned_up = False
    
    def initialize(self):
        self.initialized = True
    
    def cleanup(self):
        self.cleaned_up = True


class ServiceA:
    """Test service A."""
    def __init__(self, b: 'ServiceB' = None):
        self.b = b


class ServiceB:
    """Test service B."""
    def __init__(self, a: ServiceA = None):
        self.a = a


def test_circular_dependency_detection():
    """Test that circular dependencies are detected and reported."""
    container = Container()
    
    # Register services with circular dependency
    container.register(ServiceA, ServiceA, Scope.TRANSIENT)
    container.register(ServiceB, ServiceB, Scope.TRANSIENT)
    
    # Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="Circular dependency"):
        container.resolve(ServiceA)


def test_resource_cleanup_on_shutdown():
    """Test that components are cleaned up on shutdown."""
    container = Container()
    
    # Create and register component
    comp = MockComponent()
    container.register(MockComponent, comp, Scope.SINGLETON)
    
    # Verify initialized
    resolved = container.resolve(MockComponent)
    assert resolved.initialized
    assert not resolved.cleaned_up
    
    # Shutdown should trigger cleanup
    container.shutdown()
    assert comp.cleaned_up


def test_context_manager():
    """Test container as context manager."""
    comp = MockComponent()
    
    with Container() as container:
        container.register(MockComponent, comp, Scope.SINGLETON)
        resolved = container.resolve(MockComponent)
        assert resolved.initialized
        assert not resolved.cleaned_up
    
    # Should be cleaned up after exiting context
    assert comp.cleaned_up


def test_thread_local_isolation():
    """Test that thread-local containers are isolated."""
    ContainerContext.enable_thread_local()
    
    try:
        results = {}
        
        def thread_task(thread_id):
            """Task for each thread."""
            container = ContainerContext.get_container()
            
            # Register a different value in each thread
            class TestService:
                def __init__(self, value):
                    self.value = value
            
            service = TestService(thread_id)
            container.register(TestService, service, Scope.SINGLETON)
            
            # Resolve and store
            resolved = container.resolve(TestService)
            results[thread_id] = resolved.value
        
        # Create multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_task, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Each thread should have its own value
        assert results == {0: 0, 1: 1, 2: 2}
        
    finally:
        ContainerContext.disable_thread_local()
        ContainerContext.reset()


def test_reset_clears_container():
    """Test that reset clears container state."""
    ContainerContext.enable_thread_local()
    
    try:
        container = ContainerContext.get_container()
        
        class TestService:
            pass
        
        service = TestService()
        container.register(TestService, service, Scope.SINGLETON)
        
        # Should resolve
        assert container.resolve(TestService) is service
        
        # Reset
        ContainerContext.reset()
        
        # New container should not have the service
        new_container = ContainerContext.get_container()
        with pytest.raises(ValueError, match="not registered"):
            new_container.resolve(TestService)
    
    finally:
        ContainerContext.disable_thread_local()
        ContainerContext.reset()


def test_exception_handling_in_autowiring():
    """Test that auto-wiring handles exceptions gracefully for unregistered services."""
    container = Container()
    
    class UnregisteredService:
        pass
    
    class ServiceWithOptionalDep:
        def __init__(self, dep: UnregisteredService = None):
            self.dep = dep
    
    # Register only the service with dependency (dep is not registered)
    container.register(ServiceWithOptionalDep, ServiceWithOptionalDep, Scope.TRANSIENT)
    
    # Should create instance without the dependency since it's optional
    instance = container.resolve(ServiceWithOptionalDep)
    assert instance is not None
    assert instance.dep is None  # Dependency should not be auto-wired


def test_component_lifecycle_hooks():
    """Test that component lifecycle hooks are called correctly."""
    container = Container()
    
    comp = MockComponent()
    container.register(MockComponent, comp, Scope.SINGLETON)
    
    # Initialize should be called after registration
    assert comp.initialized
    
    # Cleanup should be called on shutdown
    container.shutdown()
    assert comp.cleaned_up


def test_multiple_components_cleanup_order():
    """Test that multiple components are cleaned up in reverse order."""
    container = Container()
    cleanup_order = []
    
    class OrderedComponent(Component):
        def __init__(self, name):
            self.name = name
        
        def initialize(self):
            pass
        
        def cleanup(self):
            cleanup_order.append(self.name)
    
    # Register multiple components
    comp1 = OrderedComponent("first")
    comp2 = OrderedComponent("second")
    comp3 = OrderedComponent("third")
    
    container.register(type(comp1), comp1, Scope.SINGLETON)
    container.resolve(type(comp1))
    
    container.register(type(comp2), comp2, Scope.SINGLETON)
    container.resolve(type(comp2))
    
    container.register(type(comp3), comp3, Scope.SINGLETON)
    container.resolve(type(comp3))
    
    # Shutdown should cleanup in reverse order
    container.shutdown()
    assert cleanup_order == ["third", "second", "first"]


def test_circular_dependency_error_message():
    """Test that circular dependency error has helpful message."""
    container = Container()
    
    container.register(ServiceA, ServiceA, Scope.TRANSIENT)
    container.register(ServiceB, ServiceB, Scope.TRANSIENT)
    
    try:
        container.resolve(ServiceA)
        pytest.fail("Expected ValueError for circular dependency")
    except ValueError as e:
        # Error message should include both service names
        assert "ServiceA" in str(e) or "ServiceB" in str(e)
        assert "Circular dependency" in str(e)
