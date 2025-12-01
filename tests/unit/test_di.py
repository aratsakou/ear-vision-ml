import pytest
from typing import Optional
from src.core.di import Container, Scope, Component

# --- Mocks ---

class IService:
    pass

class ServiceImpl(IService, Component):
    def __init__(self):
        self.initialized = False
        self.cleaned_up = False

    def initialize(self):
        self.initialized = True

    def cleanup(self):
        self.cleaned_up = True

class IDependentService:
    pass

class DependentService(IDependentService):
    def __init__(self, service: IService):
        self.service = service

class IOptionalService:
    pass

class OptionalDependentService:
    def __init__(self, service: IService, optional: Optional[IOptionalService] = None):
        self.service = service
        self.optional = optional

# --- Tests ---

def test_singleton_scope():
    container = Container()
    container.register(IService, ServiceImpl, Scope.SINGLETON)
    
    s1 = container.resolve(IService)
    s2 = container.resolve(IService)
    
    assert isinstance(s1, ServiceImpl)
    assert s1 is s2
    assert s1.initialized

def test_transient_scope():
    container = Container()
    container.register(IService, ServiceImpl, Scope.TRANSIENT)
    
    s1 = container.resolve(IService)
    s2 = container.resolve(IService)
    
    assert isinstance(s1, ServiceImpl)
    assert s1 is not s2
    assert s1.initialized

def test_request_scope():
    container = Container()
    container.register(IService, ServiceImpl, Scope.REQUEST)
    
    container.begin_request()
    s1 = container.resolve(IService)
    s2 = container.resolve(IService)
    assert s1 is s2
    
    container.end_request()
    container.begin_request()
    s3 = container.resolve(IService)
    assert s1 is not s3

def test_lifecycle_cleanup():
    container = Container()
    container.register(IService, ServiceImpl, Scope.SINGLETON)
    
    s1 = container.resolve(IService)
    assert not s1.cleaned_up
    
    container.shutdown()
    assert s1.cleaned_up

def test_auto_wiring():
    container = Container()
    container.register(IService, ServiceImpl, Scope.SINGLETON)
    container.register(IDependentService, DependentService, Scope.TRANSIENT)
    
    ds = container.resolve(IDependentService)
    assert isinstance(ds, DependentService)
    assert isinstance(ds.service, ServiceImpl)

def test_auto_wiring_optional():
    container = Container()
    container.register(IService, ServiceImpl, Scope.SINGLETON)
    # IOptionalService not registered
    
    # Register the class directly as a factory (or transient)
    container.register(OptionalDependentService, OptionalDependentService, Scope.TRANSIENT)
    
    ods = container.resolve(OptionalDependentService)
    assert isinstance(ods.service, ServiceImpl)
    assert ods.optional is None

def test_config_loading(tmp_path):
    # Create a dummy module structure for testing dynamic import
    # For simplicity, we'll use existing classes in this file, but we need them to be importable.
    # Since this test file might not be in python path, we can mock importlib or use real classes from src.
    
    # Let's use real classes from src.core.interfaces for the test
    config = {
        "src.core.interfaces.Trainer": {
            "impl": "src.core.training.standard_trainer.StandardTrainer",
            "scope": "SINGLETON"
        }
    }
    
    container = Container()
    container.load_config(config)
    
    from src.core.interfaces import Trainer
    from src.core.training.standard_trainer import StandardTrainer
    
    t = container.resolve(Trainer)
    assert isinstance(t, StandardTrainer)
