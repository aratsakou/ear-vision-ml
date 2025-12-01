from collections.abc import Callable
from enum import Enum, auto
from typing import Any, TypeVar, Type, Dict, Optional, get_type_hints
import inspect
import logging

from src.core.interfaces import Component

T = TypeVar("T")
log = logging.getLogger(__name__)

class Scope(Enum):
    SINGLETON = auto()
    TRANSIENT = auto()
    REQUEST = auto()

class Container:
    """
    Dependency Injection Container.
    Supports singleton, transient, and request scopes.
    Supports lifecycle management (initialize/cleanup).
    Supports auto-wiring based on type hints.
    """
    def __init__(self):
        self._services: Dict[Type[T], Any] = {} # Cache for singletons
        self._factories: Dict[Type[T], Callable[..., T]] = {}
        self._scopes: Dict[Type[T], Scope] = {}
        self._request_cache: Dict[Type[T], Any] = {} # Cache for request scope
        self._components: list[Component] = [] # Track components for cleanup

    def register(self, interface: Type[T], implementation: Any, scope: Scope = Scope.SINGLETON):
        """
        Register a service.
        
        Args:
            interface: The interface/type to register.
            implementation: An instance (for singleton), a class, or a factory function.
            scope: The lifecycle scope.
        """
        if scope == Scope.SINGLETON and not callable(implementation):
            # Pre-instantiated singleton
            self._services[interface] = implementation
            if isinstance(implementation, Component):
                implementation.initialize()
                self._components.append(implementation)
        else:
            # Factory or Class
            self._factories[interface] = implementation
            
        self._scopes[interface] = scope

    def register_singleton(self, interface: Type[T], instance: T):
        """Legacy support: Register a pre-instantiated object as a singleton."""
        self.register(interface, instance, Scope.SINGLETON)

    def register_factory(self, interface: Type[T], factory: Callable[..., T]):
        """Legacy support: Register a factory function (transient)."""
        self.register(interface, factory, Scope.TRANSIENT)

    def resolve(self, interface: Type[T], **kwargs) -> T:
        """
        Resolve a service instance.
        """
        # 1. Check Singleton Cache
        if interface in self._services:
            return self._services[interface]
            
        # 2. Check Request Cache
        if interface in self._request_cache:
            return self._request_cache[interface]
            
        # 3. Create Instance
        if interface in self._factories:
            factory = self._factories[interface]
            scope = self._scopes[interface]
            
            # Auto-wiring logic if it's a class
            if inspect.isclass(factory):
                instance = self._create_instance(factory, **kwargs)
            else:
                instance = factory(**kwargs)
                
            # Lifecycle hook
            if isinstance(instance, Component):
                instance.initialize()
                self._components.append(instance)
                
            # Caching based on scope
            if scope == Scope.SINGLETON:
                self._services[interface] = instance
            elif scope == Scope.REQUEST:
                self._request_cache[interface] = instance
                
            return instance
            
        raise ValueError(f"Service {interface} not registered")

    def _create_instance(self, cls: Type[T], **kwargs) -> T:
        """Creates an instance of cls, auto-wiring dependencies from __init__."""
        type_hints = get_type_hints(cls.__init__)
        dependencies = {}
        
        for name, param_type in type_hints.items():
            if name == 'return': continue
            if name in kwargs: continue # Manual override
            
            # Try to resolve dependency
            try:
                # Check if the param_type is registered
                if param_type in self._services or param_type in self._factories:
                     dependencies[name] = self.resolve(param_type)
            except ValueError:
                pass # Optional dependency or primitive
                
        # Merge auto-wired deps with manual kwargs
        final_kwargs = {**dependencies, **kwargs}
        return cls(**final_kwargs)

    def begin_request(self):
        """Start a new request scope."""
        self._request_cache.clear()

    def end_request(self):
        """End current request scope."""
        self._request_cache.clear()

    def shutdown(self):
        """Cleanup all components."""
        for component in reversed(self._components):
            try:
                component.cleanup()
            except Exception as e:
                log.error(f"Error cleaning up component {component}: {e}")
        self._components.clear()
        self._services.clear()
        self._request_cache.clear()

    def load_config(self, config: Dict[str, Any]):
        """
        Load services from configuration dictionary.
        Format:
        {
            "interface_path": {
                "impl": "implementation_path",
                "scope": "SINGLETON" | "TRANSIENT" | "REQUEST"
            }
        }
        """
        import importlib
        
        for interface_path, details in config.items():
            # Resolve interface
            module_name, class_name = interface_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            interface = getattr(module, class_name)
            
            # Resolve implementation
            impl_path = details["impl"]
            module_name, class_name = impl_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            implementation = getattr(module, class_name)
            
            # Resolve scope
            scope_str = details.get("scope", "SINGLETON").upper()
            scope = Scope[scope_str]
            
            self.register(interface, implementation, scope)

# Global container instance
_container = Container()

def get_container() -> Container:
    return _container
