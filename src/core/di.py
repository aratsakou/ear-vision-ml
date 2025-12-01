from collections.abc import Callable
from enum import Enum, auto
from typing import Any, TypeVar, Type, Dict, get_type_hints
import atexit
import inspect
import logging
import threading

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
    Supports circular dependency detection.
    Can be used as a context manager.
    """
    def __init__(self):
        self._services: Dict[Type[T], Any] = {} # Cache for singletons
        self._factories: Dict[Type[T], Callable[..., T]] = {}
        self._scopes: Dict[Type[T], Scope] = {}
        self._request_cache: Dict[Type[T], Any] = {} # Cache for request scope
        self._components: list[Component] = [] # Track components for cleanup
        self._resolving: set[Type[T]] = set()  # For circular dependency detection
        self._shutdown_registered = False
        
        # Register cleanup on exit
        atexit.register(self._atexit_cleanup)
        self._shutdown_registered = True

    def _atexit_cleanup(self):
        """Called automatically on process exit."""
        if self._shutdown_registered:
            log.debug("Auto-cleanup on process exit")
            self.shutdown()
            self._shutdown_registered = False

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
        Resolve a service instance with circular dependency detection.
        """
        # Circular dependency detection
        if interface in self._resolving:
            cycle = ' -> '.join(str(t.__name__) for t in self._resolving) + f' -> {interface.__name__}'
            raise ValueError(f"Circular dependency detected: {cycle}")
        
        # 1. Check Singleton Cache
        if interface in self._services:
            return self._services[interface]
            
        # 2. Check Request Cache
        if interface in self._request_cache:
            return self._request_cache[interface]
            
        # 3. Create Instance
        if interface in self._factories:
            self._resolving.add(interface)
            try:
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
            finally:
                self._resolving.discard(interface)
            
        raise ValueError(f"Service {interface} not registered")

    def _create_instance(self, cls: Type[T], **kwargs) -> T:
        """Creates an instance of cls, auto-wiring dependencies from __init__."""
        type_hints = get_type_hints(cls.__init__)
        dependencies = {}
        
        for name, param_type in type_hints.items():
            if name == 'return':
                continue
            if name in kwargs:
                continue  # Manual override
            
            # Try to resolve dependency
            try:
                # Check if the param_type is registered
                if param_type in self._services or param_type in self._factories:
                     dependencies[name] = self.resolve(param_type)
            except ValueError as e:
                # Log optional dependency skip, but re-raise circular deps
                if "Circular dependency" in str(e):
                    raise
                log.debug(f"Could not auto-wire {name}: {param_type.__name__} - {e}")
                
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
        self._resolving.clear()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.shutdown()
        return False  # Don't suppress exceptions

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


class ContainerContext:
    """
    Thread-local container context for test isolation.
    """
    _local = threading.local()
    _use_thread_local = False  # Feature flag
    _global_container = None  # Will be set after first instantiation

    @classmethod
    def enable_thread_local(cls):
        """Enable thread-local containers (useful for testing)."""
        cls._use_thread_local = True

    @classmethod
    def disable_thread_local(cls):
        """Disable thread-local containers (default behavior)."""
        cls._use_thread_local = False

    @classmethod
    def get_container(cls) -> Container:
        """Get the appropriate container (thread-local or global)."""
        if cls._use_thread_local:
            if not hasattr(cls._local, 'container'):
                cls._local.container = Container()
            return cls._local.container
        
        # Global container
        if cls._global_container is None:
            cls._global_container = Container()
        return cls._global_container

    @classmethod
    def reset(cls):
        """Reset container (useful for testing)."""
        if cls._use_thread_local and hasattr(cls._local, 'container'):
            cls._local.container.shutdown()
            delattr(cls._local, 'container')
        elif not cls._use_thread_local and cls._global_container is not None:
            # For global container, just clear registrations but keep instance
            cls._global_container._services.clear()
            cls._global_container._factories.clear()
            cls._global_container._scopes.clear()


# Global container instance (backward compatibility)
_container = Container()

def get_container() -> Container:
    """
    Get the DI container.
    
    In production: Returns global singleton.
    In tests: Can use ContainerContext.enable_thread_local() for isolation.
    """
    return ContainerContext.get_container()
