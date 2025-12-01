from collections.abc import Callable
from typing import Any, TypeVar, Type, Dict, Optional

T = TypeVar("T")

class Container:
    """
    Dependency Injection Container.
    Supports singleton and transient (factory) lifecycles.
    """
    def __init__(self):
        self._services: Dict[Type[T], Any] = {}
        self._factories: Dict[Type[T], Callable[..., T]] = {}

    def register_singleton(self, interface: Type[T], instance: T):
        """Register a pre-instantiated object as a singleton."""
        self._services[interface] = instance

    def register_factory(self, interface: Type[T], factory: Callable[..., T]):
        """Register a factory function that creates a new instance each time."""
        self._factories[interface] = factory

    def resolve(self, interface: Type[T], **kwargs) -> T:
        """
        Resolve a service instance.
        If registered as singleton, returns the cached instance.
        If registered as factory, calls the factory with optional kwargs.
        """
        if interface in self._services:
            return self._services[interface]
        
        if interface in self._factories:
            return self._factories[interface](**kwargs)
            
        raise ValueError(f"Service {interface} not registered")

# Global container instance
_container = Container()

def get_container() -> Container:
    return _container
