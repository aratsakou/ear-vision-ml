from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

class Container:
    def __init__(self):
        self._services: dict[type, Any] = {}
        self._factories: dict[type, Callable[[], Any]] = {}

    def register_singleton(self, interface: type[T], instance: T):
        self._services[interface] = instance

    def register_factory(self, interface: type[T], factory: Callable[[], T]):
        self._factories[interface] = factory

    def resolve(self, interface: type[T]) -> T:
        if interface in self._services:
            return self._services[interface]
        
        if interface in self._factories:
            instance = self._factories[interface]()
            # Optionally cache if we wanted lazy singletons, but for now let's keep it simple
            # self._services[interface] = instance 
            return instance
            
        raise ValueError(f"Service {interface} not registered")

# Global container instance for simplicity, though passing it around is cleaner
_container = Container()

def get_container() -> Container:
    return _container
