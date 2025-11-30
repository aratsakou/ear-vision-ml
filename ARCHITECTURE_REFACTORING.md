# Architecture Refactoring: Dependency Injection & Design Patterns

## Overview

This document describes the comprehensive architecture refactoring implemented to improve code modularity, testability, and maintainability through dependency injection and state-of-the-art design patterns.

## Key Improvements

### 1. Dependency Injection Container

**Location:** `src/core/di.py`

Implemented a lightweight dependency injection container that supports:
- **Singleton registration**: Register pre-instantiated objects
- **Factory registration**: Register factory functions for lazy instantiation
- **Type-safe resolution**: Resolve dependencies by interface type

**Benefits:**
- Decouples component creation from usage
- Enables easy mocking for testing
- Centralizes dependency configuration
- Supports interface-based programming

**Example Usage:**
```python
from src.core.di import get_container
from src.core.interfaces import ModelBuilder

container = get_container()
container.register_singleton(ModelBuilder, my_builder_instance)
builder = container.resolve(ModelBuilder)
```

### 2. Interface Definitions

**Location:** `src/core/interfaces.py`

Defined clear contracts for core components:

#### ModelBuilder Interface
```python
class ModelBuilder(ABC):
    @abstractmethod
    def build(self, cfg: Any) -> tf.keras.Model:
        pass
```

#### DataLoader Interface
```python
class DataLoader(ABC):
    @abstractmethod
    def load_train(self, cfg: Any) -> tf.data.Dataset:
        pass
    
    @abstractmethod
    def load_val(self, cfg: Any) -> tf.data.Dataset:
        pass
```

#### Trainer Interface
```python
class Trainer(ABC):
    @abstractmethod
    def train(self, model: tf.keras.Model, train_ds: tf.data.Dataset, 
              val_ds: tf.data.Dataset, cfg: Any) -> Any:
        pass
```

#### Exporter Interface
```python
class Exporter(ABC):
    @abstractmethod
    def export(self, model: tf.keras.Model, cfg: Any, 
               artifacts_dir: Any) -> Dict[str, Any]:
        pass
```

**Benefits:**
- Clear contracts for all major components
- Enables polymorphism and substitutability
- Improves code documentation
- Facilitates testing with mock implementations

### 3. Registry Pattern for Model Factory

**Location:** `src/core/models/factories/model_factory.py`

Refactored the model factory from a monolithic if-else chain to a registry-based system:

**Before:**
```python
def build_model(cfg: Any) -> tf.keras.Model:
    model_type = str(cfg.model.type).lower()
    name = str(cfg.model.name).lower()
    
    if model_type == "classification":
        if name == "cls_mobilenetv3":
            return build_cls_mobilenetv3(cfg)
        if name == "cls_efficientnetb0":
            return build_cls_efficientnetb0(cfg)
        # ... many more if statements
```

**After:**
```python
@register_model("cls_mobilenetv3")
def build_cls_mobilenetv3(cfg: Any) -> tf.keras.Model:
    # implementation
    
@register_model("cls_efficientnetb0")
def build_cls_efficientnetb0(cfg: Any) -> tf.keras.Model:
    # implementation
```

**Benefits:**
- **Open/Closed Principle**: Add new models without modifying existing code
- **Self-documenting**: Model registration happens at definition site
- **Better error messages**: Registry can list available models
- **Easier testing**: Can register test models dynamically

### 4. Strategy Pattern for Data Loading

**Location:** `src/core/data/dataset_loader.py`

Implemented preprocessor strategies for different task types:

**Architecture:**
```
Preprocessor (ABC)
├── ClassificationPreprocessor
└── SegmentationPreprocessor

DataLoader (ABC)
├── ManifestDataLoader (uses Preprocessor)
└── SyntheticDataLoader

DataLoaderFactory (selects appropriate loader)
```

**Benefits:**
- **Separation of Concerns**: Preprocessing logic separated from data loading
- **Extensibility**: Easy to add new preprocessing strategies
- **Reusability**: Preprocessors can be composed and reused
- **Task-specific logic**: Each task has its own preprocessing pipeline

### 5. Unified Trainer Implementation

**Location:** `src/core/training/standard_trainer.py`

Created a single `StandardTrainer` that handles all task types:

**Features:**
- Task-specific loss selection (classification, segmentation, cropper)
- Task-specific metrics configuration
- Unified training flow
- Centralized callback management

**Benefits:**
- **DRY Principle**: Single source of truth for training logic
- **Consistency**: All tasks use the same training flow
- **Maintainability**: Changes to training logic only need to happen in one place
- **Testability**: Single component to test instead of multiple trainers

### 6. Standardized Exporter

**Location:** `src/core/export/exporter.py`

Implemented `StandardExporter` class that implements the `Exporter` interface:

**Features:**
- Multi-format export (SavedModel, TFLite variants)
- Automatic benchmarking
- Manifest generation
- Backward compatibility with functional API

**Benefits:**
- **Interface compliance**: Can be swapped with other exporters
- **Testability**: Easy to mock for testing
- **Extensibility**: Can add new export formats by extending the class

## Refactored Components

### Task Trainers

All task trainers now follow the same pattern:

**Classification** (`src/tasks/classification/trainer.py`):
```python
def configure_services():
    container = get_container()
    container.register_singleton(ModelBuilder, global_model_builder)
    container.register_singleton(Exporter, StandardExporter())
    container.register_singleton(Trainer, StandardTrainer())

def run_classification(cfg: Any) -> None:
    configure_services()
    container = get_container()
    
    # Resolve dependencies
    model_builder = container.resolve(ModelBuilder)
    trainer = container.resolve(Trainer)
    exporter = container.resolve(Exporter)
    data_loader = DataLoaderFactory.get_loader(cfg)
    
    # Execute pipeline
    model = model_builder.build(cfg)
    ds_train = data_loader.load_train(cfg)
    ds_val = data_loader.load_val(cfg)
    result = trainer.train(model, ds_train, ds_val, cfg)
    paths_dict = exporter.export(model, cfg, ctx.artifacts_dir)
```

**Segmentation** (`src/tasks/segmentation/trainer.py`): Same pattern
**Cropper** (`src/tasks/cropper/trainer.py`): Same pattern

## Design Patterns Applied

### 1. **Dependency Injection**
- Components receive dependencies rather than creating them
- Enables loose coupling and easier testing

### 2. **Registry Pattern**
- Model factory uses registration for extensibility
- Follows Open/Closed Principle

### 3. **Strategy Pattern**
- Preprocessors encapsulate task-specific preprocessing logic
- Allows runtime selection of preprocessing strategy

### 4. **Factory Pattern**
- `DataLoaderFactory` creates appropriate data loaders
- Encapsulates object creation logic

### 5. **Template Method Pattern**
- `StandardTrainer` provides template for training flow
- Task-specific variations handled through configuration

### 6. **Interface Segregation**
- Small, focused interfaces for each component type
- Components only depend on what they need

## Testing Improvements

The new architecture significantly improves testability:

### Unit Testing
```python
# Mock dependencies easily
mock_builder = Mock(spec=ModelBuilder)
mock_trainer = Mock(spec=Trainer)
container.register_singleton(ModelBuilder, mock_builder)
container.register_singleton(Trainer, mock_trainer)
```

### Integration Testing
```python
# Test with real implementations
container.register_singleton(ModelBuilder, RegistryModelBuilder())
container.register_singleton(Trainer, StandardTrainer())
# Run full pipeline
```

## Migration Guide

### Adding a New Model

**Before:**
1. Edit `model_factory.py`
2. Add if statement in `build_model()`
3. Implement builder function

**After:**
1. Add builder function with `@register_model()` decorator
```python
@register_model("my_new_model")
def build_my_new_model(cfg: Any) -> tf.keras.Model:
    # implementation
```

### Adding a New Task

**Before:**
- Copy-paste existing task trainer
- Modify data loading logic
- Modify training logic

**After:**
1. Create preprocessor if needed:
```python
class MyTaskPreprocessor(Preprocessor):
    def preprocess(self, features, cfg):
        # implementation
```

2. Update `DataLoaderFactory` to handle new task
3. Update `StandardTrainer` to handle new task loss/metrics
4. Create task trainer using standard pattern

### Adding a New Export Format

**Before:**
- Modify `export_model()` function

**After:**
1. Extend `StandardExporter` or create new `Exporter` implementation
2. Register in DI container

## Performance Considerations

The refactoring maintains performance while improving architecture:

- **No runtime overhead**: DI resolution happens once at startup
- **Same execution path**: Training/inference logic unchanged
- **Lazy loading**: Factory pattern allows lazy instantiation
- **Memory efficient**: Singleton pattern prevents duplicate instances

## Backward Compatibility

All existing functional APIs are maintained:

```python
# Old API still works
from src.core.models.factories.model_factory import build_model
model = build_model(cfg)

# New API available
from src.core.di import get_container
builder = container.resolve(ModelBuilder)
model = builder.build(cfg)
```

## Future Enhancements

### 1. Configuration-Based DI
Move service registration to configuration files:
```yaml
services:
  model_builder: RegistryModelBuilder
  trainer: StandardTrainer
  exporter: StandardExporter
```

### 2. Lifecycle Management
Add lifecycle hooks for components:
```python
class Component(ABC):
    def initialize(self): pass
    def cleanup(self): pass
```

### 3. Scoped Dependencies
Support request-scoped dependencies for multi-tenant scenarios

### 4. Auto-wiring
Automatically resolve constructor dependencies:
```python
class MyTrainer:
    def __init__(self, model_builder: ModelBuilder, logger: Logger):
        # Auto-injected
```

## Conclusion

This architecture refactoring brings the codebase in line with modern software engineering best practices:

✅ **SOLID Principles**: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
✅ **Design Patterns**: Registry, Strategy, Factory, Template Method, Dependency Injection
✅ **Testability**: Easy mocking, clear interfaces, isolated components
✅ **Maintainability**: DRY, clear separation of concerns, extensible design
✅ **Flexibility**: Easy to add new models, tasks, and export formats

The refactoring maintains 100% backward compatibility while providing a modern, extensible foundation for future development.
