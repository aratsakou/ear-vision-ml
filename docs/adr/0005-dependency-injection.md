# ADR 0005: Dependency Injection and Component Architecture

**Status:** ✅ Accepted  
**Date:** 2025-11-30  
**Deciders:** System Architecture Team  
**Related:** ADR-0001 (Repo Structure), ADR-0002 (Config Strategy)

## Context

The codebase had grown organically with functional programming patterns, leading to:

1. **Tight Coupling**: Components directly instantiated their dependencies
2. **Testing Challenges**: Difficult to mock dependencies for unit testing
3. **Code Duplication**: Similar logic repeated across task trainers
4. **Limited Extensibility**: Adding new models/tasks required modifying existing code
5. **Unclear Contracts**: No explicit interfaces defining component responsibilities

### Example Problems

**Before:**
```python
def run_classification(cfg):
    model = build_model(cfg)  # Direct function call
    ds_train = load_dataset_from_manifest_dir(...)  # Direct instantiation
    model.compile(optimizer=make_optimizer(cfg), loss=classification_loss(), ...)
    # Training logic duplicated in each task trainer
```

This violated several SOLID principles and made testing difficult.

## Decision

Implement a comprehensive architecture refactoring based on:

1. **Dependency Injection Container**: Lightweight DI for managing component lifecycle
2. **Interface-Based Design**: Define clear contracts for all major components
3. **Registry Pattern**: For extensible model factory
4. **Strategy Pattern**: For task-specific preprocessing
5. **Unified Components**: Single implementations that handle multiple tasks

### Architecture

```
┌─────────────────────────────────────────┐
│         Dependency Injection            │
│              Container                  │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ModelBuilder  Trainer    Exporter
   (Interface)  (Interface) (Interface)
        │           │           │
        ▼           ▼           ▼
   Registry    Standard    Standard
   Builder     Trainer     Exporter
```

### Key Components

#### 1. Interfaces (`src/core/interfaces.py`)
```python
class ModelBuilder(ABC):
    def build(self, cfg) -> tf.keras.Model
    
class DataLoader(ABC):
    def load_train(self, cfg) -> tf.data.Dataset
    def load_val(self, cfg) -> tf.data.Dataset
    
class Trainer(ABC):
    def train(self, model, train_ds, val_ds, cfg) -> Any
    
class Exporter(ABC):
    def export(self, model, cfg, artifacts_dir) -> Dict
```

#### 2. DI Container (`src/core/di.py`)
```python
container = Container()
container.register_singleton(ModelBuilder, registry_builder)
container.register_singleton(Trainer, StandardTrainer())
builder = container.resolve(ModelBuilder)
```

#### 3. Registry Pattern (Model Factory)
```python
@register_model("cls_mobilenetv3")
def build_cls_mobilenetv3(cfg):
    # implementation
```

#### 4. Strategy Pattern (Preprocessors)
```python
class ClassificationPreprocessor(Preprocessor):
    def preprocess(self, features, cfg):
        # task-specific logic
```

## Consequences

### Positive

1. **Improved Testability**
   - Easy to mock dependencies
   - Isolated unit testing
   - Clear test boundaries

2. **Better Maintainability**
   - DRY: Training logic in one place
   - Clear separation of concerns
   - Self-documenting code

3. **Enhanced Extensibility**
   - Add models without modifying existing code (Open/Closed)
   - Easy to add new tasks
   - Pluggable components

4. **SOLID Compliance**
   - Single Responsibility: Each class has one job
   - Open/Closed: Extend without modification
   - Liskov Substitution: Implementations are interchangeable
   - Interface Segregation: Focused interfaces
   - Dependency Inversion: Depend on abstractions

5. **Consistent Patterns**
   - All tasks follow same structure
   - Predictable code organization
   - Easier onboarding for new developers

### Negative

1. **Learning Curve**
   - Developers need to understand DI concepts
   - More abstraction layers to navigate

2. **Initial Complexity**
   - More files and classes
   - Indirection through interfaces

3. **Migration Effort**
   - Existing code needed refactoring
   - Documentation updates required

### Mitigations

1. **Backward Compatibility**
   - All old functional APIs still work
   - Gradual migration possible

2. **Comprehensive Documentation**
   - `ARCHITECTURE_REFACTORING.md` guide
   - Code examples in devlog
   - Clear migration paths

3. **Simple DI Container**
   - Lightweight implementation
   - No external dependencies
   - Easy to understand

## Alternatives Considered

### 1. Keep Functional Approach
**Rejected**: Would not solve testability and extensibility issues

### 2. Use External DI Framework (e.g., dependency-injector, injector)
**Rejected**: 
- Adds external dependency
- Overkill for our needs
- Harder to customize

### 3. Partial Refactoring (Only Model Factory)
**Rejected**: Would not provide full benefits, inconsistent patterns

### 4. Service Locator Pattern
**Rejected**: 
- Hides dependencies
- Makes testing harder
- Considered anti-pattern

## Implementation Notes

### Backward Compatibility Strategy

All existing APIs maintained:
```python
# Old way still works
from src.core.models.factories.model_factory import build_model
model = build_model(cfg)

# New way available
builder = container.resolve(ModelBuilder)
model = builder.build(cfg)
```

### Migration Path

1. ✅ Create interfaces and DI container
2. ✅ Refactor model factory to registry pattern
3. ✅ Create StandardTrainer
4. ✅ Refactor data loaders with strategies
5. ✅ Update all task trainers
6. ✅ Add comprehensive documentation
7. ✅ Create integration tests

### Performance Considerations

- No runtime overhead (DI resolution at startup)
- Same execution paths for hot code
- Memory efficient (singleton pattern)

## Validation

### Testing
- ✅ Integration test validates full DI pipeline
- ✅ All existing tests still pass
- ✅ Backward compatibility verified

### Code Quality
- ✅ Follows SOLID principles
- ✅ Clear interfaces and contracts
- ✅ Self-documenting code

### Documentation
- ✅ Architecture guide created
- ✅ Devlog entry added
- ✅ Migration examples provided

## Future Considerations

1. **Configuration-Based DI**: Move service registration to YAML
2. **Lifecycle Hooks**: Add initialize/cleanup methods
3. **Scoped Dependencies**: Support request-scoped services
4. **Auto-wiring**: Automatically resolve constructor dependencies

## References

- Martin Fowler: "Inversion of Control Containers and the Dependency Injection pattern"
- Robert C. Martin: "Clean Architecture" (SOLID principles)
- Gang of Four: "Design Patterns" (Registry, Strategy, Factory patterns)
- ADR-0001: Repository Structure
- ADR-0002: Configuration Strategy

## Related Files

**New:**
- `src/core/di.py`
- `src/core/interfaces.py`
- `src/core/training/standard_trainer.py`
- `ARCHITECTURE_REFACTORING.md`
- `docs/devlog/0011-architecture-refactoring-di.md`

**Modified:**
- `src/core/models/factories/model_factory.py`
- `src/core/data/dataset_loader.py`
- `src/core/export/exporter.py`
- `src/tasks/*/trainer.py`
