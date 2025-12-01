# Contributing to ear-vision-ml

Welcome to the team! We follow a strict but efficient workflow designed to maintain production quality at all times.

## 🚀 Quick Start for New Engineers

1.  **Read the Rules**: Start by reading [docs/repo_rules.md](docs/repo_rules.md). These are non-negotiable.
2.  **Setup Environment**:
    ```bash
    conda env create -f config/env/conda-tf217.yml
    conda activate ear-vision-ml
    pip install -e .
    ```
3.  **Run Tests**: Ensure everything works locally.
    ```bash
    pytest -n auto
    ```

## 🛠 Development Workflow

### 1. Create a Branch
Always work on a feature branch.
```bash
git checkout -b feature/my-new-feature
```

### 2. Test-Driven Development (TDD)
We strictly follow TDD.
1.  **Write a failing test** in `tests/`.
2.  **Run the test** to confirm it fails.
3.  **Implement the feature** to make the test pass.
4.  **Refactor** if necessary.

### 3. Documentation-Driven Development
Every meaningful change requires documentation.
1.  **Devlogs**: Create a new entry in `docs/devlog/` for every feature or significant change. Use `docs/devlog/0000-template.md`.
2.  **ADRs**: For architectural decisions, create an entry in `docs/adr/`. Use `docs/adr/0000-template.md`.
3.  **Docstrings**: All public functions and classes must have Google-style docstrings.

### 4. Code Quality
Before committing, ensure your code meets our quality standards:
```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type Checking
mypy src/core/
```

### 5. Submit a Pull Request
-   Ensure all tests pass (`pytest`).
-   Include a link to your devlog entry.
-   Describe the changes and the "why".

## 🏗 Architecture Guidelines

-   **Dependency Injection**: Use `src/core/di.py` for service management.
-   **Registry Pattern**: Register new models and pipelines in their respective registries.
-   **Config First**: Use Hydra configs (`configs/`) for all parameters. Avoid hardcoding.
-   **Contracts**: Adhere to `src/core/contracts/`. Changes here require an ADR.

## 📚 Useful Resources

-   [Repository Rules](docs/repo_rules.md)
-   [Architecture Overview](ARCHITECTURE_REFACTORING.md)
-   [Developer Cheat Sheet (CLAUDE.md)](CLAUDE.md) - Contains useful commands.

## ✅ Pull Request Checklist

Before submitting your PR, ensure:

- [ ] All tests pass locally (`pytest -v`)
- [ ] Code is linted (`ruff check .`)
- [ ] Code is formatted (`ruff format .`)
- [ ] Type checking passes (`mypy src/core/`)
- [ ] New features have tests (maintain 100% pass rate)
- [ ] Documentation updated (devlog entry required)
- [ ] No hardcoded paths or magic numbers
- [ ] Config changes documented
- [ ] Breaking changes noted in ADR

## 🧪 Testing Standards

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end workflows
- **No Network Calls**: All tests must run offline
- **Fixtures**: Use `scripts/generate_fixtures.py` for test data
- **Coverage**: Aim for comprehensive behavioral coverage, not just line coverage
