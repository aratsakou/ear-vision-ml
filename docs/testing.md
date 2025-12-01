# Testing Guide

We use `pytest` for testing. The repository has a comprehensive test suite covering unit and integration tests.

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_di.py

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Test Structure

- `tests/unit/`: Unit tests for individual components. Fast and isolated.
- `tests/integration/`: Integration tests for workflows (training, export, etc.). Slower.

## Writing Tests

- Place new tests in `tests/unit/` or `tests/integration/`.
- Follow the naming convention `test_*.py`.
- Use `pytest` fixtures for setup/teardown.
- Ensure tests are deterministic and offline (mock external services).
