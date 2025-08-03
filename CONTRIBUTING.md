# Contributing to PyModConn

Thank you for your interest in contributing to PyModConn! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Setting up the development environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/pymodconn.git
   cd pymodconn
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the package in development mode with all dependencies:
   ```bash
   pip install -e ".[dev,test,docs]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning

These tools are automatically run via pre-commit hooks, but you can also run them manually:

```bash
# Format code
black pymodconn tests

# Sort imports
isort pymodconn tests

# Lint code
flake8 pymodconn

# Type check
mypy pymodconn

# Security scan
bandit -r pymodconn
```

### Testing

We use pytest for testing. Run tests with:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pymodconn --cov-report=html

# Run specific test file
pytest tests/test_basic.py
```

### Type Hints

All new code should include comprehensive type hints. We use:

- Standard library `typing` module for Python < 3.9
- Built-in generics for Python >= 3.9
- `from __future__ import annotations` for forward references

Example:
```python
from __future__ import annotations

from typing import Dict, List, Optional, Union
import tensorflow as tf

def process_config(
    config: Dict[str, Union[str, int, float]], 
    model_type: str
) -> Optional[tf.keras.Model]:
    """Process configuration and return model."""
    ...
```

### Documentation

- All public functions and classes must have docstrings
- Use Google-style docstrings
- Include type information in docstrings when helpful
- Update README.md for significant changes

Example docstring:
```python
def build_model(self, config: Dict[str, Any]) -> tf.keras.Model:
    """Build a neural network model based on configuration.
    
    Args:
        config: Configuration dictionary containing model parameters.
            Must include 'n_past', 'n_future', and layer specifications.
    
    Returns:
        Compiled Keras model ready for training.
    
    Raises:
        ValueError: If required configuration keys are missing.
        TypeError: If configuration values have incorrect types.
    
    Example:
        >>> model_gen = Model_Gen(config, datetime.now())
        >>> model = model_gen.build_model()
        >>> model.summary()
    """
```

## Pull Request Process

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, ensuring:
   - Code follows style guidelines
   - Tests pass
   - New functionality is tested
   - Documentation is updated

3. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request on GitHub

### Commit Message Format

We follow conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Code Review Guidelines

### For Contributors

- Keep PRs focused and small
- Write clear commit messages
- Include tests for new functionality
- Update documentation as needed
- Respond to review feedback promptly

### For Reviewers

- Be constructive and respectful
- Focus on code quality, not personal preferences
- Suggest improvements with examples
- Approve when ready, request changes when needed

## Issue Reporting

When reporting issues:

1. Use the issue templates when available
2. Include Python version and OS
3. Provide minimal reproducible example
4. Include relevant error messages and stack traces
5. Describe expected vs actual behavior

## Development Guidelines

### Architecture Principles

- **Modularity**: Keep components loosely coupled
- **Extensibility**: Design for easy extension
- **Type Safety**: Use type hints throughout
- **Documentation**: Code should be self-documenting
- **Testing**: Maintain high test coverage

### Performance Considerations

- Profile before optimizing
- Use appropriate data structures
- Consider memory usage for large models
- Leverage TensorFlow/Keras optimizations

### Compatibility

- Support Python 3.8+
- Use modern Python features appropriately
- Maintain backward compatibility when possible
- Document breaking changes clearly

## Getting Help

- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Join our community channels (if available)
- Contact maintainers for urgent issues

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes for major features

Thank you for contributing to PyModConn!
