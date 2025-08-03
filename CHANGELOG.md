# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive development tooling setup
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipeline
- Type checking with mypy
- Code formatting with black and isort
- Security scanning with bandit
- Test framework with pytest
- Development documentation (CONTRIBUTING.md)

### Changed
- Updated project configuration to use modern pyproject.toml
- Upgraded Python version requirement to >=3.8
- Updated dependency versions to latest stable releases
- Improved .gitignore with modern Python patterns
- Enhanced package metadata and classifiers

### Fixed
- Version consistency across all configuration files
- Package structure and imports

## [2.0.0] - 2024-XX-XX

### Added
- Modular sequence-to-sequence neural network architecture
- Support for multiple encoder/decoder configurations
- TCN (Temporal Convolutional Network) layers
- Multi-head attention mechanisms
- RNN blocks with bidirectional support
- Configurable probabilistic and non-probabilistic forecasting
- YAML-based configuration system
- Custom Keras layers for neural network components

### Features
- **Encoder Components:**
  - TCN layers with additive normalization
  - RNN blocks (GRU, LSTM, SimpleRNN)
  - Self-attention mechanisms
  - Configurable depth and parameters

- **Decoder Components:**
  - Contextual Information Transfer (CIT) options
  - Cross-attention mechanisms
  - Multiple decoder architectures
  - State manipulation methods

- **Model Types:**
  - Probabilistic forecasting with parametric/non-parametric losses
  - Non-probabilistic deterministic forecasting
  - Quantile regression support
  - Custom loss functions

- **Configuration:**
  - Flexible YAML configuration system
  - Multiple pre-defined model architectures
  - Extensive hyperparameter customization
  - Model saving and loading capabilities

### Technical Details
- Built on TensorFlow/Keras framework
- Support for custom layer implementations
- Memory-efficient model architectures
- Comprehensive model visualization and summary tools
- Deterministic training with seed support

## [1.x.x] - Previous Versions

Previous versions of this project had different versioning and feature sets.
For historical information, please refer to the git history.

---

## Version Guidelines

### Version Numbers
- **Major (X.0.0)**: Breaking changes, major new features
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

### Change Categories
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

### Release Process
1. Update version in `pyproject.toml`, `setup.cfg`, and `pymodconn/__init__.py`
2. Update CHANGELOG.md with release notes
3. Create release tag: `git tag -a v2.0.0 -m "Release v2.0.0"`
4. Push tag: `git push origin v2.0.0`
5. GitHub Actions will automatically build and potentially publish the release
