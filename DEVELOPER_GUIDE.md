# PyModConn Developer Guide: Adding New Layer Types

This guide provides a comprehensive checklist and best practices for adding new layer types to PyModConn while minimizing the required changes across the codebase.

## 🎯 Overview

When adding a new layer type to PyModConn, you need to update several components to maintain consistency and ensure proper testing. This guide breaks down the process into manageable steps and provides templates to minimize boilerplate code.

## 📋 Quick Checklist

### ✅ Core Implementation
- [ ] Create the new layer class in `pymodconn/utils_layers.py`
- [ ] Add layer to `__all__` exports in `pymodconn/__init__.py`
- [ ] Update base classes if needed

### ✅ Configuration Management
- [ ] Add layer configuration schema to `ConfigManager`
- [ ] Update configuration validation rules
- [ ] Add configuration templates/examples

### ✅ Testing Infrastructure
- [ ] Create unit tests for the new layer
- [ ] Add integration tests with existing components
- [ ] Update test fixtures and configurations
- [ ] Add performance/edge case tests

### ✅ Documentation & Examples
- [ ] Update API documentation
- [ ] Add usage examples
- [ ] Update configuration file templates

---

## 🔧 Step-by-Step Implementation Guide

### Step 1: Core Layer Implementation

#### 1.1 Create the Layer Class

Add your new layer to `pymodconn/utils_layers.py`:

```python
class YourNewLayer(tf.keras.layers.Layer):
    """
    Your new layer implementation.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        use_time_distributed: Whether to use TimeDistributed wrapper
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        param1: int,
        param2: float = 0.1,
        use_time_distributed: bool = False,
        **kwargs
    ) -> None:
        """Initialize YourNewLayer with specified parameters."""
        super().__init__(**kwargs)
        
        # Validate parameters
        if param1 <= 0:
            raise ValueError("param1 must be positive")
        if not (0.0 <= param2 <= 1.0):
            raise ValueError("param2 must be between 0 and 1")
            
        self.param1 = param1
        self.param2 = param2
        self.use_time_distributed = use_time_distributed
        
        # Initialize sub-layers
        self._build_layers()
    
    def _build_layers(self) -> None:
        """Build internal layers."""
        # Your layer implementation here
        pass
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the layer."""
        # Your forward pass implementation
        return inputs
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "param1": self.param1,
            "param2": self.param2,
            "use_time_distributed": self.use_time_distributed,
        })
        return config
```

#### 1.2 Update Exports

Add to `pymodconn/__init__.py`:

```python
from pymodconn.utils_layers import (
    # ... existing imports ...
    YourNewLayer,
)

__all__ = [
    # ... existing exports ...
    "YourNewLayer",
]
```

### Step 2: Configuration Management

#### 2.1 Update ConfigManager Validation

Add validation rules to `pymodconn/config_manager.py`:

```python
class ConfigManager:
    def _validate_your_new_layer_config(self, config: Dict[str, Any], component_path: str) -> None:
        """Validate YourNewLayer configuration."""
        if "your_new_layer" not in config:
            return  # Layer is optional
            
        layer_config = config["your_new_layer"]
        
        # Validate required parameters
        required_keys = ["param1", "param2"]
        ConfigValidator.validate_required_keys(
            layer_config, required_keys, f"{component_path}.your_new_layer"
        )
        
        # Validate parameter types and ranges
        ConfigValidator.validate_positive_int(
            layer_config["param1"], "param1", f"{component_path}.your_new_layer"
        )
        ConfigValidator.validate_float_range(
            layer_config["param2"], "param2", f"{component_path}.your_new_layer", 0.0, 1.0
        )
    
    def _validate_encoder_config(self) -> None:
        """Validate encoder configuration."""
        encoder_config = self.config["encoder"]
        
        # ... existing validation ...
        
        # Add validation for your new layer
        self._validate_your_new_layer_config(encoder_config, "encoder")
    
    def _validate_decoder_config(self) -> None:
        """Validate decoder configuration."""
        decoder_config = self.config["decoder"]
        
        # ... existing validation ...
        
        # Add validation for your new layer
        self._validate_your_new_layer_config(decoder_config, "decoder")
```

#### 2.2 Update Configuration Templates

Add to configuration YAML files (e.g., `pymodconn/configs/default_config.yaml`):

```yaml
encoder:
  # ... existing configuration ...
  your_new_layer:
    IF_YOUR_NEW_LAYER: true
    param1: 64
    param2: 0.1
    use_time_distributed: false

decoder:
  # ... existing configuration ...
  your_new_layer:
    IF_YOUR_NEW_LAYER: true
    param1: 32
    param2: 0.2
    use_time_distributed: true
```

### Step 3: Testing Infrastructure

#### 3.1 Create Unit Tests

Add to `tests/test_utils_layers.py`:

```python
class TestYourNewLayer:
    """Test YourNewLayer functionality."""
    
    def test_your_new_layer_init(self):
        """Test YourNewLayer initialization."""
        layer = YourNewLayer(param1=64, param2=0.1)
        
        assert layer.param1 == 64
        assert layer.param2 == 0.1
        assert layer.use_time_distributed == False
    
    def test_your_new_layer_init_validation(self):
        """Test YourNewLayer parameter validation."""
        # Test invalid param1
        with pytest.raises(ValueError, match="param1 must be positive"):
            YourNewLayer(param1=-1, param2=0.1)
        
        # Test invalid param2
        with pytest.raises(ValueError, match="param2 must be between 0 and 1"):
            YourNewLayer(param1=64, param2=1.5)
    
    def test_your_new_layer_call(self):
        """Test YourNewLayer forward pass."""
        layer = YourNewLayer(param1=32, param2=0.1)
        
        x = tf.random.normal([4, 10, 16])
        output = layer(x, training=False)
        
        # Verify output shape and properties
        assert output.shape == x.shape  # Adjust based on your layer
        assert tf.reduce_all(tf.math.is_finite(output))
    
    def test_your_new_layer_with_time_distributed(self):
        """Test YourNewLayer with time distribution."""
        layer = YourNewLayer(param1=32, param2=0.1, use_time_distributed=True)
        
        x = tf.random.normal([4, 8, 10, 16])  # 4D tensor for time distributed
        output = layer(x, training=False)
        
        assert output.shape == x.shape  # Adjust based on your layer
    
    def test_your_new_layer_get_config(self):
        """Test YourNewLayer configuration serialization."""
        layer = YourNewLayer(param1=64, param2=0.2, use_time_distributed=True)
        
        config = layer.get_config()
        
        assert config['param1'] == 64
        assert config['param2'] == 0.2
        assert config['use_time_distributed'] == True
    
    @pytest.mark.parametrize("param1,param2", [
        (16, 0.0),
        (32, 0.5),
        (64, 1.0),
        (128, 0.25),
    ])
    def test_your_new_layer_parametrized(self, param1, param2):
        """Test YourNewLayer with various parameter combinations."""
        layer = YourNewLayer(param1=param1, param2=param2)
        
        x = tf.random.normal([2, 8])
        output = layer(x, training=False)
        
        assert tf.reduce_all(tf.math.is_finite(output))
```

#### 3.2 Update Test Fixtures

Add to `tests/conftest.py`:

```python
@pytest.fixture
def sample_config_with_your_new_layer(sample_config) -> Dict[str, Any]:
    """Provide configuration with YourNewLayer enabled."""
    config = sample_config.copy()
    config["encoder"]["your_new_layer"] = {
        "IF_YOUR_NEW_LAYER": True,
        "param1": 64,
        "param2": 0.1,
        "use_time_distributed": False
    }
    config["decoder"]["your_new_layer"] = {
        "IF_YOUR_NEW_LAYER": True,
        "param1": 32,
        "param2": 0.2,
        "use_time_distributed": True
    }
    return config
```

#### 3.3 Add Integration Tests

Add to `tests/test_integration.py`:

```python
@pytest.mark.integration
class TestYourNewLayerIntegration:
    """Integration tests for YourNewLayer."""
    
    def test_model_with_your_new_layer(self, sample_config_with_your_new_layer, temp_model_dir):
        """Test complete model workflow with YourNewLayer."""
        config = sample_config_with_your_new_layer.copy()
        config["save_models_dir"] = temp_model_dir
        
        current_dt = "your_new_layer_test"
        
        # Build model
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Verify model was built successfully
        assert model_gen.model is not None
        
        # Test prediction
        batch_size = 4
        x_past = np.random.randn(batch_size, config["n_past"], config["known_past_features"])
        x_future = np.random.randn(batch_size, config["n_future"], config["known_future_features"])
        
        predictions = model_gen.model.predict([x_past, x_future], verbose=0)
        
        expected_shape = (batch_size, config["n_future"], config["unknown_future_features"])
        assert predictions.shape == expected_shape
        assert np.all(np.isfinite(predictions))
```

#### 3.4 Update Configuration Tests

Add to `tests/test_config_manager.py`:

```python
class TestYourNewLayerConfigValidation:
    """Test YourNewLayer configuration validation."""
    
    def test_validate_your_new_layer_config_success(self, sample_config, temp_model_dir):
        """Test successful YourNewLayer configuration validation."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        # Add valid YourNewLayer configuration
        config = sample_config.copy()
        config["encoder"]["your_new_layer"] = {
            "IF_YOUR_NEW_LAYER": True,
            "param1": 64,
            "param2": 0.1
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        config_manager = ConfigManager(config_path)
        loaded_config = config_manager.load_config()  # Should not raise
        
        assert loaded_config["encoder"]["your_new_layer"]["param1"] == 64
    
    def test_validate_your_new_layer_config_invalid_param1(self, sample_config, temp_model_dir):
        """Test YourNewLayer configuration validation with invalid param1."""
        config_path = Path(temp_model_dir) / "invalid_config.yaml"
        
        config = sample_config.copy()
        config["encoder"]["your_new_layer"] = {
            "IF_YOUR_NEW_LAYER": True,
            "param1": -5,  # Invalid
            "param2": 0.1
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="encoder.your_new_layer.param1 must be a positive integer"):
            config_manager.load_config()
```

---

## 🚀 Automation & Best Practices

### Template Generator Script

Create `scripts/generate_layer_template.py`:

```python
#!/usr/bin/env python3
"""
Script to generate boilerplate code for new layer types.
Usage: python scripts/generate_layer_template.py YourNewLayer
"""

import sys
from pathlib import Path

def generate_layer_template(layer_name: str):
    """Generate boilerplate code for a new layer."""
    
    # Generate layer class template
    layer_template = f'''
class {layer_name}(tf.keras.layers.Layer):
    """
    {layer_name} implementation.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        use_time_distributed: Whether to use TimeDistributed wrapper
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        param1: int,
        param2: float = 0.1,
        use_time_distributed: bool = False,
        **kwargs
    ) -> None:
        """Initialize {layer_name} with specified parameters."""
        super().__init__(**kwargs)
        
        # TODO: Add parameter validation
        self.param1 = param1
        self.param2 = param2
        self.use_time_distributed = use_time_distributed
        
        # TODO: Initialize sub-layers
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the layer."""
        # TODO: Implement forward pass
        return inputs
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({{
            "param1": self.param1,
            "param2": self.param2,
            "use_time_distributed": self.use_time_distributed,
        }})
        return config
'''
    
    # Generate test template
    test_template = f'''
class Test{layer_name}:
    """Test {layer_name} functionality."""
    
    def test_{layer_name.lower()}_init(self):
        """Test {layer_name} initialization."""
        layer = {layer_name}(param1=64, param2=0.1)
        
        assert layer.param1 == 64
        assert layer.param2 == 0.1
        assert layer.use_time_distributed == False
    
    def test_{layer_name.lower()}_call(self):
        """Test {layer_name} forward pass."""
        layer = {layer_name}(param1=32, param2=0.1)
        
        x = tf.random.normal([4, 10, 16])
        output = layer(x, training=False)
        
        # TODO: Verify output shape and properties
        assert tf.reduce_all(tf.math.is_finite(output))
    
    def test_{layer_name.lower()}_get_config(self):
        """Test {layer_name} configuration serialization."""
        layer = {layer_name}(param1=64, param2=0.2, use_time_distributed=True)
        
        config = layer.get_config()
        
        assert config['param1'] == 64
        assert config['param2'] == 0.2
        assert config['use_time_distributed'] == True
'''
    
    print(f"Generated templates for {layer_name}")
    print("=" * 50)
    print("Layer implementation (add to utils_layers.py):")
    print(layer_template)
    print("=" * 50)
    print("Test implementation (add to test_utils_layers.py):")
    print(test_template)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_layer_template.py LayerName")
        sys.exit(1)
    
    layer_name = sys.argv[1]
    generate_layer_template(layer_name)
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: check-new-layer-tests
        name: Check new layer tests
        entry: python scripts/check_layer_coverage.py
        language: python
        files: 'pymodconn/utils_layers\.py'
        
      - id: validate-config-schema
        name: Validate configuration schema
        entry: python scripts/validate_config_schema.py
        language: python
        files: 'pymodconn/config_manager\.py'
```

### Documentation Generator

Create `scripts/generate_layer_docs.py`:

```python
#!/usr/bin/env python3
"""Generate documentation for all layers."""

import inspect
from pathlib import Path
import pymodconn.utils_layers as layers

def generate_layer_docs():
    """Generate markdown documentation for all layers."""
    
    doc_content = "# PyModConn Layers Reference\n\n"
    
    # Get all layer classes
    layer_classes = [
        (name, cls) for name, cls in inspect.getmembers(layers, inspect.isclass)
        if name.endswith('Layer') or name in ['AddNorm', 'STATES_MANIPULATION_BLOCK', 'MERGE_LIST']
    ]
    
    for name, cls in sorted(layer_classes):
        doc_content += f"## {name}\n\n"
        
        if cls.__doc__:
            doc_content += f"{cls.__doc__}\n\n"
        
        # Get constructor parameters
        try:
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters.values() if p.name != 'self']
            
            if params:
                doc_content += "### Parameters\n\n"
                for param in params:
                    doc_content += f"- **{param.name}**"
                    if param.annotation != inspect.Parameter.empty:
                        doc_content += f" ({param.annotation})"
                    if param.default != inspect.Parameter.empty:
                        doc_content += f" = {param.default}"
                    doc_content += "\n"
                doc_content += "\n"
        except Exception:
            pass
        
        doc_content += "---\n\n"
    
    # Write documentation
    docs_path = Path("docs/layers_reference.md")
    docs_path.parent.mkdir(exist_ok=True)
    
    with open(docs_path, 'w') as f:
        f.write(doc_content)
    
    print(f"Generated layer documentation: {docs_path}")

if __name__ == "__main__":
    generate_layer_docs()
```

---

## 📝 Maintenance Checklist

### When Adding a New Layer Type:

1. **Core Implementation** (Required)
   - [ ] Implement layer class with proper validation
   - [ ] Add comprehensive docstrings
   - [ ] Include type hints for all parameters
   - [ ] Implement `get_config()` for serialization

2. **Configuration Management** (Required)
   - [ ] Add validation rules to `ConfigManager`
   - [ ] Update configuration templates
   - [ ] Add parameter validation with clear error messages

3. **Testing** (Required)
   - [ ] Create unit tests with edge cases
   - [ ] Add integration tests with existing components
   - [ ] Update test fixtures
   - [ ] Add parametrized tests for various configurations

4. **Documentation** (Recommended)
   - [ ] Update API documentation
   - [ ] Add usage examples
   - [ ] Update developer guide

5. **Automation** (Optional but Recommended)
   - [ ] Run template generator for boilerplate
   - [ ] Use pre-commit hooks for validation
   - [ ] Generate documentation automatically

### Maintenance Scripts

Create `scripts/check_layer_coverage.py`:

```python
#!/usr/bin/env python3
"""Check that all layers have corresponding tests."""

import ast
import sys
from pathlib import Path

def check_layer_coverage():
    """Check that all layers have corresponding tests."""
    
    # Parse utils_layers.py to find all layer classes
    utils_layers_path = Path("pymodconn/utils_layers.py")
    with open(utils_layers_path, 'r') as f:
        tree = ast.parse(f.read())
    
    layer_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if (node.name.endswith('Layer') or 
                node.name in ['AddNorm', 'STATES_MANIPULATION_BLOCK', 'MERGE_LIST']):
                layer_classes.append(node.name)
    
    # Parse test file to find test classes
    test_file_path = Path("tests/test_utils_layers.py")
    with open(test_file_path, 'r') as f:
        test_tree = ast.parse(f.read())
    
    test_classes = []
    for node in ast.walk(test_tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            test_classes.append(node.name)
    
    # Check coverage
    missing_tests = []
    for layer_class in layer_classes:
        expected_test_class = f"Test{layer_class}"
        if expected_test_class not in test_classes:
            missing_tests.append(layer_class)
    
    if missing_tests:
        print("❌ Missing tests for the following layers:")
        for layer in missing_tests:
            print(f"  - {layer}")
        sys.exit(1)
    else:
        print("✅ All layers have corresponding tests")

if __name__ == "__main__":
    check_layer_coverage()
```

---

## 🎯 Summary

This guide provides a systematic approach to adding new layer types to PyModConn with minimal friction. By following these templates and checklists, developers can:

1. **Reduce boilerplate** with automated template generation
2. **Ensure consistency** with standardized patterns
3. **Maintain quality** with comprehensive testing requirements
4. **Automate validation** with pre-commit hooks and scripts
5. **Keep documentation current** with automated generation

The key is to establish these patterns early and use automation to reduce the manual overhead of maintaining consistency across the growing codebase.
