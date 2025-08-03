#!/usr/bin/env python3
"""
Script to generate boilerplate code for new layer types.
Usage: python scripts/generate_layer_template.py YourNewLayer
"""

import sys
from pathlib import Path
from typing import Dict, Any


def generate_layer_template(layer_name: str) -> Dict[str, str]:
    """Generate boilerplate code for a new layer."""
    
    # Generate layer class template
    layer_template = f'''class {layer_name}(tf.keras.layers.Layer):
    """
    {layer_name} implementation.
    
    This layer provides [DESCRIBE FUNCTIONALITY HERE].
    
    Args:
        param1: Description of parameter 1 (e.g., number of units)
        param2: Description of parameter 2 (e.g., dropout rate)
        use_time_distributed: Whether to use TimeDistributed wrapper
        **kwargs: Additional keyword arguments for parent Layer class
        
    Example:
        >>> layer = {layer_name}(param1=64, param2=0.1)
        >>> x = tf.random.normal([4, 10, 16])
        >>> output = layer(x)
        >>> print(output.shape)  # (4, 10, 64) or similar
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
        
        # Parameter validation
        if param1 <= 0:
            raise ValueError(f"param1 must be positive, got {{param1}}")
        if not (0.0 <= param2 <= 1.0):
            raise ValueError(f"param2 must be between 0 and 1, got {{param2}}")
            
        self.param1 = param1
        self.param2 = param2
        self.use_time_distributed = use_time_distributed
        
        # Initialize sub-layers
        self._build_layers()
    
    def _build_layers(self) -> None:
        """Build internal layers."""
        # TODO: Initialize your sub-layers here
        # Example:
        # self.dense_layer = tf.keras.layers.Dense(self.param1)
        # self.dropout_layer = tf.keras.layers.Dropout(self.param2)
        # 
        # if self.use_time_distributed:
        #     self.dense_layer = tf.keras.layers.TimeDistributed(self.dense_layer)
        pass
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor with shape [..., input_dim]
            training: Whether in training mode
            
        Returns:
            Output tensor with shape [..., param1]
        """
        # TODO: Implement your forward pass here
        # Example:
        # x = self.dense_layer(inputs)
        # if training and self.param2 > 0:
        #     x = self.dropout_layer(x, training=training)
        # return x
        
        return inputs  # Placeholder - replace with actual implementation
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({{
            "param1": self.param1,
            "param2": self.param2,
            "use_time_distributed": self.use_time_distributed,
        }})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "{layer_name}":
        """Create layer from configuration."""
        return cls(**config)
'''
    
    # Generate test template
    test_template = f'''class Test{layer_name}:
    """Test {layer_name} functionality."""
    
    def test_{layer_name.lower()}_init(self):
        """Test {layer_name} initialization."""
        layer = {layer_name}(param1=64, param2=0.1)
        
        assert layer.param1 == 64
        assert layer.param2 == 0.1
        assert layer.use_time_distributed == False
    
    def test_{layer_name.lower()}_init_validation(self):
        """Test {layer_name} parameter validation."""
        # Test invalid param1
        with pytest.raises(ValueError, match="param1 must be positive"):
            {layer_name}(param1=-1, param2=0.1)
        
        # Test invalid param2
        with pytest.raises(ValueError, match="param2 must be between 0 and 1"):
            {layer_name}(param1=64, param2=1.5)
    
    def test_{layer_name.lower()}_call(self):
        """Test {layer_name} forward pass."""
        layer = {layer_name}(param1=32, param2=0.1)
        
        x = tf.random.normal([4, 10, 16])
        output = layer(x, training=False)
        
        # TODO: Verify output shape and properties based on your implementation
        # Example assertions:
        # assert output.shape == (4, 10, 32)  # Adjust based on your layer
        assert tf.reduce_all(tf.math.is_finite(output))
    
    def test_{layer_name.lower()}_with_time_distributed(self):
        """Test {layer_name} with time distribution."""
        layer = {layer_name}(param1=32, param2=0.1, use_time_distributed=True)
        
        x = tf.random.normal([4, 8, 10, 16])  # 4D tensor for time distributed
        output = layer(x, training=False)
        
        # TODO: Verify output shape for time distributed case
        # assert output.shape == (4, 8, 10, 32)  # Adjust based on your layer
        assert tf.reduce_all(tf.math.is_finite(output))
    
    def test_{layer_name.lower()}_training_mode(self):
        """Test {layer_name} in training vs inference mode."""
        layer = {layer_name}(param1=16, param2=0.5)  # Higher dropout for testing
        
        x = tf.random.normal([2, 8])
        
        # Test training mode
        output_train = layer(x, training=True)
        assert tf.reduce_all(tf.math.is_finite(output_train))
        
        # Test inference mode
        output_infer = layer(x, training=False)
        assert tf.reduce_all(tf.math.is_finite(output_infer))
        
        # TODO: Add specific assertions based on your layer's behavior
        # For example, if your layer has dropout, outputs might differ
    
    def test_{layer_name.lower()}_get_config(self):
        """Test {layer_name} configuration serialization."""
        layer = {layer_name}(param1=64, param2=0.2, use_time_distributed=True)
        
        config = layer.get_config()
        
        assert config['param1'] == 64
        assert config['param2'] == 0.2
        assert config['use_time_distributed'] == True
    
    def test_{layer_name.lower()}_from_config(self):
        """Test {layer_name} creation from configuration."""
        original_layer = {layer_name}(param1=32, param2=0.15, use_time_distributed=True)
        config = original_layer.get_config()
        
        # Create new layer from config
        new_layer = {layer_name}.from_config(config)
        
        assert new_layer.param1 == original_layer.param1
        assert new_layer.param2 == original_layer.param2
        assert new_layer.use_time_distributed == original_layer.use_time_distributed
    
    @pytest.mark.parametrize("param1,param2,use_time_distributed", [
        (16, 0.0, False),
        (32, 0.5, False),
        (64, 1.0, False),
        (128, 0.25, True),
        (256, 0.1, True),
    ])
    def test_{layer_name.lower()}_parametrized(self, param1, param2, use_time_distributed):
        """Test {layer_name} with various parameter combinations."""
        layer = {layer_name}(param1=param1, param2=param2, use_time_distributed=use_time_distributed)
        
        if use_time_distributed:
            x = tf.random.normal([2, 5, 8])  # 3D for time distributed
        else:
            x = tf.random.normal([2, 8])  # 2D for regular
            
        output = layer(x, training=False)
        
        assert tf.reduce_all(tf.math.is_finite(output))
        # TODO: Add more specific assertions based on your layer's behavior
    
    def test_{layer_name.lower()}_edge_cases(self):
        """Test {layer_name} edge cases."""
        layer = {layer_name}(param1=1, param2=0.0)  # Minimal valid parameters
        
        # Test with minimal input
        x = tf.random.normal([1, 1])
        output = layer(x, training=False)
        
        assert tf.reduce_all(tf.math.is_finite(output))
        
        # Test with larger input
        x_large = tf.random.normal([100, 50])
        output_large = layer(x_large, training=False)
        
        assert tf.reduce_all(tf.math.is_finite(output_large))
'''
    
    # Generate config validation template
    config_validation_template = f'''    def _validate_{layer_name.lower()}_config(self, config: Dict[str, Any], component_path: str) -> None:
        """Validate {layer_name} configuration."""
        layer_key = "{layer_name.lower()}"
        
        if layer_key not in config:
            return  # Layer is optional
            
        layer_config = config[layer_key]
        
        # Validate required parameters
        required_keys = ["IF_{layer_name.upper()}", "param1", "param2"]
        ConfigValidator.validate_required_keys(
            layer_config, required_keys, f"{{component_path}}.{{layer_key}}"
        )
        
        # Validate IF flag
        if not isinstance(layer_config["IF_{layer_name.upper()}"], bool):
            raise ValueError(f"{{component_path}}.{{layer_key}}.IF_{layer_name.upper()} must be boolean")
        
        # Only validate other parameters if layer is enabled
        if not layer_config["IF_{layer_name.upper()}"]:
            return
            
        # Validate parameter types and ranges
        ConfigValidator.validate_positive_int(
            layer_config["param1"], "param1", f"{{component_path}}.{{layer_key}}"
        )
        ConfigValidator.validate_float_range(
            layer_config["param2"], "param2", f"{{component_path}}.{{layer_key}}", 0.0, 1.0
        )
        
        # Validate optional parameters
        if "use_time_distributed" in layer_config:
            if not isinstance(layer_config["use_time_distributed"], bool):
                raise ValueError(f"{{component_path}}.{{layer_key}}.use_time_distributed must be boolean")'''
    
    # Generate config test template
    config_test_template = f'''class Test{layer_name}ConfigValidation:
    """Test {layer_name} configuration validation."""
    
    def test_validate_{layer_name.lower()}_config_success(self, sample_config, temp_model_dir):
        """Test successful {layer_name} configuration validation."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        # Add valid {layer_name} configuration
        config = sample_config.copy()
        config["encoder"]["{layer_name.lower()}"] = {{
            "IF_{layer_name.upper()}": True,
            "param1": 64,
            "param2": 0.1,
            "use_time_distributed": False
        }}
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        config_manager = ConfigManager(config_path)
        loaded_config = config_manager.load_config()  # Should not raise
        
        assert loaded_config["encoder"]["{layer_name.lower()}"]["param1"] == 64
    
    def test_validate_{layer_name.lower()}_config_disabled(self, sample_config, temp_model_dir):
        """Test {layer_name} configuration when disabled."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        config = sample_config.copy()
        config["encoder"]["{layer_name.lower()}"] = {{
            "IF_{layer_name.upper()}": False,
            # Other parameters can be invalid when disabled
            "param1": -1,
            "param2": 2.0
        }}
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        config_manager = ConfigManager(config_path)
        loaded_config = config_manager.load_config()  # Should not raise
        
        assert loaded_config["encoder"]["{layer_name.lower()}"]["IF_{layer_name.upper()}"] == False
    
    def test_validate_{layer_name.lower()}_config_invalid_param1(self, sample_config, temp_model_dir):
        """Test {layer_name} configuration validation with invalid param1."""
        config_path = Path(temp_model_dir) / "invalid_config.yaml"
        
        config = sample_config.copy()
        config["encoder"]["{layer_name.lower()}"] = {{
            "IF_{layer_name.upper()}": True,
            "param1": -5,  # Invalid
            "param2": 0.1
        }}
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="encoder.{layer_name.lower()}.param1 must be a positive integer"):
            config_manager.load_config()
    
    def test_validate_{layer_name.lower()}_config_invalid_param2(self, sample_config, temp_model_dir):
        """Test {layer_name} configuration validation with invalid param2."""
        config_path = Path(temp_model_dir) / "invalid_config.yaml"
        
        config = sample_config.copy()
        config["encoder"]["{layer_name.lower()}"] = {{
            "IF_{layer_name.upper()}": True,
            "param1": 64,
            "param2": 1.5  # Invalid
        }}
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="encoder.{layer_name.lower()}.param2 must be between 0.0 and 1.0"):
            config_manager.load_config()
    
    def test_validate_{layer_name.lower()}_config_missing_keys(self, sample_config, temp_model_dir):
        """Test {layer_name} configuration validation with missing keys."""
        config_path = Path(temp_model_dir) / "incomplete_config.yaml"
        
        config = sample_config.copy()
        config["encoder"]["{layer_name.lower()}"] = {{
            "IF_{layer_name.upper()}": True,
            # Missing param1 and param2
        }}
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="encoder.{layer_name.lower()} missing required configuration keys"):
            config_manager.load_config()'''
    
    # Generate fixture template
    fixture_template = f'''@pytest.fixture
def sample_config_with_{layer_name.lower()}(sample_config) -> Dict[str, Any]:
    """Provide configuration with {layer_name} enabled."""
    config = sample_config.copy()
    config["encoder"]["{layer_name.lower()}"] = {{
        "IF_{layer_name.upper()}": True,
        "param1": 64,
        "param2": 0.1,
        "use_time_distributed": False
    }}
    config["decoder"]["{layer_name.lower()}"] = {{
        "IF_{layer_name.upper()}": True,
        "param1": 32,
        "param2": 0.2,
        "use_time_distributed": True
    }}
    return config'''
    
    # Generate integration test template
    integration_test_template = f'''@pytest.mark.integration
class Test{layer_name}Integration:
    """Integration tests for {layer_name}."""
    
    def test_model_with_{layer_name.lower()}(self, sample_config_with_{layer_name.lower()}, temp_model_dir):
        """Test complete model workflow with {layer_name}."""
        config = sample_config_with_{layer_name.lower()}.copy()
        config["save_models_dir"] = temp_model_dir
        
        current_dt = "{layer_name.lower()}_integration_test"
        
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
    
    def test_{layer_name.lower()}_with_other_layers(self, sample_config_with_{layer_name.lower()}, temp_model_dir):
        """Test {layer_name} working with other layers."""
        config = sample_config_with_{layer_name.lower()}.copy()
        config["save_models_dir"] = temp_model_dir
        
        # Enable other layers for comprehensive testing
        config["encoder"]["RNN_block_input"]["IF_RNN"] = True
        config["encoder"]["self_MHA_block"]["IF_MHA"] = True
        
        current_dt = "{layer_name.lower()}_multi_layer_test"
        
        # Build model
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Verify model works with multiple layer types
        assert model_gen.model is not None
        
        # Test training step
        batch_size = 8
        x_past = np.random.randn(batch_size, config["n_past"], config["known_past_features"])
        x_future = np.random.randn(batch_size, config["n_future"], config["known_future_features"])
        y_future = np.random.randn(batch_size, config["n_future"], config["unknown_future_features"])
        
        # Single training step
        history = model_gen.model.fit(
            [x_past, x_future], y_future,
            batch_size=4, epochs=1, verbose=0
        )
        
        assert "loss" in history.history
        assert np.isfinite(history.history["loss"][0])'''
    
    return {
        "layer_implementation": layer_template,
        "unit_tests": test_template,
        "config_validation": config_validation_template,
        "config_tests": config_test_template,
        "fixture": fixture_template,
        "integration_tests": integration_test_template
    }


def main():
    """Main entry point for template generation."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_layer_template.py LayerName")
        print("Example: python scripts/generate_layer_template.py AttentionLayer")
        sys.exit(1)
    
    layer_name = sys.argv[1]
    
    # Validate layer name
    if not layer_name.isidentifier():
        print(f"Error: '{layer_name}' is not a valid Python identifier")
        sys.exit(1)
    
    if not layer_name[0].isupper():
        print(f"Warning: Layer name should start with uppercase letter (got '{layer_name}')")
        layer_name = layer_name.capitalize()
    
    print(f"Generating templates for {layer_name}...")
    print("=" * 80)
    
    templates = generate_layer_template(layer_name)
    
    # Display templates
    print("\\n📝 LAYER IMPLEMENTATION")
    print("Add this to pymodconn/utils_layers.py:")
    print("-" * 40)
    print(templates["layer_implementation"])
    
    print("\\n🧪 UNIT TESTS")
    print("Add this to tests/test_utils_layers.py:")
    print("-" * 40)
    print(templates["unit_tests"])
    
    print("\\n⚙️ CONFIG VALIDATION")
    print("Add this method to ConfigManager class in pymodconn/config_manager.py:")
    print("-" * 40)
    print(templates["config_validation"])
    
    print("\\n🔧 CONFIG TESTS")
    print("Add this to tests/test_config_manager.py:")
    print("-" * 40)
    print(templates["config_tests"])
    
    print("\\n🏗️ TEST FIXTURE")
    print("Add this to tests/conftest.py:")
    print("-" * 40)
    print(templates["fixture"])
    
    print("\\n🔗 INTEGRATION TESTS")
    print("Add this to tests/test_integration.py:")
    print("-" * 40)
    print(templates["integration_tests"])
    
    print("\\n" + "=" * 80)
    print("✅ Template generation complete!")
    print("\\n📋 Next steps:")
    print("1. Copy the layer implementation to pymodconn/utils_layers.py")
    print("2. Add the layer to __all__ in pymodconn/__init__.py")
    print("3. Copy the tests to their respective test files")
    print("4. Update ConfigManager._validate_encoder_config() and _validate_decoder_config()")
    print("5. Add configuration examples to YAML files")
    print("6. Run tests to verify everything works")
    print("\\n💡 Don't forget to implement the actual layer logic in _build_layers() and call()!")


if __name__ == "__main__":
    main()
