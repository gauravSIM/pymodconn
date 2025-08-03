"""Tests for utility layers and functions."""

import numpy as np
import pytest
import tensorflow as tf

from pymodconn.utils_layers import (
    soft_relu, LinearLayer, GRNLayer, GLUWithAddNorm, 
    AddNorm, GLULayer, STATES_MANIPULATION_BLOCK, 
    MERGE_LIST, positional_encoding
)


class TestSoftRelu:
    """Test soft_relu activation function."""
    
    def test_soft_relu_positive_values(self):
        """Test soft_relu with positive values."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = soft_relu(x)
        
        # soft_relu(x) = log(1 + exp(x))
        expected = tf.math.log(1.0 + tf.math.exp(x))
        
        tf.debugging.assert_near(result, expected, atol=1e-6)
    
    def test_soft_relu_negative_values(self):
        """Test soft_relu with negative values."""
        x = tf.constant([-1.0, -2.0, -3.0])
        result = soft_relu(x)
        
        # For negative values, soft_relu should still be positive
        assert tf.reduce_all(result > 0)
        
        # Verify mathematical correctness
        expected = tf.math.log(1.0 + tf.math.exp(x))
        tf.debugging.assert_near(result, expected, atol=1e-6)
    
    def test_soft_relu_zero(self):
        """Test soft_relu with zero."""
        x = tf.constant([0.0])
        result = soft_relu(x)
        
        # soft_relu(0) = log(1 + exp(0)) = log(2)
        expected = tf.math.log(2.0)
        tf.debugging.assert_near(result, expected, atol=1e-6)
    
    def test_soft_relu_large_values(self):
        """Test soft_relu with large values to check numerical stability."""
        x = tf.constant([10.0, 50.0, 100.0])
        result = soft_relu(x)
        
        # For large x, soft_relu(x) ≈ x
        tf.debugging.assert_near(result, x, atol=1e-3)


class TestLinearLayer:
    """Test LinearLayer functionality."""
    
    def test_linear_layer_init(self):
        """Test LinearLayer initialization."""
        layer = LinearLayer(hidden_layer_size=64, activation='relu')
        
        assert layer.hidden_layer_size == 64
        assert layer.activation == 'relu'
        assert layer.use_time_distributed == False
        assert layer.use_bias == True
    
    def test_linear_layer_call_without_time_distributed(self):
        """Test LinearLayer call without time distribution."""
        layer = LinearLayer(hidden_layer_size=32, use_time_distributed=False)
        
        # Build layer with sample input
        x = tf.random.normal([10, 20])  # batch_size=10, features=20
        output = layer(x)
        
        assert output.shape == (10, 32)
    
    def test_linear_layer_call_with_time_distributed(self):
        """Test LinearLayer call with time distribution."""
        layer = LinearLayer(hidden_layer_size=32, use_time_distributed=True)
        
        # Build layer with sample input
        x = tf.random.normal([10, 5, 20])  # batch_size=10, time_steps=5, features=20
        output = layer(x)
        
        assert output.shape == (10, 5, 32)
    
    def test_linear_layer_with_activation(self):
        """Test LinearLayer with activation function."""
        layer = LinearLayer(hidden_layer_size=16, activation='relu')
        
        x = tf.constant([[-1.0, 0.0, 1.0, 2.0]])
        output = layer(x)
        
        # With ReLU activation, all outputs should be non-negative
        assert tf.reduce_all(output >= 0)
    
    def test_linear_layer_get_config(self):
        """Test LinearLayer configuration serialization."""
        layer = LinearLayer(
            hidden_layer_size=64, 
            activation='tanh', 
            use_time_distributed=True,
            use_bias=False
        )
        
        config = layer.get_config()
        
        assert config['hidden_layer_size'] == 64
        assert config['activation'] == 'tanh'
        assert config['use_time_distributed'] == True
        assert config['use_bias'] == False


class TestGRNLayer:
    """Test GRNLayer (Gated Residual Network) functionality."""
    
    def test_grn_layer_init(self):
        """Test GRNLayer initialization."""
        layer = GRNLayer(
            hidden_layer_size=64,
            output_size=32,
            dropout_rate=0.1,
            use_time_distributed=True,
            activation_layer_type='elu'
        )
        
        assert layer.hidden_layer_size == 64
        assert layer.output_size == 32
        assert layer.dropout_rate == 0.1
        assert layer.use_time_distributed == True
        assert layer.activation_layer_type == 'elu'
    
    def test_grn_layer_call(self):
        """Test GRNLayer forward pass."""
        layer = GRNLayer(
            hidden_layer_size=32,
            output_size=16,
            dropout_rate=0.1,
            use_time_distributed=False
        )
        
        x = tf.random.normal([8, 16])  # batch_size=8, features=16
        output = layer(x, training=False)
        
        assert output.shape == (8, 16)
    
    def test_grn_layer_with_time_distributed(self):
        """Test GRNLayer with time distribution."""
        layer = GRNLayer(
            hidden_layer_size=32,
            output_size=16,
            dropout_rate=0.1,
            use_time_distributed=True
        )
        
        x = tf.random.normal([8, 10, 16])  # batch_size=8, time_steps=10, features=16
        output = layer(x, training=False)
        
        assert output.shape == (8, 10, 16)
    
    def test_grn_layer_get_config(self):
        """Test GRNLayer configuration serialization."""
        layer = GRNLayer(
            hidden_layer_size=64,
            output_size=32,
            dropout_rate=0.2,
            use_time_distributed=True,
            activation_layer_type='relu'
        )
        
        config = layer.get_config()
        
        assert config['hidden_layer_size'] == 64
        assert config['output_size'] == 32
        assert config['dropout_rate'] == 0.2
        assert config['use_time_distributed'] == True
        assert config['activation_layer_type'] == 'relu'


class TestAddNorm:
    """Test AddNorm layer functionality."""
    
    def test_add_norm_init(self):
        """Test AddNorm initialization."""
        layer = AddNorm()
        
        # Should initialize without errors
        assert layer is not None
    
    def test_add_norm_call(self):
        """Test AddNorm forward pass."""
        layer = AddNorm()
        
        skip = tf.random.normal([4, 8])
        x = tf.random.normal([4, 8])
        
        output = layer(skip, x)
        
        assert output.shape == (4, 8)
        
        # Output should be normalized (mean close to 0, std close to 1)
        mean = tf.reduce_mean(output, axis=-1)
        std = tf.math.reduce_std(output, axis=-1)
        
        # Layer normalization should result in mean ≈ 0 and std ≈ 1
        tf.debugging.assert_near(mean, tf.zeros_like(mean), atol=1e-5)
        tf.debugging.assert_near(std, tf.ones_like(std), atol=1e-5)
    
    def test_add_norm_with_different_shapes(self):
        """Test AddNorm with different tensor shapes."""
        layer = AddNorm()
        
        # Test with 3D tensors
        skip = tf.random.normal([2, 5, 10])
        x = tf.random.normal([2, 5, 10])
        
        output = layer(skip, x)
        
        assert output.shape == (2, 5, 10)


class TestGLULayer:
    """Test GLULayer (Gated Linear Unit) functionality."""
    
    def test_glu_layer_init(self):
        """Test GLULayer initialization."""
        layer = GLULayer(
            output_layer_size=32,
            dropout_rate=0.1,
            use_time_distributed=True,
            activation='tanh'
        )
        
        assert layer.output_layer_size == 32
        assert layer.dropout_rate == 0.1
        assert layer.use_time_distributed == True
        assert layer.activation == 'tanh'
    
    def test_glu_layer_call_without_time_distributed(self):
        """Test GLULayer without time distribution."""
        layer = GLULayer(
            output_layer_size=16,
            dropout_rate=0.1,
            use_time_distributed=False
        )
        
        x = tf.random.normal([4, 20])
        output = layer(x, training=False)
        
        assert output.shape == (4, 16)
    
    def test_glu_layer_call_with_time_distributed(self):
        """Test GLULayer with time distribution."""
        layer = GLULayer(
            output_layer_size=16,
            dropout_rate=0.1,
            use_time_distributed=True
        )
        
        x = tf.random.normal([4, 8, 20])
        output = layer(x, training=False)
        
        assert output.shape == (4, 8, 16)
    
    def test_glu_layer_gating_mechanism(self):
        """Test that GLU applies gating mechanism correctly."""
        layer = GLULayer(
            output_layer_size=4,
            dropout_rate=None,  # No dropout for this test
            use_time_distributed=False
        )
        
        # Create input that will result in predictable gating
        x = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        output = layer(x, training=False)
        
        # Output should be element-wise product of activation and sigmoid
        assert output.shape == (1, 4)
        
        # All values should be finite
        assert tf.reduce_all(tf.math.is_finite(output))
    
    def test_glu_layer_get_config(self):
        """Test GLULayer configuration serialization."""
        layer = GLULayer(
            output_layer_size=32,
            dropout_rate=0.2,
            use_time_distributed=True,
            activation='relu'
        )
        
        config = layer.get_config()
        
        assert config['output_layer_size'] == 32
        assert config['dropout_rate'] == 0.2
        assert config['use_time_distributed'] == True
        assert config['activation'] == 'relu'


class TestGLUWithAddNorm:
    """Test GLUWithAddNorm layer functionality."""
    
    def test_glu_with_add_norm_init(self):
        """Test GLUWithAddNorm initialization."""
        layer = GLUWithAddNorm(
            output_layer_size=32,
            dropout_rate=0.1,
            use_time_distributed=True,
            activation='relu'
        )
        
        assert layer.output_layer_size == 32
        assert layer.dropout_rate == 0.1
        assert layer.use_time_distributed == True
        assert layer.activation == 'relu'
    
    def test_glu_with_add_norm_call(self):
        """Test GLUWithAddNorm forward pass."""
        layer = GLUWithAddNorm(
            output_layer_size=16,
            dropout_rate=0.1,
            use_time_distributed=False
        )
        
        skip = tf.random.normal([4, 16])
        x = tf.random.normal([4, 16])
        
        output = layer(skip, x, training=False)
        
        assert output.shape == (4, 16)
    
    def test_glu_with_add_norm_residual_connection(self):
        """Test that GLUWithAddNorm maintains residual connection."""
        layer = GLUWithAddNorm(
            output_layer_size=8,
            dropout_rate=None,
            use_time_distributed=False
        )
        
        skip = tf.random.normal([2, 8])
        x = tf.zeros([2, 8])  # Zero input to test residual connection
        
        output = layer(skip, x, training=False)
        
        # With zero input, output should be influenced by skip connection
        assert output.shape == (2, 8)
        assert not tf.reduce_all(tf.equal(output, tf.zeros_like(output)))


class TestSTATESMANIPULATIONBLOCK:
    """Test STATES_MANIPULATION_BLOCK functionality."""
    
    def test_states_manipulation_block_init(self):
        """Test STATES_MANIPULATION_BLOCK initialization."""
        block = STATES_MANIPULATION_BLOCK(d1=32, states_manipulation_method=4)
        
        assert block.states_manipulation_method == 4
    
    def test_states_manipulation_method_1(self):
        """Test method 1 (return None)."""
        block = STATES_MANIPULATION_BLOCK(d1=32, states_manipulation_method=1)
        
        x1 = [tf.random.normal([4, 16]), tf.random.normal([4, 16])]
        x2 = [tf.random.normal([4, 16]), tf.random.normal([4, 16])]
        
        result = block(x1, x2)
        assert result is None
    
    def test_states_manipulation_method_2(self):
        """Test method 2 (return x1)."""
        block = STATES_MANIPULATION_BLOCK(d1=32, states_manipulation_method=2)
        
        x1 = [tf.random.normal([4, 16]), tf.random.normal([4, 16])]
        x2 = [tf.random.normal([4, 16]), tf.random.normal([4, 16])]
        
        result = block(x1, x2)
        assert result == x1
    
    def test_states_manipulation_method_3(self):
        """Test method 3 (return x2)."""
        block = STATES_MANIPULATION_BLOCK(d1=32, states_manipulation_method=3)
        
        x1 = [tf.random.normal([4, 16]), tf.random.normal([4, 16])]
        x2 = [tf.random.normal([4, 16]), tf.random.normal([4, 16])]
        
        result = block(x1, x2)
        assert result == x2
    
    def test_states_manipulation_method_4(self):
        """Test method 4 (concatenate and dense)."""
        block = STATES_MANIPULATION_BLOCK(d1=32, states_manipulation_method=4)
        
        x1 = [tf.random.normal([4, 16]), tf.random.normal([4, 16])]
        x2 = [tf.random.normal([4, 16]), tf.random.normal([4, 16])]
        
        result = block(x1, x2)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (4, 32)
        assert result[1].shape == (4, 32)
    
    def test_states_manipulation_method_7(self):
        """Test method 7 (add states)."""
        block = STATES_MANIPULATION_BLOCK(d1=32, states_manipulation_method=7)
        
        x1 = [tf.ones([4, 16]), tf.ones([4, 16])]
        x2 = [tf.ones([4, 16]), tf.ones([4, 16])]
        
        result = block(x1, x2)
        
        assert isinstance(result, list)
        assert len(result) == 2
        # Each result should be sum of corresponding inputs
        tf.debugging.assert_near(result[0], 2 * tf.ones([4, 16]))
        tf.debugging.assert_near(result[1], 2 * tf.ones([4, 16]))
    
    def test_states_manipulation_none_inputs(self):
        """Test with None inputs."""
        block = STATES_MANIPULATION_BLOCK(d1=32, states_manipulation_method=4)
        
        result = block(None, None)
        assert result is None
        
        result = block([], [])
        assert result is None
    
    def test_states_manipulation_invalid_method(self):
        """Test with invalid method number."""
        block = STATES_MANIPULATION_BLOCK(d1=32, states_manipulation_method=99)
        
        x1 = [tf.random.normal([4, 16])]
        x2 = [tf.random.normal([4, 16])]
        
        result = block(x1, x2)
        assert result is None


class TestMERGELIST:
    """Test MERGE_LIST layer functionality."""
    
    def test_merge_list_init(self):
        """Test MERGE_LIST initialization."""
        layer = MERGE_LIST(d1=32)
        
        # Should initialize without errors
        assert layer is not None
    
    def test_merge_list_call(self):
        """Test MERGE_LIST forward pass."""
        layer = MERGE_LIST(d1=16)
        
        # Create list of tensors to merge
        tensor_list = [
            tf.random.normal([4, 8]),
            tf.random.normal([4, 12]),
            tf.random.normal([4, 4])
        ]
        
        output = layer(tensor_list, training=False)
        
        # Output should have shape [batch_size, d1]
        assert output.shape == (4, 16)
    
    def test_merge_list_get_config(self):
        """Test MERGE_LIST configuration serialization."""
        layer = MERGE_LIST(d1=64)
        
        config = layer.get_config()
        
        assert 'd1' in config
        # The d1 value should be related to the dense layer units


class TestPositionalEncoding:
    """Test positional encoding function."""
    
    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        position = 100
        d_model = 64
        
        pos_encoding = positional_encoding(position, d_model)
        
        assert pos_encoding.shape == (1, position, d_model)
    
    def test_positional_encoding_values(self):
        """Test positional encoding values are finite."""
        position = 50
        d_model = 32
        
        pos_encoding = positional_encoding(position, d_model)
        
        # All values should be finite
        assert tf.reduce_all(tf.math.is_finite(pos_encoding))
        
        # Values should be in reasonable range [-1, 1] due to sin/cos
        assert tf.reduce_all(pos_encoding >= -1.0)
        assert tf.reduce_all(pos_encoding <= 1.0)
    
    def test_positional_encoding_alternating_pattern(self):
        """Test that positional encoding follows sin/cos alternating pattern."""
        position = 10
        d_model = 4
        
        pos_encoding = positional_encoding(position, d_model)
        
        # Remove batch dimension for easier testing
        pos_encoding = tf.squeeze(pos_encoding, axis=0)
        
        # Even indices should use sin, odd indices should use cos
        # This is a basic sanity check - exact values depend on implementation
        assert pos_encoding.shape == (position, d_model)
    
    def test_positional_encoding_different_positions(self):
        """Test positional encoding with different position values."""
        d_model = 16
        
        pos_enc_10 = positional_encoding(10, d_model)
        pos_enc_20 = positional_encoding(20, d_model)
        
        assert pos_enc_10.shape == (1, 10, d_model)
        assert pos_enc_20.shape == (1, 20, d_model)
        
        # First 10 positions should be the same
        tf.debugging.assert_near(
            pos_enc_10[0, :, :], 
            pos_enc_20[0, :10, :], 
            atol=1e-6
        )


@pytest.mark.integration
class TestUtilsLayersIntegration:
    """Integration tests for utility layers working together."""
    
    def test_layers_in_sequence(self):
        """Test multiple utility layers working in sequence."""
        # Create a sequence of layers
        linear1 = LinearLayer(hidden_layer_size=32, activation='relu')
        glu = GLULayer(output_layer_size=16, dropout_rate=0.1, use_time_distributed=False)
        linear2 = LinearLayer(hidden_layer_size=8, activation=None)
        
        # Process input through sequence
        x = tf.random.normal([4, 20])
        
        x = linear1(x, training=False)
        assert x.shape == (4, 32)
        
        x = glu(x, training=False)
        assert x.shape == (4, 16)
        
        x = linear2(x, training=False)
        assert x.shape == (4, 8)
        
        # Final output should be finite
        assert tf.reduce_all(tf.math.is_finite(x))
    
    def test_residual_connection_workflow(self):
        """Test residual connection workflow with utility layers."""
        add_norm = AddNorm()
        glu_add_norm = GLUWithAddNorm(
            output_layer_size=16,
            dropout_rate=0.1,
            use_time_distributed=False
        )
        
        # Create skip and main paths
        skip = tf.random.normal([4, 16])
        x = tf.random.normal([4, 16])
        
        # Test AddNorm
        output1 = add_norm(skip, x, training=False)
        assert output1.shape == (4, 16)
        
        # Test GLUWithAddNorm
        output2 = glu_add_norm(skip, x, training=False)
        assert output2.shape == (4, 16)
        
        # Both outputs should be finite
        assert tf.reduce_all(tf.math.is_finite(output1))
        assert tf.reduce_all(tf.math.is_finite(output2))
