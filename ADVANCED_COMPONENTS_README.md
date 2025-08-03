# PyModConn Advanced Components

This document describes the new advanced components added to PyModConn for state-of-the-art time series forecasting and sequence modeling.

## Overview

PyModConn has been enhanced with cutting-edge deep learning components that provide:

- **Memory-efficient attention mechanisms** (Flash Attention)
- **Advanced feature selection** (Variable Selection Networks from TFT)
- **Improved normalization techniques** (RMSNorm, Adaptive LayerNorm, etc.)
- **Modern positional encodings** (Rotary Position Embedding, Relative Position Embedding)
- **Integrated advanced architectures** combining all components seamlessly

## New Components

### 1. Flash Attention (`pymodconn.flash_attention_layer`)

Flash Attention provides memory-efficient attention computation that reduces memory complexity from O(n²) to O(n).

#### Key Features:
- Block-wise attention computation
- Online softmax for numerical stability
- Significant memory savings for long sequences
- Drop-in replacement for standard MultiHeadAttention

#### Usage:
```python
from pymodconn import FlashAttentionLayer, FlashMultiHeadAttentionBlock

# Basic Flash Attention
flash_attention = FlashAttentionLayer(
    num_heads=8,
    key_dim=64,
    block_size=32,
    dropout_rate=0.1
)

# Flash Attention with residual connections and normalization
flash_mha_block = FlashMultiHeadAttentionBlock(
    num_heads=8,
    key_dim=64,
    dropout_rate=0.1,
    use_layer_norm=True
)
```

### 2. Temporal Fusion Transformer Components (`pymodconn.tft_components`)

Implementation of key components from the Temporal Fusion Transformer paper, providing interpretable and high-performance time series forecasting.

#### Variable Selection Networks (VSN)
Automatically selects the most relevant features for prediction:

```python
from pymodconn import VariableSelectionNetwork

vsn = VariableSelectionNetwork(
    num_features=10,
    hidden_layer_size=128,
    dropout_rate=0.1,
    use_time_distributed=True
)

selected_features, feature_weights = vsn(inputs)
```

#### Static Covariate Encoder
Processes time-invariant features:

```python
from pymodconn import StaticCovariateEncoder

static_encoder = StaticCovariateEncoder(
    num_static_features=5,
    hidden_layer_size=128,
    num_context_vectors=4
)

context_vectors = static_encoder(static_inputs)
```

#### Enhanced Temporal Self-Attention
Multi-head attention with interpretability features:

```python
from pymodconn import TemporalSelfAttention

temporal_attention = TemporalSelfAttention(
    num_heads=8,
    key_dim=16,
    use_relative_position=True,
    max_relative_position=128
)

output, attention_weights = temporal_attention(inputs)
```

#### Quantile Forecaster
Generates probabilistic predictions:

```python
from pymodconn import QuantileForecaster

quantile_forecaster = QuantileForecaster(
    quantiles=[0.1, 0.5, 0.9],
    output_size=1,
    use_shared_weights=True
)

quantile_predictions = quantile_forecaster(inputs)
```

#### Complete TFT Block
Integrates all TFT components:

```python
from pymodconn import TFTBlock

tft_block = TFTBlock(
    num_features=10,
    num_static_features=3,
    hidden_layer_size=128,
    num_heads=8,
    quantiles=[0.1, 0.5, 0.9]
)

results = tft_block(temporal_inputs, static_inputs)
```

### 3. Advanced Normalization (`pymodconn.advanced_normalization`)

Modern normalization techniques for improved training stability and performance.

#### RMSNorm
More efficient alternative to LayerNorm:

```python
from pymodconn import RMSNorm

rms_norm = RMSNorm(epsilon=1e-6, scale=True, center=False)
normalized = rms_norm(inputs)
```

#### Adaptive Layer Normalization
Context-dependent normalization parameters:

```python
from pymodconn import AdaptiveLayerNorm

adaptive_norm = AdaptiveLayerNorm(context_dim=64)
normalized = adaptive_norm([inputs, context])
```

#### Group Normalization
Normalizes within channel groups:

```python
from pymodconn import GroupNorm

group_norm = GroupNorm(groups=32, epsilon=1e-6)
normalized = group_norm(inputs)
```

#### Spectral Normalization
Constrains spectral norm of weight matrices:

```python
from pymodconn import SpectralNorm
import tensorflow as tf

# Wrap any layer with spectral normalization
dense_layer = tf.keras.layers.Dense(128)
spectral_dense = SpectralNorm(dense_layer, power_iterations=1)
```

#### Layer Scale
Learnable scaling for deep networks:

```python
from pymodconn import LayerScale

layer_scale = LayerScale(init_value=1e-4)
scaled_output = layer_scale(inputs)
```

### 4. Advanced Positional Encodings (`pymodconn.positional_encodings`)

Modern positional encoding techniques for better sequence understanding.

#### Rotary Position Embedding (RoPE)
Encodes position by rotating query and key vectors:

```python
from pymodconn import RotaryPositionalEmbedding

rope = RotaryPositionalEmbedding(
    dim=64,
    max_seq_len=2048,
    base=10000.0
)

rotated_query, rotated_key = rope(query, key)
```

#### Relative Positional Embedding
Incorporates relative position information:

```python
from pymodconn import RelativePositionalEmbedding

rel_pos = RelativePositionalEmbedding(
    num_heads=8,
    max_relative_position=128
)

modified_scores, rel_values = rel_pos(query, attention_scores)
```

#### Learnable Positional Embedding
Fully learnable position embeddings:

```python
from pymodconn import LearnablePositionalEmbedding

learnable_pos = LearnablePositionalEmbedding(
    max_seq_len=1024,
    embed_dim=512
)

positioned_inputs = learnable_pos(inputs)
```

#### Enhanced Sinusoidal Embedding
Improved version of classic sinusoidal encoding:

```python
from pymodconn import SinusoidalPositionalEmbedding

sin_pos = SinusoidalPositionalEmbedding(
    embed_dim=512,
    max_seq_len=1024,
    temperature=10000.0,
    learnable_scale=True
)

positioned_inputs = sin_pos(inputs)
```

#### Unified Positional Encoding Block
Easy switching between different encoding types:

```python
from pymodconn import PositionalEncodingBlock

pos_block = PositionalEncodingBlock(
    encoding_type='rope',  # 'rope', 'relative', 'learnable', 'sinusoidal'
    embed_dim=512,
    max_seq_len=1024,
    num_heads=8
)

encoded_inputs = pos_block(inputs)
```

### 5. Advanced Integration (`pymodconn.advanced_layers_integration`)

High-level components that integrate multiple advanced techniques.

#### Advanced Encoder Block
Combines Flash Attention, RMSNorm, RoPE, and GRN:

```python
from pymodconn import AdvancedEncoderBlock

encoder_block = AdvancedEncoderBlock(
    num_features=10,
    hidden_size=128,
    num_heads=8,
    use_flash_attention=True,
    use_rmsnorm=True,
    use_rope=True,
    dropout_rate=0.1
)

encoded_output = encoder_block(inputs)
```

#### Advanced Decoder Block
Integrates VSN, enhanced attention, and quantile forecasting:

```python
from pymodconn import AdvancedDecoderBlock

decoder_block = AdvancedDecoderBlock(
    num_features=5,
    num_static_features=3,
    hidden_size=128,
    num_heads=8,
    quantiles=[0.1, 0.5, 0.9],
    use_variable_selection=True,
    use_rmsnorm=True
)

decoder_results = decoder_block(
    decoder_inputs, 
    encoder_outputs, 
    static_inputs
)
```

#### Complete Advanced Model
Full sequence-to-sequence model with all advanced components:

```python
from pymodconn import AdvancedSequenceToSequenceModel, create_advanced_model_config

# Create configuration
config = create_advanced_model_config(
    num_past_features=10,
    num_future_features=5,
    num_static_features=3,
    hidden_size=128,
    num_heads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    quantiles=[0.1, 0.5, 0.9],
    use_flash_attention=True,
    use_rmsnorm=True,
    use_rope=True
)

# Create model
advanced_model = AdvancedSequenceToSequenceModel(**config)

# Forward pass
results = advanced_model(
    encoder_inputs=past_data,
    decoder_inputs=future_known,
    static_inputs=static_features
)
```

## Key Benefits

### Performance Improvements
- **Flash Attention**: Up to 4x memory reduction for long sequences
- **RMSNorm**: Faster and more stable training than LayerNorm
- **RoPE**: Better handling of relative positions in sequences
- **Variable Selection**: Automatic feature selection reduces overfitting

### Enhanced Interpretability
- **Attention weights**: Understand which time steps are important
- **Feature weights**: See which features are selected by VSN
- **Quantile predictions**: Uncertainty quantification
- **Static contexts**: Understand time-invariant feature contributions

### Ease of Use
- **Drop-in replacements**: Easy to replace existing components
- **Unified interfaces**: Consistent API across all components
- **Configuration helpers**: Simple model configuration
- **Comprehensive examples**: Ready-to-use code samples

## Usage Examples

### Basic Usage
```python
import numpy as np
from pymodconn import (
    FlashAttentionLayer,
    VariableSelectionNetwork,
    RMSNorm,
    RotaryPositionalEmbedding
)

# Sample data
batch_size, seq_len, features = 32, 96, 64
inputs = np.random.randn(batch_size, seq_len, features)

# Flash Attention
flash_attn = FlashAttentionLayer(num_heads=8, key_dim=64)
attn_output = flash_attn(inputs)

# Variable Selection
vsn = VariableSelectionNetwork(num_features=features, hidden_layer_size=128)
selected_features, weights = vsn(inputs)

# RMS Normalization
rms_norm = RMSNorm()
normalized = rms_norm(inputs)
```

### Advanced Model Creation
```python
from pymodconn import create_advanced_model_config, AdvancedSequenceToSequenceModel

# Configuration
config = create_advanced_model_config(
    num_past_features=20,
    num_future_features=10,
    num_static_features=5,
    hidden_size=256,
    num_heads=16,
    use_flash_attention=True,
    use_rmsnorm=True,
    use_rope=True
)

# Model
model = AdvancedSequenceToSequenceModel(**config)

# Usage
results = model(encoder_inputs, decoder_inputs, static_inputs)
predictions = results['predictions']
interpretability = results['interpretability']
```

### Keras Integration
```python
import tensorflow as tf
from pymodconn import AdvancedSequenceToSequenceModel, create_advanced_model_config

# Define inputs
encoder_input = tf.keras.layers.Input(shape=(96, 20))
decoder_input = tf.keras.layers.Input(shape=(24, 10))
static_input = tf.keras.layers.Input(shape=(5,))

# Create advanced layer
config = create_advanced_model_config(
    num_past_features=20,
    num_future_features=10,
    num_static_features=5
)
advanced_layer = AdvancedSequenceToSequenceModel(**config)

# Apply layer
outputs = advanced_layer(
    encoder_inputs=encoder_input,
    decoder_inputs=decoder_input,
    static_inputs=static_input
)

# Create Keras model
keras_model = tf.keras.Model(
    inputs=[encoder_input, decoder_input, static_input],
    outputs=outputs['predictions']
)
```

## Running Examples

A comprehensive example file is provided at `pymodconn/examples/advanced_model_example.py`:

```bash
cd pymodconn/examples
python advanced_model_example.py
```

This example demonstrates:
- Individual component usage
- Advanced encoder-decoder blocks
- Complete model integration
- TFT block functionality
- Keras model integration

## Performance Considerations

### Memory Usage
- Flash Attention reduces memory usage significantly for sequences > 512
- RMSNorm is more memory efficient than LayerNorm
- Variable Selection Networks add minimal overhead

### Training Speed
- Flash Attention can be slower for very short sequences (< 128)
- RMSNorm typically trains 10-15% faster than LayerNorm
- Rotary Position Embedding adds minimal computational overhead

### Model Size
- Advanced components add parameters but improve performance
- Use `use_shared_weights=True` in QuantileForecaster to reduce parameters
- Layer scaling helps with very deep networks

## Migration Guide

### From Standard Attention to Flash Attention
```python
# Before
attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)

# After
from pymodconn import FlashMultiHeadAttentionBlock
attention = FlashMultiHeadAttentionBlock(num_heads=8, key_dim=64)
```

### From LayerNorm to RMSNorm
```python
# Before
norm = tf.keras.layers.LayerNormalization()

# After
from pymodconn import RMSNorm
norm = RMSNorm()
```

### Adding Variable Selection
```python
# Before
dense = tf.keras.layers.Dense(hidden_size)
output = dense(inputs)

# After
from pymodconn import VariableSelectionNetwork
vsn = VariableSelectionNetwork(num_features=input_features, hidden_layer_size=hidden_size)
output, weights = vsn(inputs)
```

## Future Enhancements

Planned additions include:
- **Mamba/State Space Models**: For very long sequence modeling
- **Mixture of Experts**: For handling diverse patterns
- **Linear Attention**: O(n) complexity attention mechanisms
- **Advanced regularization**: DropPath, Stochastic Depth
- **Fourier Feature Networks**: For periodic pattern modeling

## Contributing

To contribute new advanced components:

1. Follow the existing code structure and documentation style
2. Include comprehensive docstrings and type hints
3. Add unit tests for new components
4. Update this README with usage examples
5. Ensure compatibility with existing PyModConn architecture

## References

- **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **Temporal Fusion Transformer**: Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- **RMSNorm**: Zhang & Sennrich, "Root Mean Square Layer Normalization"
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **Relative Position**: Shaw et al., "Self-Attention with Relative Position Representations"
