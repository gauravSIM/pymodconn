"""
Advanced PyModConn Model Example

This example demonstrates how to use the new advanced components in PyModConn,
including Flash Attention, TFT components, advanced normalization, and 
modern positional encodings.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional

# Import PyModConn components
from pymodconn import (
    # Advanced integrated models
    AdvancedEncoderBlock,
    AdvancedDecoderBlock,
    AdvancedSequenceToSequenceModel,
    create_advanced_model_config,
    
    # Individual components
    FlashAttentionLayer,
    VariableSelectionNetwork,
    RMSNorm,
    RotaryPositionalEmbedding,
    TFTBlock
)


def create_sample_data(
    num_samples: int = 1000,
    seq_len_past: int = 96,
    seq_len_future: int = 24,
    num_past_features: int = 10,
    num_future_features: int = 5,
    num_static_features: int = 3
) -> Dict[str, np.ndarray]:
    """
    Create sample time series data for demonstration.
    
    Args:
        num_samples: Number of samples
        seq_len_past: Length of past sequence
        seq_len_future: Length of future sequence
        num_past_features: Number of past features
        num_future_features: Number of future features
        num_static_features: Number of static features
        
    Returns:
        Dictionary containing sample data
    """
    # Generate synthetic time series data
    np.random.seed(42)
    
    # Past data (encoder inputs)
    past_data = np.random.randn(num_samples, seq_len_past, num_past_features)
    
    # Add some temporal patterns
    time_steps = np.arange(seq_len_past)
    for i in range(num_past_features):
        # Add sinusoidal patterns with different frequencies
        freq = 0.1 + i * 0.05
        past_data[:, :, i] += np.sin(2 * np.pi * freq * time_steps)
    
    # Future known data (decoder inputs)
    future_known = np.random.randn(num_samples, seq_len_future, num_future_features)
    
    # Future unknown data (targets)
    future_unknown = np.random.randn(num_samples, seq_len_future, 1)
    
    # Add correlation with past data
    for i in range(num_samples):
        # Simple correlation: future depends on last few past values
        last_values = past_data[i, -5:, :3].mean(axis=0)
        future_unknown[i, :, 0] += 0.5 * last_values.mean()
    
    # Static features (time-invariant)
    static_features = np.random.randn(num_samples, num_static_features)
    
    return {
        'past_data': past_data.astype(np.float32),
        'future_known': future_known.astype(np.float32),
        'future_unknown': future_unknown.astype(np.float32),
        'static_features': static_features.astype(np.float32)
    }


def example_individual_components():
    """Demonstrate individual advanced components."""
    print("=== Individual Components Example ===")
    
    # Sample data
    batch_size, seq_len, features = 32, 96, 64
    sample_input = tf.random.normal((batch_size, seq_len, features))
    
    print(f"Input shape: {sample_input.shape}")
    
    # 1. Flash Attention
    print("\n1. Flash Attention Layer")
    flash_attention = FlashAttentionLayer(
        num_heads=8,
        key_dim=64,
        block_size=32,
        dropout_rate=0.1
    )
    
    flash_output = flash_attention(sample_input, training=False)
    print(f"Flash Attention output shape: {flash_output.shape}")
    
    # 2. Variable Selection Network
    print("\n2. Variable Selection Network")
    vsn = VariableSelectionNetwork(
        num_features=features,
        hidden_layer_size=128,
        dropout_rate=0.1
    )
    
    selected_features, feature_weights = vsn(sample_input, training=False)
    print(f"Selected features shape: {selected_features.shape}")
    print(f"Feature weights shape: {feature_weights.shape}")
    
    # 3. RMSNorm
    print("\n3. RMS Normalization")
    rms_norm = RMSNorm()
    normalized_output = rms_norm(sample_input, training=False)
    print(f"RMSNorm output shape: {normalized_output.shape}")
    
    # 4. Rotary Position Embedding
    print("\n4. Rotary Position Embedding")
    rope = RotaryPositionalEmbedding(dim=64, max_seq_len=seq_len)
    
    # Split input for query and key
    query = sample_input[:, :, :64]
    key = sample_input[:, :, :64]
    
    # Reshape for multi-head attention format
    query = tf.reshape(query, (batch_size, seq_len, 8, 8))
    key = tf.reshape(key, (batch_size, seq_len, 8, 8))
    
    rotated_query, rotated_key = rope(query, key)
    print(f"Rotated query shape: {rotated_query.shape}")
    print(f"Rotated key shape: {rotated_key.shape}")


def example_advanced_encoder_decoder():
    """Demonstrate advanced encoder-decoder blocks."""
    print("\n=== Advanced Encoder-Decoder Example ===")
    
    # Sample data
    batch_size = 16
    past_seq_len, future_seq_len = 96, 24
    past_features, future_features = 10, 5
    hidden_size = 128
    
    # Create sample inputs
    encoder_inputs = tf.random.normal((batch_size, past_seq_len, past_features))
    decoder_inputs = tf.random.normal((batch_size, future_seq_len, future_features))
    
    print(f"Encoder inputs shape: {encoder_inputs.shape}")
    print(f"Decoder inputs shape: {decoder_inputs.shape}")
    
    # Advanced Encoder Block
    print("\n1. Advanced Encoder Block")
    encoder_block = AdvancedEncoderBlock(
        num_features=past_features,
        hidden_size=hidden_size,
        num_heads=8,
        use_flash_attention=True,
        use_rmsnorm=True,
        use_rope=True,
        dropout_rate=0.1
    )
    
    encoder_outputs = encoder_block(encoder_inputs, training=False)
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    
    # Advanced Decoder Block
    print("\n2. Advanced Decoder Block")
    decoder_block = AdvancedDecoderBlock(
        num_features=future_features,
        num_static_features=0,
        hidden_size=hidden_size,
        num_heads=8,
        quantiles=[0.1, 0.5, 0.9],
        use_variable_selection=True,
        use_rmsnorm=True,
        dropout_rate=0.1
    )
    
    decoder_results = decoder_block(
        decoder_inputs, 
        encoder_outputs, 
        training=False
    )
    
    print(f"Decoder predictions shape: {decoder_results['predictions'].shape}")
    print(f"Variable weights shape: {decoder_results['variable_weights'].shape}")
    print("Available decoder outputs:", list(decoder_results.keys()))


def example_complete_advanced_model():
    """Demonstrate the complete advanced sequence-to-sequence model."""
    print("\n=== Complete Advanced Model Example ===")
    
    # Generate sample data
    data = create_sample_data(
        num_samples=100,
        seq_len_past=96,
        seq_len_future=24,
        num_past_features=10,
        num_future_features=5,
        num_static_features=3
    )
    
    print("Sample data shapes:")
    for key, value in data.items():
        print(f"  {key}: {value.shape}")
    
    # Create model configuration
    model_config = create_advanced_model_config(
        num_past_features=10,
        num_future_features=5,
        num_static_features=3,
        hidden_size=128,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        quantiles=[0.1, 0.5, 0.9],
        use_flash_attention=True,
        use_rmsnorm=True,
        use_rope=True,
        dropout_rate=0.1
    )
    
    print(f"\nModel configuration: {model_config}")
    
    # Create the advanced model
    advanced_model = AdvancedSequenceToSequenceModel(**model_config)
    
    # Forward pass
    results = advanced_model(
        encoder_inputs=data['past_data'][:10],  # Use first 10 samples
        decoder_inputs=data['future_known'][:10],
        static_inputs=data['static_features'][:10],
        training=False
    )
    
    print(f"\nModel results:")
    print(f"  Predictions shape: {results['predictions'].shape}")
    print(f"  Encoder outputs shape: {results['encoder_outputs'].shape}")
    print(f"  Decoder outputs shape: {results['decoder_outputs'].shape}")
    print(f"  Number of decoder layers with predictions: {len(results['all_layer_predictions'])}")
    
    # Interpretability information
    interpretability = results['interpretability']
    print(f"\nInterpretability information:")
    print(f"  Variable weights layers: {len(interpretability['variable_weights'])}")
    print(f"  Self-attention weights layers: {len(interpretability['self_attention_weights'])}")
    print(f"  Static contexts layers: {len(interpretability['static_contexts'])}")


def example_tft_block():
    """Demonstrate the complete TFT block."""
    print("\n=== TFT Block Example ===")
    
    # Sample data
    batch_size, seq_len = 16, 48
    num_features, num_static_features = 8, 4
    
    temporal_inputs = tf.random.normal((batch_size, seq_len, num_features))
    static_inputs = tf.random.normal((batch_size, num_static_features))
    
    print(f"Temporal inputs shape: {temporal_inputs.shape}")
    print(f"Static inputs shape: {static_inputs.shape}")
    
    # Create TFT block
    tft_block = TFTBlock(
        num_features=num_features,
        num_static_features=num_static_features,
        hidden_layer_size=64,
        num_heads=4,
        quantiles=[0.1, 0.5, 0.9],
        dropout_rate=0.1
    )
    
    # Forward pass
    tft_results = tft_block(
        temporal_inputs=temporal_inputs,
        static_inputs=static_inputs,
        training=False
    )
    
    print(f"\nTFT Block results:")
    print(f"  Predictions shape: {tft_results['predictions'].shape}")
    print(f"  Temporal weights shape: {tft_results['temporal_weights'].shape}")
    print(f"  Attention weights shape: {tft_results['attention_weights'].shape}")
    print(f"  Static contexts: {len(tft_results['static_contexts']) if tft_results['static_contexts'] else 'None'}")


def example_keras_model_integration():
    """Demonstrate how to integrate advanced components into a Keras model."""
    print("\n=== Keras Model Integration Example ===")
    
    # Model parameters
    past_seq_len, future_seq_len = 96, 24
    past_features, future_features = 10, 5
    static_features = 3
    hidden_size = 128
    
    # Define inputs
    encoder_input = tf.keras.layers.Input(
        shape=(past_seq_len, past_features), 
        name='encoder_input'
    )
    decoder_input = tf.keras.layers.Input(
        shape=(future_seq_len, future_features), 
        name='decoder_input'
    )
    static_input = tf.keras.layers.Input(
        shape=(static_features,), 
        name='static_input'
    )
    
    # Create model configuration
    model_config = create_advanced_model_config(
        num_past_features=past_features,
        num_future_features=future_features,
        num_static_features=static_features,
        hidden_size=hidden_size,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        quantiles=[0.1, 0.5, 0.9],
        use_flash_attention=True,
        use_rmsnorm=True,
        use_rope=True,
        dropout_rate=0.1
    )
    
    # Create the advanced model layer
    advanced_layer = AdvancedSequenceToSequenceModel(**model_config)
    
    # Apply the layer
    outputs = advanced_layer(
        encoder_inputs=encoder_input,
        decoder_inputs=decoder_input,
        static_inputs=static_input
    )
    
    # Create Keras model
    keras_model = tf.keras.Model(
        inputs=[encoder_input, decoder_input, static_input],
        outputs=outputs['predictions'],
        name='AdvancedTimeSeriesModel'
    )
    
    # Model summary
    print("Keras Model Summary:")
    keras_model.summary()
    
    # Test with sample data
    sample_data = create_sample_data(num_samples=10)
    
    predictions = keras_model([
        sample_data['past_data'],
        sample_data['future_known'],
        sample_data['static_features']
    ])
    
    print(f"\nPredictions shape: {predictions.shape}")
    print("Model successfully created and tested!")


def main():
    """Run all examples."""
    print("PyModConn Advanced Components Examples")
    print("=" * 50)
    
    try:
        # Run individual component examples
        example_individual_components()
        
        # Run encoder-decoder examples
        example_advanced_encoder_decoder()
        
        # Run complete model example
        example_complete_advanced_model()
        
        # Run TFT block example
        example_tft_block()
        
        # Run Keras integration example
        example_keras_model_integration()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey benefits of the advanced components:")
        print("- Flash Attention: Reduced memory usage for long sequences")
        print("- Variable Selection Networks: Automatic feature selection")
        print("- RMSNorm: More stable training than LayerNorm")
        print("- Rotary Position Embedding: Better positional understanding")
        print("- TFT Components: Enhanced interpretability and performance")
        print("- Integrated Architecture: Easy-to-use advanced models")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
