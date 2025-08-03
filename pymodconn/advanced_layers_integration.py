"""Integration layer for advanced PyModConn components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from pymodconn.flash_attention_layer import FlashAttentionLayer, FlashMultiHeadAttentionBlock
from pymodconn.tft_components import (
    VariableSelectionNetwork, 
    StaticCovariateEncoder, 
    TemporalSelfAttention,
    QuantileForecaster,
    TFTBlock
)
from pymodconn.advanced_normalization import RMSNorm, AdaptiveLayerNorm, GroupNorm, LayerScale
from pymodconn.positional_encodings import (
    RotaryPositionalEmbedding, 
    RelativePositionalEmbedding,
    PositionalEncodingBlock
)
from pymodconn.utils_layers import GRNLayer, GLUWithAddNorm

logger = logging.getLogger(__name__)


class AdvancedEncoderBlock(tf.keras.layers.Layer):
    """
    Advanced Encoder Block integrating multiple state-of-the-art components.
    
    This block combines Flash Attention, RMSNorm, Rotary Position Embedding,
    and other advanced techniques for improved performance and efficiency.
    
    Args:
        num_features: Number of input features
        hidden_size: Hidden layer size
        num_heads: Number of attention heads
        use_flash_attention: Whether to use Flash Attention
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm
        use_rope: Whether to use Rotary Position Embedding
        dropout_rate: Dropout rate
        layer_scale_init: Initial value for layer scale
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_heads: int = 8,
        use_flash_attention: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        dropout_rate: float = 0.1,
        layer_scale_init: float = 1e-4,
        **kwargs
    ) -> None:
        """Initialize Advanced Encoder Block."""
        super().__init__(**kwargs)
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention
        self.use_rmsnorm = use_rmsnorm
        self.use_rope = use_rope
        self.dropout_rate = dropout_rate
        self.layer_scale_init = layer_scale_init
        
        # Input projection
        self.input_projection = tf.keras.layers.Dense(hidden_size)
        
        # Positional encoding
        if use_rope:
            self.pos_encoding = RotaryPositionalEmbedding(
                dim=hidden_size // num_heads,
                max_seq_len=2048
            )
        else:
            self.pos_encoding = PositionalEncodingBlock(
                encoding_type='sinusoidal',
                embed_dim=hidden_size,
                max_seq_len=2048
            )
        
        # Attention layer
        if use_flash_attention:
            self.attention = FlashMultiHeadAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_size // num_heads,
                dropout_rate=dropout_rate,
                use_layer_norm=False  # We'll use our own normalization
            )
        else:
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=hidden_size // num_heads,
                dropout=dropout_rate
            )
        
        # Normalization layers
        if use_rmsnorm:
            self.norm1 = RMSNorm()
            self.norm2 = RMSNorm()
        else:
            self.norm1 = tf.keras.layers.LayerNormalization()
            self.norm2 = tf.keras.layers.LayerNormalization()
        
        # Feed-forward network with GRN
        self.ffn = GRNLayer(
            hidden_layer_size=hidden_size * 4,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activation_layer_type='gelu'
        )
        
        # Layer scaling
        self.layer_scale1 = LayerScale(init_value=layer_scale_init)
        self.layer_scale2 = LayerScale(init_value=layer_scale_init)
        
        # Residual connections
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()

    def call(
        self, 
        inputs: tf.Tensor, 
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through Advanced Encoder Block.
        
        Args:
            inputs: Input tensor of shape (batch, seq_len, features)
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Encoded tensor
        """
        # Input projection
        x = self.input_projection(inputs)
        
        # Apply positional encoding
        if self.use_rope:
            # For RoPE, we need to split into query and key
            # This is a simplified version - in practice, you'd apply RoPE within attention
            x_pos = self.pos_encoding(x, x)  # Using same tensor for both q and k
            if isinstance(x_pos, tuple):
                x = x_pos[0]  # Use the first output (query)
        else:
            x = self.pos_encoding(x)
        
        # First residual block: Attention
        residual1 = x
        x = self.norm1(x, training=training)
        
        if self.use_flash_attention:
            x = self.attention(x, training=training)
        else:
            x = self.attention(x, x, attention_mask=attention_mask, training=training)
        
        x = self.layer_scale1(x, training=training)
        x = self.add1([residual1, x])
        
        # Second residual block: Feed-forward
        residual2 = x
        x = self.norm2(x, training=training)
        x = self.ffn(x, training=training)
        x = self.layer_scale2(x, training=training)
        x = self.add2([residual2, x])
        
        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_features": self.num_features,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "use_flash_attention": self.use_flash_attention,
            "use_rmsnorm": self.use_rmsnorm,
            "use_rope": self.use_rope,
            "dropout_rate": self.dropout_rate,
            "layer_scale_init": self.layer_scale_init,
        })
        return config


class AdvancedDecoderBlock(tf.keras.layers.Layer):
    """
    Advanced Decoder Block with TFT components and modern techniques.
    
    This block integrates Variable Selection Networks, enhanced attention,
    and advanced normalization for improved decoder performance.
    
    Args:
        num_features: Number of input features
        num_static_features: Number of static features
        hidden_size: Hidden layer size
        num_heads: Number of attention heads
        quantiles: List of quantiles for forecasting
        use_variable_selection: Whether to use Variable Selection Networks
        use_rmsnorm: Whether to use RMSNorm
        dropout_rate: Dropout rate
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_features: int,
        num_static_features: int = 0,
        hidden_size: int = 128,
        num_heads: int = 8,
        quantiles: Optional[List[float]] = None,
        use_variable_selection: bool = True,
        use_rmsnorm: bool = True,
        dropout_rate: float = 0.1,
        **kwargs
    ) -> None:
        """Initialize Advanced Decoder Block."""
        super().__init__(**kwargs)
        
        self.num_features = num_features
        self.num_static_features = num_static_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.use_variable_selection = use_variable_selection
        self.use_rmsnorm = use_rmsnorm
        self.dropout_rate = dropout_rate
        
        # Input projection
        self.input_projection = tf.keras.layers.Dense(hidden_size)
        
        # Variable Selection Network
        if use_variable_selection:
            self.vsn = VariableSelectionNetwork(
                num_features=num_features,
                hidden_layer_size=hidden_size,
                dropout_rate=dropout_rate,
                use_time_distributed=True
            )
        
        # Static covariate encoder
        if num_static_features > 0:
            self.static_encoder = StaticCovariateEncoder(
                num_static_features=num_static_features,
                hidden_layer_size=hidden_size,
                dropout_rate=dropout_rate
            )
        else:
            self.static_encoder = None
        
        # Enhanced temporal attention
        self.temporal_attention = TemporalSelfAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dropout_rate=dropout_rate,
            use_relative_position=True
        )
        
        # Cross attention for encoder-decoder interaction
        self.cross_attention = FlashMultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dropout_rate=dropout_rate,
            use_layer_norm=False
        )
        
        # Normalization layers
        if use_rmsnorm:
            self.norm1 = RMSNorm()
            self.norm2 = RMSNorm()
            self.norm3 = RMSNorm()
        else:
            self.norm1 = tf.keras.layers.LayerNormalization()
            self.norm2 = tf.keras.layers.LayerNormalization()
            self.norm3 = tf.keras.layers.LayerNormalization()
        
        # Feed-forward network
        self.ffn = GRNLayer(
            hidden_layer_size=hidden_size * 4,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activation_layer_type='gelu'
        )
        
        # Quantile forecaster
        self.quantile_forecaster = QuantileForecaster(
            quantiles=self.quantiles,
            output_size=1,
            use_shared_weights=True
        )
        
        # Residual connections
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()
        self.add3 = tf.keras.layers.Add()

    def call(
        self,
        decoder_inputs: tf.Tensor,
        encoder_outputs: tf.Tensor,
        static_inputs: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass through Advanced Decoder Block.
        
        Args:
            decoder_inputs: Decoder input tensor
            encoder_outputs: Encoder output tensor
            static_inputs: Optional static input tensor
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Dictionary containing predictions and interpretability information
        """
        # Input projection
        x = self.input_projection(decoder_inputs)
        
        # Variable selection
        if self.use_variable_selection:
            x, variable_weights = self.vsn(x, training=training)
        else:
            variable_weights = None
        
        # Process static inputs
        static_contexts = None
        if self.static_encoder is not None and static_inputs is not None:
            static_contexts = self.static_encoder(static_inputs, training=training)
        
        # First residual block: Self-attention
        residual1 = x
        x = self.norm1(x, training=training)
        x, self_attention_weights = self.temporal_attention(
            x, attention_mask=attention_mask, training=training
        )
        x = self.add1([residual1, x])
        
        # Second residual block: Cross-attention with encoder
        residual2 = x
        x = self.norm2(x, training=training)
        x = self.cross_attention(
            query=x, 
            key=encoder_outputs, 
            value=encoder_outputs,
            training=training
        )
        x = self.add2([residual2, x])
        
        # Third residual block: Feed-forward
        residual3 = x
        x = self.norm3(x, training=training)
        x = self.ffn(x, training=training)
        x = self.add3([residual3, x])
        
        # Generate quantile predictions
        predictions = self.quantile_forecaster(x, training=training)
        
        return {
            'predictions': predictions,
            'variable_weights': variable_weights,
            'self_attention_weights': self_attention_weights,
            'static_contexts': static_contexts,
            'final_features': x
        }

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_features": self.num_features,
            "num_static_features": self.num_static_features,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "quantiles": self.quantiles,
            "use_variable_selection": self.use_variable_selection,
            "use_rmsnorm": self.use_rmsnorm,
            "dropout_rate": self.dropout_rate,
        })
        return config


class AdvancedSequenceToSequenceModel(tf.keras.layers.Layer):
    """
    Complete Advanced Sequence-to-Sequence Model.
    
    This model integrates all advanced components into a complete
    sequence-to-sequence architecture for time series forecasting.
    
    Args:
        encoder_config: Configuration for encoder blocks
        decoder_config: Configuration for decoder blocks
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        **kwargs
    ) -> None:
        """Initialize Advanced Sequence-to-Sequence Model."""
        super().__init__(**kwargs)
        
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Build encoder layers
        self.encoder_layers = []
        for i in range(num_encoder_layers):
            encoder_layer = AdvancedEncoderBlock(
                **encoder_config,
                name=f'encoder_layer_{i}'
            )
            self.encoder_layers.append(encoder_layer)
        
        # Build decoder layers
        self.decoder_layers = []
        for i in range(num_decoder_layers):
            decoder_layer = AdvancedDecoderBlock(
                **decoder_config,
                name=f'decoder_layer_{i}'
            )
            self.decoder_layers.append(decoder_layer)

    def call(
        self,
        encoder_inputs: tf.Tensor,
        decoder_inputs: tf.Tensor,
        static_inputs: Optional[tf.Tensor] = None,
        encoder_mask: Optional[tf.Tensor] = None,
        decoder_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through the complete model.
        
        Args:
            encoder_inputs: Encoder input tensor
            decoder_inputs: Decoder input tensor
            static_inputs: Optional static input tensor
            encoder_mask: Optional encoder attention mask
            decoder_mask: Optional decoder attention mask
            training: Whether in training mode
            
        Returns:
            Dictionary containing predictions and interpretability information
        """
        # Encode
        encoder_outputs = encoder_inputs
        for encoder_layer in self.encoder_layers:
            encoder_outputs = encoder_layer(
                encoder_outputs,
                attention_mask=encoder_mask,
                training=training
            )
        
        # Decode
        decoder_outputs = decoder_inputs
        all_predictions = []
        all_interpretability = {
            'variable_weights': [],
            'self_attention_weights': [],
            'static_contexts': []
        }
        
        for decoder_layer in self.decoder_layers:
            decoder_result = decoder_layer(
                decoder_outputs,
                encoder_outputs,
                static_inputs=static_inputs,
                attention_mask=decoder_mask,
                training=training
            )
            
            decoder_outputs = decoder_result['final_features']
            all_predictions.append(decoder_result['predictions'])
            
            # Collect interpretability information
            if decoder_result['variable_weights'] is not None:
                all_interpretability['variable_weights'].append(decoder_result['variable_weights'])
            if decoder_result['self_attention_weights'] is not None:
                all_interpretability['self_attention_weights'].append(decoder_result['self_attention_weights'])
            if decoder_result['static_contexts'] is not None:
                all_interpretability['static_contexts'].append(decoder_result['static_contexts'])
        
        # Use the last layer's predictions as final output
        final_predictions = all_predictions[-1]
        
        return {
            'predictions': final_predictions,
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs,
            'interpretability': all_interpretability,
            'all_layer_predictions': all_predictions
        }

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "encoder_config": self.encoder_config,
            "decoder_config": self.decoder_config,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
        })
        return config


def create_advanced_model_config(
    num_past_features: int,
    num_future_features: int,
    num_static_features: int = 0,
    hidden_size: int = 128,
    num_heads: int = 8,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    quantiles: Optional[List[float]] = None,
    use_flash_attention: bool = True,
    use_rmsnorm: bool = True,
    use_rope: bool = True,
    dropout_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Create a configuration for the advanced model.
    
    Args:
        num_past_features: Number of past features
        num_future_features: Number of future features
        num_static_features: Number of static features
        hidden_size: Hidden layer size
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        quantiles: List of quantiles for forecasting
        use_flash_attention: Whether to use Flash Attention
        use_rmsnorm: Whether to use RMSNorm
        use_rope: Whether to use Rotary Position Embedding
        dropout_rate: Dropout rate
        
    Returns:
        Model configuration dictionary
    """
    encoder_config = {
        'num_features': num_past_features,
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'use_flash_attention': use_flash_attention,
        'use_rmsnorm': use_rmsnorm,
        'use_rope': use_rope,
        'dropout_rate': dropout_rate
    }
    
    decoder_config = {
        'num_features': num_future_features,
        'num_static_features': num_static_features,
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'quantiles': quantiles or [0.1, 0.5, 0.9],
        'use_variable_selection': True,
        'use_rmsnorm': use_rmsnorm,
        'dropout_rate': dropout_rate
    }
    
    return {
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
        'num_encoder_layers': num_encoder_layers,
        'num_decoder_layers': num_decoder_layers
    }
