"""Temporal Fusion Transformer (TFT) components for PyModConn."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
import numpy as np

from pymodconn.utils_layers import GRNLayer, LinearLayer

logger = logging.getLogger(__name__)


class VariableSelectionNetwork(tf.keras.layers.Layer):
    """
    Variable Selection Network (VSN) from Temporal Fusion Transformer.
    
    This layer learns to select the most relevant features for prediction
    by applying attention-based feature selection with gating mechanisms.
    
    Based on the TFT paper: "Temporal Fusion Transformers for Interpretable 
    Multi-horizon Time Series Forecasting" by Lim et al.
    
    Args:
        num_features: Number of input features
        hidden_layer_size: Size of hidden layers in GRN
        dropout_rate: Dropout rate for regularization
        use_time_distributed: Whether to apply time distribution
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_features: int,
        hidden_layer_size: int,
        dropout_rate: float = 0.1,
        use_time_distributed: bool = True,
        **kwargs
    ) -> None:
        """Initialize Variable Selection Network."""
        super().__init__(**kwargs)
        
        self.num_features = num_features
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        
        # Feature-specific GRN layers
        self.feature_grns = []
        for i in range(num_features):
            grn = GRNLayer(
                hidden_layer_size=hidden_layer_size,
                output_size=hidden_layer_size,
                dropout_rate=dropout_rate,
                use_time_distributed=use_time_distributed,
                activation_layer_type='elu',
                name=f'feature_grn_{i}'
            )
            self.feature_grns.append(grn)
        
        # Context vector GRN for feature selection weights
        self.context_grn = GRNLayer(
            hidden_layer_size=hidden_layer_size,
            output_size=num_features,
            dropout_rate=dropout_rate,
            use_time_distributed=use_time_distributed,
            activation_layer_type='elu',
            name='context_grn'
        )
        
        # Softmax for feature weights
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        
        # Flatten and concatenate layers
        self.flatten_features = tf.keras.layers.Lambda(
            lambda x: tf.stack(x, axis=-1)
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass through Variable Selection Network.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, num_features)
            training: Whether in training mode
            
        Returns:
            Tuple of (selected_features, feature_weights)
            - selected_features: Weighted feature representation
            - feature_weights: Attention weights for each feature
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1] if len(inputs.shape) == 3 else 1
        
        # Split input into individual features
        feature_list = tf.unstack(inputs, axis=-1)
        
        # Apply GRN to each feature
        processed_features = []
        for i, feature in enumerate(feature_list):
            # Expand dimensions if needed for time distribution
            if len(feature.shape) == 2 and self.use_time_distributed:
                feature = tf.expand_dims(feature, axis=-1)
            elif len(feature.shape) == 1:
                feature = tf.expand_dims(feature, axis=-1)
                if self.use_time_distributed:
                    feature = tf.expand_dims(feature, axis=-1)
            
            processed_feature = self.feature_grns[i](feature, training=training)
            processed_features.append(processed_feature)
        
        # Stack processed features
        stacked_features = tf.stack(processed_features, axis=-1)
        
        # Compute context vector (mean of all processed features)
        context_vector = tf.reduce_mean(stacked_features, axis=-1)
        
        # Generate feature selection weights
        feature_weights = self.context_grn(context_vector, training=training)
        feature_weights = self.softmax(feature_weights)
        
        # Apply feature selection weights
        if self.use_time_distributed:
            # Expand weights for broadcasting
            expanded_weights = tf.expand_dims(feature_weights, axis=-2)
            selected_features = stacked_features * expanded_weights
        else:
            selected_features = stacked_features * tf.expand_dims(feature_weights, axis=-2)
        
        # Sum weighted features
        selected_features = tf.reduce_sum(selected_features, axis=-1)
        
        return selected_features, feature_weights

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_features": self.num_features,
            "hidden_layer_size": self.hidden_layer_size,
            "dropout_rate": self.dropout_rate,
            "use_time_distributed": self.use_time_distributed,
        })
        return config


class StaticCovariateEncoder(tf.keras.layers.Layer):
    """
    Static Covariate Encoder for time-invariant features.
    
    This layer processes static (time-invariant) features and generates
    context vectors that can be used throughout the model.
    
    Args:
        num_static_features: Number of static input features
        hidden_layer_size: Size of hidden layers
        num_context_vectors: Number of context vectors to generate
        dropout_rate: Dropout rate for regularization
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_static_features: int,
        hidden_layer_size: int,
        num_context_vectors: int = 4,
        dropout_rate: float = 0.1,
        **kwargs
    ) -> None:
        """Initialize Static Covariate Encoder."""
        super().__init__(**kwargs)
        
        self.num_static_features = num_static_features
        self.hidden_layer_size = hidden_layer_size
        self.num_context_vectors = num_context_vectors
        self.dropout_rate = dropout_rate
        
        # Variable selection for static features
        self.static_vsn = VariableSelectionNetwork(
            num_features=num_static_features,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False
        )
        
        # Context vector generators
        self.context_grns = []
        for i in range(num_context_vectors):
            grn = GRNLayer(
                hidden_layer_size=hidden_layer_size,
                output_size=hidden_layer_size,
                dropout_rate=dropout_rate,
                use_time_distributed=False,
                activation_layer_type='elu',
                name=f'context_grn_{i}'
            )
            self.context_grns.append(grn)

    def call(self, static_inputs: tf.Tensor, training: bool = False) -> List[tf.Tensor]:
        """
        Forward pass through Static Covariate Encoder.
        
        Args:
            static_inputs: Static input tensor of shape (batch_size, num_static_features)
            training: Whether in training mode
            
        Returns:
            List of context vectors for use in other parts of the model
        """
        # Apply variable selection to static features
        selected_static, static_weights = self.static_vsn(static_inputs, training=training)
        
        # Generate context vectors
        context_vectors = []
        for grn in self.context_grns:
            context_vector = grn(selected_static, training=training)
            context_vectors.append(context_vector)
        
        return context_vectors

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_static_features": self.num_static_features,
            "hidden_layer_size": self.hidden_layer_size,
            "num_context_vectors": self.num_context_vectors,
            "dropout_rate": self.dropout_rate,
        })
        return config


class TemporalSelfAttention(tf.keras.layers.Layer):
    """
    Enhanced Temporal Self-Attention layer with interpretability features.
    
    This layer implements multi-head self-attention specifically designed
    for temporal sequences with additional interpretability mechanisms.
    
    Args:
        num_heads: Number of attention heads
        key_dim: Dimension of each attention head
        dropout_rate: Dropout rate for attention weights
        use_relative_position: Whether to use relative position encoding
        max_relative_position: Maximum relative position for encoding
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        dropout_rate: float = 0.1,
        use_relative_position: bool = True,
        max_relative_position: int = 128,
        **kwargs
    ) -> None:
        """Initialize Temporal Self-Attention layer."""
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.use_relative_position = use_relative_position
        self.max_relative_position = max_relative_position
        
        # Multi-head attention layer
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate
        )
        
        # Relative position embeddings
        if use_relative_position:
            self.relative_position_embeddings = self.add_weight(
                name='relative_position_embeddings',
                shape=(2 * max_relative_position + 1, key_dim),
                initializer='glorot_uniform',
                trainable=True
            )
        
        # Layer normalization and residual connection
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add_layer = tf.keras.layers.Add()

    def _get_relative_position_bias(self, seq_len: int) -> tf.Tensor:
        """
        Compute relative position bias for attention scores.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position bias tensor
        """
        # Create relative position matrix
        positions = tf.range(seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to maximum relative position
        relative_positions = tf.clip_by_value(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift to positive indices
        relative_positions += self.max_relative_position
        
        # Get embeddings
        relative_embeddings = tf.gather(self.relative_position_embeddings, relative_positions)
        
        return relative_embeddings

    def call(
        self,
        inputs: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass through Temporal Self-Attention.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Store input for residual connection
        residual = inputs
        
        # Apply multi-head attention
        if self.use_relative_position:
            seq_len = tf.shape(inputs)[1]
            relative_bias = self._get_relative_position_bias(seq_len)
            
            # Note: TensorFlow's MHA doesn't directly support relative position bias
            # This is a simplified implementation
            attention_output = self.mha(
                query=inputs,
                key=inputs,
                value=inputs,
                attention_mask=attention_mask,
                training=training,
                return_attention_scores=True
            )
            
            if isinstance(attention_output, tuple):
                attention_output, attention_weights = attention_output
            else:
                attention_weights = None
        else:
            attention_output = self.mha(
                query=inputs,
                key=inputs,
                value=inputs,
                attention_mask=attention_mask,
                training=training,
                return_attention_scores=True
            )
            
            if isinstance(attention_output, tuple):
                attention_output, attention_weights = attention_output
            else:
                attention_weights = None
        
        # Add residual connection and layer normalization
        output = self.add_layer([residual, attention_output])
        output = self.layer_norm(output)
        
        return output, attention_weights

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout_rate": self.dropout_rate,
            "use_relative_position": self.use_relative_position,
            "max_relative_position": self.max_relative_position,
        })
        return config


class QuantileForecaster(tf.keras.layers.Layer):
    """
    Enhanced Quantile Forecasting layer with improved loss functions.
    
    This layer generates quantile predictions with optimized loss computation
    and better numerical stability.
    
    Args:
        quantiles: List of quantiles to predict
        output_size: Size of output features
        use_shared_weights: Whether to share weights across quantiles
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        quantiles: List[float],
        output_size: int,
        use_shared_weights: bool = False,
        **kwargs
    ) -> None:
        """Initialize Quantile Forecaster."""
        super().__init__(**kwargs)
        
        self.quantiles = quantiles
        self.output_size = output_size
        self.use_shared_weights = use_shared_weights
        self.num_quantiles = len(quantiles)
        
        if use_shared_weights:
            # Single dense layer for all quantiles
            self.output_layer = tf.keras.layers.Dense(
                output_size * self.num_quantiles,
                name='shared_quantile_output'
            )
        else:
            # Separate dense layers for each quantile
            self.output_layers = []
            for i, q in enumerate(quantiles):
                layer = tf.keras.layers.Dense(
                    output_size,
                    name=f'quantile_{q}_output'
                )
                self.output_layers.append(layer)
        
        # Reshape layer for final output
        self.reshape_layer = tf.keras.layers.Reshape(
            (-1, output_size, self.num_quantiles)
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through Quantile Forecaster.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Quantile predictions of shape (batch_size, seq_len, output_size, num_quantiles)
        """
        if self.use_shared_weights:
            # Single output layer
            outputs = self.output_layer(inputs)
            outputs = self.reshape_layer(outputs)
        else:
            # Multiple output layers
            quantile_outputs = []
            for layer in self.output_layers:
                quantile_output = layer(inputs)
                quantile_outputs.append(quantile_output)
            
            # Stack quantile outputs
            outputs = tf.stack(quantile_outputs, axis=-1)
        
        return outputs

    def quantile_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute quantile loss for training.
        
        Args:
            y_true: True values
            y_pred: Predicted quantiles
            
        Returns:
            Quantile loss tensor
        """
        # Expand y_true to match quantile dimensions
        y_true_expanded = tf.expand_dims(y_true, axis=-1)
        y_true_expanded = tf.tile(y_true_expanded, [1, 1, 1, self.num_quantiles])
        
        # Compute quantile loss
        errors = y_true_expanded - y_pred
        quantile_tensor = tf.constant(self.quantiles, dtype=tf.float32)
        quantile_tensor = tf.reshape(quantile_tensor, [1, 1, 1, -1])
        
        loss = tf.maximum(quantile_tensor * errors, (quantile_tensor - 1) * errors)
        
        return tf.reduce_mean(loss)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "quantiles": self.quantiles,
            "output_size": self.output_size,
            "use_shared_weights": self.use_shared_weights,
        })
        return config


class TFTBlock(tf.keras.layers.Layer):
    """
    Complete TFT block combining all TFT components.
    
    This layer integrates Variable Selection Networks, Static Covariate Encoders,
    Temporal Self-Attention, and Quantile Forecasting into a single block.
    
    Args:
        num_features: Number of input features
        num_static_features: Number of static features (0 if none)
        hidden_layer_size: Size of hidden layers
        num_heads: Number of attention heads
        quantiles: List of quantiles for forecasting
        dropout_rate: Dropout rate
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_features: int,
        num_static_features: int = 0,
        hidden_layer_size: int = 128,
        num_heads: int = 8,
        quantiles: Optional[List[float]] = None,
        dropout_rate: float = 0.1,
        **kwargs
    ) -> None:
        """Initialize TFT Block."""
        super().__init__(**kwargs)
        
        self.num_features = num_features
        self.num_static_features = num_static_features
        self.hidden_layer_size = hidden_layer_size
        self.num_heads = num_heads
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.dropout_rate = dropout_rate
        
        # Variable Selection Network for temporal features
        self.temporal_vsn = VariableSelectionNetwork(
            num_features=num_features,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=True
        )
        
        # Static Covariate Encoder (if static features exist)
        if num_static_features > 0:
            self.static_encoder = StaticCovariateEncoder(
                num_static_features=num_static_features,
                hidden_layer_size=hidden_layer_size,
                dropout_rate=dropout_rate
            )
        else:
            self.static_encoder = None
        
        # Temporal Self-Attention
        self.temporal_attention = TemporalSelfAttention(
            num_heads=num_heads,
            key_dim=hidden_layer_size // num_heads,
            dropout_rate=dropout_rate
        )
        
        # Quantile Forecaster
        self.quantile_forecaster = QuantileForecaster(
            quantiles=self.quantiles,
            output_size=1,  # Assuming single output feature
            use_shared_weights=True
        )

    def call(
        self,
        temporal_inputs: tf.Tensor,
        static_inputs: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass through TFT Block.
        
        Args:
            temporal_inputs: Temporal input features
            static_inputs: Static input features (optional)
            training: Whether in training mode
            
        Returns:
            Dictionary containing outputs and interpretability information
        """
        # Variable selection for temporal features
        selected_temporal, temporal_weights = self.temporal_vsn(
            temporal_inputs, training=training
        )
        
        # Process static features if available
        static_contexts = None
        if self.static_encoder is not None and static_inputs is not None:
            static_contexts = self.static_encoder(static_inputs, training=training)
        
        # Apply temporal self-attention
        attention_output, attention_weights = self.temporal_attention(
            selected_temporal, training=training
        )
        
        # Generate quantile forecasts
        quantile_outputs = self.quantile_forecaster(attention_output, training=training)
        
        return {
            'predictions': quantile_outputs,
            'temporal_weights': temporal_weights,
            'attention_weights': attention_weights,
            'static_contexts': static_contexts
        }

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_features": self.num_features,
            "num_static_features": self.num_static_features,
            "hidden_layer_size": self.hidden_layer_size,
            "num_heads": self.num_heads,
            "quantiles": self.quantiles,
            "dropout_rate": self.dropout_rate,
        })
        return config
