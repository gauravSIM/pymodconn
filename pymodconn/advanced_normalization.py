"""Advanced normalization layers for PyModConn."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class RMSNorm(tf.keras.layers.Layer):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is a simpler and more efficient alternative to LayerNorm that
    normalizes using only the root mean square of the inputs, without
    centering (subtracting the mean). This often provides better training
    stability and faster convergence.
    
    Based on the paper: "Root Mean Square Layer Normalization" by Zhang & Sennrich.
    
    Args:
        epsilon: Small constant for numerical stability
        center: Whether to learn an additive offset parameter (beta)
        scale: Whether to learn a multiplicative scale parameter (gamma)
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        center: bool = False,
        scale: bool = True,
        **kwargs
    ) -> None:
        """Initialize RMSNorm layer."""
        super().__init__(**kwargs)
        
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        """Build the layer weights."""
        super().build(input_shape)
        
        # Get the feature dimension (last dimension)
        feature_dim = input_shape[-1]
        
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=(feature_dim,),
                initializer='ones',
                trainable=True
            )
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=(feature_dim,),
                initializer='zeros',
                trainable=True
            )
        else:
            self.beta = None

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through RMSNorm layer.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Normalized tensor
        """
        # Compute RMS
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        rms = tf.sqrt(mean_square + self.epsilon)
        
        # Normalize
        normalized = inputs / rms
        
        # Apply scale and center if enabled
        if self.scale:
            normalized = normalized * self.gamma
            
        if self.center:
            normalized = normalized + self.beta
            
        return normalized

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
        })
        return config


class AdaptiveLayerNorm(tf.keras.layers.Layer):
    """
    Adaptive Layer Normalization with context-dependent parameters.
    
    This layer applies layer normalization with parameters that are
    dynamically computed based on a context vector, allowing for
    more flexible normalization that can adapt to different conditions.
    
    Args:
        epsilon: Small constant for numerical stability
        context_dim: Dimension of the context vector
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        context_dim: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize Adaptive Layer Normalization."""
        super().__init__(**kwargs)
        
        self.epsilon = epsilon
        self.context_dim = context_dim

    def build(self, input_shape):
        """Build the layer weights."""
        super().build(input_shape)
        
        # Get the feature dimension (last dimension)
        if isinstance(input_shape, list):
            # Multiple inputs: [main_input, context_input]
            feature_dim = input_shape[0][-1]
            if self.context_dim is None:
                self.context_dim = input_shape[1][-1]
        else:
            # Single input
            feature_dim = input_shape[-1]
            if self.context_dim is None:
                self.context_dim = feature_dim
        
        # Context-to-gamma transformation
        self.gamma_dense = tf.keras.layers.Dense(
            feature_dim,
            activation='sigmoid',
            name='gamma_transform'
        )
        
        # Context-to-beta transformation
        self.beta_dense = tf.keras.layers.Dense(
            feature_dim,
            activation='tanh',
            name='beta_transform'
        )
        
        # Default gamma and beta
        self.default_gamma = self.add_weight(
            name='default_gamma',
            shape=(feature_dim,),
            initializer='ones',
            trainable=True
        )
        
        self.default_beta = self.add_weight(
            name='default_beta',
            shape=(feature_dim,),
            initializer='zeros',
            trainable=True
        )

    def call(
        self, 
        inputs, 
        context: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through Adaptive Layer Normalization.
        
        Args:
            inputs: Input tensor or list of [input_tensor, context_tensor]
            context: Context tensor (if not provided in inputs list)
            training: Whether in training mode
            
        Returns:
            Normalized tensor with adaptive parameters
        """
        if isinstance(inputs, list):
            # Multiple inputs provided
            main_input, context_input = inputs
            context = context_input
        else:
            # Single input provided
            main_input = inputs
            
        # Compute mean and variance
        mean = tf.reduce_mean(main_input, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(main_input - mean), axis=-1, keepdims=True)
        
        # Normalize
        normalized = (main_input - mean) / tf.sqrt(variance + self.epsilon)
        
        if context is not None:
            # Compute adaptive parameters from context
            gamma = self.gamma_dense(context)
            beta = self.beta_dense(context)
            
            # Expand dimensions if needed for broadcasting
            if len(main_input.shape) > len(gamma.shape):
                for _ in range(len(main_input.shape) - len(gamma.shape)):
                    gamma = tf.expand_dims(gamma, axis=-2)
                    beta = tf.expand_dims(beta, axis=-2)
        else:
            # Use default parameters
            gamma = self.default_gamma
            beta = self.default_beta
        
        # Apply adaptive normalization
        output = normalized * gamma + beta
        
        return output

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "epsilon": self.epsilon,
            "context_dim": self.context_dim,
        })
        return config


class GroupNorm(tf.keras.layers.Layer):
    """
    Group Normalization layer.
    
    Group Normalization divides channels into groups and normalizes
    within each group. This can be more stable than Layer Normalization
    for certain architectures and batch sizes.
    
    Based on the paper: "Group Normalization" by Wu & He.
    
    Args:
        groups: Number of groups to divide channels into
        epsilon: Small constant for numerical stability
        center: Whether to learn an additive offset parameter
        scale: Whether to learn a multiplicative scale parameter
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        groups: int = 32,
        epsilon: float = 1e-6,
        center: bool = True,
        scale: bool = True,
        **kwargs
    ) -> None:
        """Initialize Group Normalization layer."""
        super().__init__(**kwargs)
        
        self.groups = groups
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        """Build the layer weights."""
        super().build(input_shape)
        
        # Get the feature dimension (last dimension)
        feature_dim = input_shape[-1]
        
        if feature_dim % self.groups != 0:
            raise ValueError(
                f"Number of channels ({feature_dim}) must be divisible by "
                f"number of groups ({self.groups})"
            )
        
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=(feature_dim,),
                initializer='ones',
                trainable=True
            )
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=(feature_dim,),
                initializer='zeros',
                trainable=True
            )
        else:
            self.beta = None

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through Group Normalization layer.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Group normalized tensor
        """
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        
        # Handle different input shapes
        if len(inputs.shape) == 3:
            # (batch, seq_len, features)
            seq_len = input_shape[1]
            features = input_shape[2]
            
            # Reshape for group normalization
            reshaped = tf.reshape(inputs, [batch_size, seq_len, self.groups, features // self.groups])
            
            # Compute mean and variance over group dimensions
            mean = tf.reduce_mean(reshaped, axis=[3], keepdims=True)
            variance = tf.reduce_mean(tf.square(reshaped - mean), axis=[3], keepdims=True)
            
            # Normalize
            normalized = (reshaped - mean) / tf.sqrt(variance + self.epsilon)
            
            # Reshape back
            normalized = tf.reshape(normalized, [batch_size, seq_len, features])
            
        elif len(inputs.shape) == 2:
            # (batch, features)
            features = input_shape[1]
            
            # Reshape for group normalization
            reshaped = tf.reshape(inputs, [batch_size, self.groups, features // self.groups])
            
            # Compute mean and variance over group dimensions
            mean = tf.reduce_mean(reshaped, axis=[2], keepdims=True)
            variance = tf.reduce_mean(tf.square(reshaped - mean), axis=[2], keepdims=True)
            
            # Normalize
            normalized = (reshaped - mean) / tf.sqrt(variance + self.epsilon)
            
            # Reshape back
            normalized = tf.reshape(normalized, [batch_size, features])
            
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")
        
        # Apply scale and center if enabled
        if self.scale:
            normalized = normalized * self.gamma
            
        if self.center:
            normalized = normalized + self.beta
            
        return normalized

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "groups": self.groups,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
        })
        return config


class SpectralNorm(tf.keras.layers.Wrapper):
    """
    Spectral Normalization wrapper for any layer with weights.
    
    Spectral normalization constrains the spectral norm (largest singular value)
    of weight matrices to improve training stability, especially in adversarial
    training and GANs.
    
    Based on the paper: "Spectral Normalization for Generative Adversarial Networks"
    by Miyato et al.
    
    Args:
        layer: The layer to wrap with spectral normalization
        power_iterations: Number of power iterations for spectral norm estimation
        **kwargs: Additional keyword arguments for the wrapper
    """

    def __init__(
        self,
        layer: tf.keras.layers.Layer,
        power_iterations: int = 1,
        **kwargs
    ) -> None:
        """Initialize Spectral Normalization wrapper."""
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        """Build the wrapped layer and spectral norm variables."""
        super().build(input_shape)
        
        # Find weight matrices in the wrapped layer
        self.weight_vars = []
        self.u_vars = []
        
        for weight in self.layer.trainable_weights:
            if len(weight.shape) >= 2:  # Only normalize matrices
                self.weight_vars.append(weight)
                
                # Initialize u vector for power iteration
                u_shape = (1, weight.shape[-1])
                u = self.add_weight(
                    name=f'u_{weight.name.replace(":", "_")}',
                    shape=u_shape,
                    initializer='random_normal',
                    trainable=False
                )
                self.u_vars.append(u)

    def _spectral_normalize(self, weight: tf.Tensor, u: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply spectral normalization to a weight matrix.
        
        Args:
            weight: Weight matrix to normalize
            u: Left singular vector estimate
            
        Returns:
            Tuple of (normalized_weight, updated_u)
        """
        # Reshape weight to 2D matrix
        weight_shape = tf.shape(weight)
        weight_2d = tf.reshape(weight, [-1, weight_shape[-1]])
        
        # Power iteration to estimate spectral norm
        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, weight_2d, transpose_b=True), axis=1)
            u = tf.nn.l2_normalize(tf.matmul(v, weight_2d), axis=1)
        
        # Compute spectral norm
        sigma = tf.reduce_sum(tf.matmul(u, weight_2d) * v)
        
        # Normalize weight
        normalized_weight = weight / sigma
        
        return normalized_weight, u

    def call(self, inputs, training=None, **kwargs):
        """Forward pass with spectral normalization applied."""
        if training:
            # Apply spectral normalization to weights
            normalized_weights = []
            updated_us = []
            
            for weight, u in zip(self.weight_vars, self.u_vars):
                normalized_weight, updated_u = self._spectral_normalize(weight, u)
                normalized_weights.append(normalized_weight)
                updated_us.append(updated_u)
                
                # Update u variable
                u.assign(updated_u)
                
                # Replace weight in layer
                weight.assign(normalized_weight)
        
        # Call the wrapped layer
        return self.layer(inputs, training=training, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Get wrapper configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "power_iterations": self.power_iterations,
        })
        return config


class LayerScale(tf.keras.layers.Layer):
    """
    Layer Scale for improving training stability in deep networks.
    
    Layer Scale applies a learnable scaling factor to layer outputs,
    which can help with training very deep networks by controlling
    the contribution of each layer.
    
    Based on the paper: "Going deeper with Image Transformers" by Touvron et al.
    
    Args:
        init_value: Initial value for the scaling parameter
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        init_value: float = 1e-4,
        **kwargs
    ) -> None:
        """Initialize Layer Scale."""
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        """Build the layer scale parameter."""
        super().build(input_shape)
        
        # Get the feature dimension (last dimension)
        feature_dim = input_shape[-1]
        
        self.scale = self.add_weight(
            name='scale',
            shape=(feature_dim,),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through Layer Scale.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Scaled tensor
        """
        return inputs * self.scale

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "init_value": self.init_value,
        })
        return config
