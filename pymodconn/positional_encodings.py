"""Advanced positional encoding layers for PyModConn."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class RotaryPositionalEmbedding(tf.keras.layers.Layer):
    """
    Rotary Position Embedding (RoPE) layer.
    
    RoPE encodes positional information by rotating query and key vectors
    in a way that naturally incorporates relative position information
    into the attention mechanism. This often provides better performance
    than absolute positional encodings.
    
    Based on the paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    by Su et al.
    
    Args:
        dim: Dimension of the embeddings (should be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for the geometric progression (default: 10000)
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        **kwargs
    ) -> None:
        """Initialize Rotary Position Embedding."""
        super().__init__(**kwargs)
        
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        self.inv_freq = self._compute_inv_freq()

    def _compute_inv_freq(self) -> tf.Tensor:
        """Compute inverse frequencies for rotary embedding."""
        # Create frequency matrix
        inv_freq = 1.0 / (self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))
        return inv_freq

    def _compute_cos_sin_tables(self, seq_len: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute cosine and sine tables for rotary embedding.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos_table, sin_table)
        """
        # Create position indices
        positions = tf.range(seq_len, dtype=tf.float32)
        
        # Compute angles
        angles = tf.einsum('i,j->ij', positions, self.inv_freq)
        
        # Duplicate angles for pairs
        angles = tf.repeat(angles, 2, axis=-1)
        
        # Compute cos and sin
        cos_table = tf.cos(angles)
        sin_table = tf.sin(angles)
        
        return cos_table, sin_table

    def _rotate_half(self, x: tf.Tensor) -> tf.Tensor:
        """
        Rotate half of the features.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Rotated tensor
        """
        # Split into two halves
        x1, x2 = tf.split(x, 2, axis=-1)
        
        # Rotate: [-x2, x1]
        return tf.concat([-x2, x1], axis=-1)

    def _apply_rotary_pos_emb(
        self, 
        x: tf.Tensor, 
        cos: tf.Tensor, 
        sin: tf.Tensor
    ) -> tf.Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor
            cos: Cosine table
            sin: Sine table
            
        Returns:
            Tensor with rotary position embedding applied
        """
        # Apply rotation: x * cos + rotate_half(x) * sin
        return x * cos + self._rotate_half(x) * sin

    def call(
        self, 
        query: tf.Tensor, 
        key: tf.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            query: Query tensor of shape (batch, seq_len, heads, dim)
            key: Key tensor of shape (batch, seq_len, heads, dim)
            seq_len: Sequence length (inferred if None)
            
        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        if seq_len is None:
            seq_len = tf.shape(query)[1]
        
        # Compute cos and sin tables
        cos_table, sin_table = self._compute_cos_sin_tables(seq_len)
        
        # Expand dimensions for broadcasting
        # Shape: (1, seq_len, 1, dim)
        cos_table = cos_table[None, :, None, :]
        sin_table = sin_table[None, :, None, :]
        
        # Apply rotary embedding
        rotated_query = self._apply_rotary_pos_emb(query, cos_table, sin_table)
        rotated_key = self._apply_rotary_pos_emb(key, cos_table, sin_table)
        
        return rotated_query, rotated_key

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "dim": self.dim,
            "max_seq_len": self.max_seq_len,
            "base": self.base,
        })
        return config


class RelativePositionalEmbedding(tf.keras.layers.Layer):
    """
    Relative Positional Embedding layer.
    
    This layer computes relative position embeddings that can be added
    to attention scores to incorporate relative position information.
    
    Based on the paper: "Self-Attention with Relative Position Representations"
    by Shaw et al.
    
    Args:
        num_heads: Number of attention heads
        max_relative_position: Maximum relative position to consider
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        num_heads: int,
        max_relative_position: int = 128,
        **kwargs
    ) -> None:
        """Initialize Relative Positional Embedding."""
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        
        # Total number of relative positions: [-max_rel_pos, max_rel_pos]
        self.num_relative_positions = 2 * max_relative_position + 1

    def build(self, input_shape):
        """Build the relative position embedding weights."""
        super().build(input_shape)
        
        # Relative position embeddings for keys
        self.relative_positions_keys = self.add_weight(
            name='relative_positions_keys',
            shape=(self.num_relative_positions, self.num_heads),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Relative position embeddings for values
        self.relative_positions_values = self.add_weight(
            name='relative_positions_values',
            shape=(self.num_relative_positions, self.num_heads),
            initializer='glorot_uniform',
            trainable=True
        )

    def _get_relative_position_matrix(self, seq_len: int) -> tf.Tensor:
        """
        Get relative position matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position matrix of shape (seq_len, seq_len)
        """
        # Create position indices
        positions = tf.range(seq_len)
        
        # Compute relative positions
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to maximum relative position
        relative_positions = tf.clip_by_value(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift to positive indices
        relative_positions += self.max_relative_position
        
        return relative_positions

    def call(
        self, 
        query: tf.Tensor,
        attention_scores: tf.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply relative positional embedding.
        
        Args:
            query: Query tensor of shape (batch, heads, seq_len, dim)
            attention_scores: Attention scores of shape (batch, heads, seq_len, seq_len)
            seq_len: Sequence length (inferred if None)
            
        Returns:
            Tuple of (modified_attention_scores, relative_values_bias)
        """
        if seq_len is None:
            seq_len = tf.shape(query)[2]
        
        # Get relative position matrix
        relative_positions = self._get_relative_position_matrix(seq_len)
        
        # Get relative position embeddings for keys
        relative_keys = tf.gather(self.relative_positions_keys, relative_positions)
        # Shape: (seq_len, seq_len, num_heads)
        
        # Compute relative position bias for attention scores
        # query: (batch, heads, seq_len, dim)
        # We need to compute: query @ relative_keys.T
        query_for_rel = tf.transpose(query, [0, 2, 1, 3])  # (batch, seq_len, heads, dim)
        
        # Simplified relative position bias (this is a basic implementation)
        # In practice, you might want a more sophisticated computation
        relative_keys_bias = tf.transpose(relative_keys, [2, 0, 1])  # (heads, seq_len, seq_len)
        relative_keys_bias = relative_keys_bias[None, :, :, :]  # (1, heads, seq_len, seq_len)
        
        # Add relative position bias to attention scores
        modified_attention_scores = attention_scores + relative_keys_bias
        
        # Get relative position embeddings for values
        relative_values = tf.gather(self.relative_positions_values, relative_positions)
        relative_values = tf.transpose(relative_values, [2, 0, 1])  # (heads, seq_len, seq_len)
        relative_values = relative_values[None, :, :, :]  # (1, heads, seq_len, seq_len)
        
        return modified_attention_scores, relative_values

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "max_relative_position": self.max_relative_position,
        })
        return config


class LearnablePositionalEmbedding(tf.keras.layers.Layer):
    """
    Learnable Positional Embedding layer.
    
    This layer learns absolute positional embeddings that can be added
    to input embeddings. Unlike sinusoidal encodings, these are fully
    learnable parameters.
    
    Args:
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        **kwargs
    ) -> None:
        """Initialize Learnable Positional Embedding."""
        super().__init__(**kwargs)
        
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        """Build the positional embedding weights."""
        super().build(input_shape)
        
        self.position_embeddings = self.add_weight(
            name='position_embeddings',
            shape=(self.max_seq_len, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(
        self, 
        inputs: tf.Tensor,
        start_pos: int = 0
    ) -> tf.Tensor:
        """
        Add positional embeddings to inputs.
        
        Args:
            inputs: Input tensor of shape (batch, seq_len, embed_dim)
            start_pos: Starting position for positional embeddings
            
        Returns:
            Input tensor with positional embeddings added
        """
        seq_len = tf.shape(inputs)[1]
        
        # Get positional embeddings for the sequence
        position_embeddings = self.position_embeddings[start_pos:start_pos + seq_len]
        
        # Add positional embeddings to inputs
        return inputs + position_embeddings

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
        })
        return config


class SinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    """
    Sinusoidal Positional Embedding layer (enhanced version).
    
    This is an enhanced version of the classic sinusoidal positional
    encoding from the Transformer paper, with additional features
    like temperature scaling and learnable scaling factors.
    
    Args:
        embed_dim: Embedding dimension
        max_seq_len: Maximum sequence length
        temperature: Temperature for scaling frequencies
        learnable_scale: Whether to learn a scaling factor
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 10000,
        temperature: float = 10000.0,
        learnable_scale: bool = False,
        **kwargs
    ) -> None:
        """Initialize Sinusoidal Positional Embedding."""
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.learnable_scale = learnable_scale

    def build(self, input_shape):
        """Build the positional embedding."""
        super().build(input_shape)
        
        # Precompute positional encodings
        self.position_encodings = self._get_position_encodings()
        
        if self.learnable_scale:
            self.scale = self.add_weight(
                name='scale',
                shape=(1,),
                initializer='ones',
                trainable=True
            )
        else:
            self.scale = 1.0

    def _get_position_encodings(self) -> tf.Tensor:
        """
        Generate sinusoidal position encodings.
        
        Returns:
            Position encodings of shape (max_seq_len, embed_dim)
        """
        # Create position indices
        positions = tf.range(self.max_seq_len, dtype=tf.float32)[:, None]
        
        # Create dimension indices
        dimensions = tf.range(self.embed_dim, dtype=tf.float32)[None, :]
        
        # Compute angles
        angles = positions / tf.pow(self.temperature, (2 * (dimensions // 2)) / self.embed_dim)
        
        # Apply sin to even indices and cos to odd indices
        position_encodings = tf.where(
            tf.equal(dimensions % 2, 0),
            tf.sin(angles),
            tf.cos(angles)
        )
        
        return position_encodings

    def call(
        self, 
        inputs: tf.Tensor,
        start_pos: int = 0
    ) -> tf.Tensor:
        """
        Add sinusoidal positional embeddings to inputs.
        
        Args:
            inputs: Input tensor of shape (batch, seq_len, embed_dim)
            start_pos: Starting position for positional embeddings
            
        Returns:
            Input tensor with positional embeddings added
        """
        seq_len = tf.shape(inputs)[1]
        
        # Get positional encodings for the sequence
        position_encodings = self.position_encodings[start_pos:start_pos + seq_len]
        
        # Apply scaling if learnable
        position_encodings = position_encodings * self.scale
        
        # Add positional encodings to inputs
        return inputs + position_encodings

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "embed_dim": self.embed_dim,
            "max_seq_len": self.max_seq_len,
            "temperature": self.temperature,
            "learnable_scale": self.learnable_scale,
        })
        return config


class PositionalEncodingBlock(tf.keras.layers.Layer):
    """
    Unified Positional Encoding Block supporting multiple encoding types.
    
    This layer provides a unified interface for different types of
    positional encodings, making it easy to switch between them.
    
    Args:
        encoding_type: Type of positional encoding ('rope', 'relative', 'learnable', 'sinusoidal')
        embed_dim: Embedding dimension
        max_seq_len: Maximum sequence length
        num_heads: Number of attention heads (for relative encoding)
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        encoding_type: str = 'sinusoidal',
        embed_dim: int = 512,
        max_seq_len: int = 2048,
        num_heads: int = 8,
        **kwargs
    ) -> None:
        """Initialize Positional Encoding Block."""
        super().__init__(**kwargs)
        
        self.encoding_type = encoding_type
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        
        # Initialize the appropriate encoding layer
        if encoding_type == 'rope':
            self.encoding_layer = RotaryPositionalEmbedding(
                dim=embed_dim,
                max_seq_len=max_seq_len
            )
        elif encoding_type == 'relative':
            self.encoding_layer = RelativePositionalEmbedding(
                num_heads=num_heads,
                max_relative_position=max_seq_len // 4
            )
        elif encoding_type == 'learnable':
            self.encoding_layer = LearnablePositionalEmbedding(
                max_seq_len=max_seq_len,
                embed_dim=embed_dim
            )
        elif encoding_type == 'sinusoidal':
            self.encoding_layer = SinusoidalPositionalEmbedding(
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                learnable_scale=True
            )
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")

    def call(self, inputs, **kwargs):
        """
        Apply positional encoding.
        
        Args:
            inputs: Input tensor(s)
            **kwargs: Additional arguments for specific encoding types
            
        Returns:
            Encoded tensor(s)
        """
        return self.encoding_layer(inputs, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "encoding_type": self.encoding_type,
            "embed_dim": self.embed_dim,
            "max_seq_len": self.max_seq_len,
            "num_heads": self.num_heads,
        })
        return config
