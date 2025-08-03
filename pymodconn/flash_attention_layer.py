"""Flash Attention layer implementation for PyModConn."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class FlashAttentionLayer(tf.keras.layers.Layer):
    """
    Flash Attention implementation for memory-efficient attention computation.
    
    This layer implements a memory-efficient version of multi-head attention
    that reduces memory complexity from O(n²) to O(n) by computing attention
    in blocks and using online softmax computation.
    
    Based on the Flash Attention paper: "FlashAttention: Fast and Memory-Efficient 
    Exact Attention with IO-Awareness" by Dao et al.
    
    Args:
        num_heads: Number of attention heads
        key_dim: Size of each attention head for query and key
        value_dim: Size of each attention head for value (defaults to key_dim)
        block_size: Block size for flash attention computation (default: 64)
        dropout_rate: Dropout rate for attention weights
        use_bias: Whether to use bias in linear projections
        **kwargs: Additional keyword arguments for the base layer
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: Optional[int] = None,
        block_size: int = 64,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        **kwargs
    ) -> None:
        """Initialize FlashAttentionLayer with specified parameters."""
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim or key_dim
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        # Ensure dimensions are divisible by number of heads
        if self.key_dim % self.num_heads != 0:
            raise ValueError(f"key_dim ({self.key_dim}) must be divisible by num_heads ({self.num_heads})")
        if self.value_dim % self.num_heads != 0:
            raise ValueError(f"value_dim ({self.value_dim}) must be divisible by num_heads ({self.num_heads})")
            
        self.head_dim = self.key_dim // self.num_heads
        self.value_head_dim = self.value_dim // self.num_heads
        
        # Scale factor for attention scores
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # Linear projections
        self.query_dense = tf.keras.layers.Dense(
            self.key_dim, use_bias=self.use_bias, name="query"
        )
        self.key_dense = tf.keras.layers.Dense(
            self.key_dim, use_bias=self.use_bias, name="key"
        )
        self.value_dense = tf.keras.layers.Dense(
            self.value_dim, use_bias=self.use_bias, name="value"
        )
        self.output_dense = tf.keras.layers.Dense(
            self.value_dim, use_bias=self.use_bias, name="output"
        )
        
        # Dropout layer
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def _split_heads(self, x: tf.Tensor, head_dim: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, head_dim)."""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        x = tf.reshape(x, [batch_size, seq_len, self.num_heads, head_dim])
        return tf.transpose(x, [0, 2, 1, 3])  # (batch, heads, seq_len, head_dim)

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Combine heads back to original shape."""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        
        x = tf.transpose(x, [0, 2, 1, 3])  # (batch, seq_len, heads, head_dim)
        return tf.reshape(x, [batch_size, seq_len, self.value_dim])

    def _flash_attention_block(
        self,
        q_block: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute flash attention for a single query block.
        
        Args:
            q_block: Query block of shape (batch, heads, block_size, head_dim)
            k: Full key tensor of shape (batch, heads, seq_len, head_dim)
            v: Full value tensor of shape (batch, heads, seq_len, head_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output_block, max_scores, sum_exp_scores)
        """
        # Compute attention scores for the block
        scores = tf.matmul(q_block, k, transpose_b=True) * self.scale
        
        if mask is not None:
            scores += mask
        
        # Online softmax computation
        max_scores = tf.reduce_max(scores, axis=-1, keepdims=True)
        exp_scores = tf.exp(scores - max_scores)
        sum_exp_scores = tf.reduce_sum(exp_scores, axis=-1, keepdims=True)
        
        # Compute attention weights and output
        attention_weights = exp_scores / sum_exp_scores
        attention_weights = self.dropout_layer(attention_weights)
        
        output_block = tf.matmul(attention_weights, v)
        
        return output_block, max_scores, sum_exp_scores

    def _flash_attention(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute flash attention using block-wise computation.
        
        Args:
            query: Query tensor of shape (batch, heads, seq_len, head_dim)
            key: Key tensor of shape (batch, heads, seq_len, head_dim)
            value: Value tensor of shape (batch, heads, seq_len, head_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, heads, seq_len, value_head_dim)
        """
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[2]
        
        # Initialize output and normalization terms
        output = tf.zeros_like(value)
        max_scores = tf.fill([batch_size, self.num_heads, seq_len, 1], -tf.float32.max)
        sum_exp_scores = tf.zeros([batch_size, self.num_heads, seq_len, 1])
        
        # Process query in blocks
        num_blocks = tf.math.ceil(tf.cast(seq_len, tf.float32) / tf.cast(self.block_size, tf.float32))
        num_blocks = tf.cast(num_blocks, tf.int32)
        
        for i in tf.range(num_blocks):
            start_idx = i * self.block_size
            end_idx = tf.minimum((i + 1) * self.block_size, seq_len)
            
            # Extract query block
            q_block = query[:, :, start_idx:end_idx, :]
            
            # Extract corresponding mask block if provided
            mask_block = None
            if mask is not None:
                mask_block = mask[:, :, start_idx:end_idx, :]
            
            # Compute attention for this block
            output_block, max_block, sum_block = self._flash_attention_block(
                q_block, key, value, mask_block
            )
            
            # Update running statistics using online algorithm
            old_max = max_scores[:, :, start_idx:end_idx, :]
            new_max = tf.maximum(old_max, max_block)
            
            # Rescale previous outputs and sums
            exp_old = tf.exp(old_max - new_max)
            exp_new = tf.exp(max_block - new_max)
            
            old_sum = sum_exp_scores[:, :, start_idx:end_idx, :]
            new_sum = exp_old * old_sum + exp_new * sum_block
            
            # Update output
            old_output = output[:, :, start_idx:end_idx, :]
            new_output = (exp_old * old_sum * old_output + exp_new * sum_block * output_block) / new_sum
            
            # Store updated values
            output = tf.concat([
                output[:, :, :start_idx, :],
                new_output,
                output[:, :, end_idx:, :]
            ], axis=2)
            
            max_scores = tf.concat([
                max_scores[:, :, :start_idx, :],
                new_max,
                max_scores[:, :, end_idx:, :]
            ], axis=2)
            
            sum_exp_scores = tf.concat([
                sum_exp_scores[:, :, :start_idx, :],
                new_sum,
                sum_exp_scores[:, :, end_idx:, :]
            ], axis=2)
        
        return output

    def call(
        self,
        query: tf.Tensor,
        key: Optional[tf.Tensor] = None,
        value: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through Flash Attention layer.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, features)
            key: Key tensor (defaults to query if None)
            value: Value tensor (defaults to key if None)
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, seq_len, value_dim)
        """
        if key is None:
            key = query
        if value is None:
            value = key
            
        # Linear projections
        q = self.query_dense(query)
        k = self.key_dense(key)
        v = self.value_dense(value)
        
        # Split into multiple heads
        q = self._split_heads(q, self.head_dim)
        k = self._split_heads(k, self.head_dim)
        v = self._split_heads(v, self.value_head_dim)
        
        # Prepare attention mask if provided
        if attention_mask is not None:
            # Expand mask dimensions for heads
            attention_mask = tf.expand_dims(attention_mask, axis=1)  # Add head dimension
            attention_mask = tf.expand_dims(attention_mask, axis=1)  # Add query dimension
            attention_mask = (1.0 - tf.cast(attention_mask, tf.float32)) * -1e9
        
        # Apply flash attention
        attention_output = self._flash_attention(q, k, v, attention_mask)
        
        # Combine heads
        attention_output = self._combine_heads(attention_output)
        
        # Final linear projection
        output = self.output_dense(attention_output)
        
        return output

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "block_size": self.block_size,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FlashAttentionLayer":
        """Create layer from configuration."""
        return cls(**config)


class FlashMultiHeadAttentionBlock(tf.keras.layers.Layer):
    """
    Flash Multi-Head Attention block compatible with existing PyModConn architecture.
    
    This layer provides a drop-in replacement for the standard MHA block
    with Flash Attention for improved memory efficiency.
    
    Args:
        num_heads: Number of attention heads
        key_dim: Size of each attention head
        block_size: Block size for flash attention
        dropout_rate: Dropout rate
        use_layer_norm: Whether to apply layer normalization
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        block_size: int = 64,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
        **kwargs
    ) -> None:
        """Initialize FlashMultiHeadAttentionBlock."""
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Flash attention layer
        self.flash_attention = FlashAttentionLayer(
            num_heads=num_heads,
            key_dim=key_dim,
            block_size=block_size,
            dropout_rate=dropout_rate
        )
        
        # Layer normalization
        if self.use_layer_norm:
            self.layer_norm = tf.keras.layers.LayerNormalization()
        
        # Residual connection
        self.add_layer = tf.keras.layers.Add()

    def call(
        self,
        query: tf.Tensor,
        key: Optional[tf.Tensor] = None,
        value: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through Flash MHA block.
        
        Args:
            query: Query tensor
            key: Key tensor (defaults to query)
            value: Value tensor (defaults to key)
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Output tensor with residual connection and normalization
        """
        # Store input for residual connection
        residual = query
        
        # Apply flash attention
        attention_output = self.flash_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            training=training
        )
        
        # Add residual connection
        output = self.add_layer([residual, attention_output])
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config().copy()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "block_size": self.block_size,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
        })
        return config
