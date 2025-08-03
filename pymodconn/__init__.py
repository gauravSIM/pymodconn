#   -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""
PyModConn: Modular Control-Oriented Neural Networks

A Python package for developing sequence-to-sequence control-oriented 
deep neural networks in a highly modular way.

This package now includes advanced components such as:
- Flash Attention for memory-efficient attention computation
- Temporal Fusion Transformer (TFT) components with Variable Selection Networks
- Advanced normalization techniques (RMSNorm, Adaptive LayerNorm, etc.)
- Modern positional encodings (RoPE, Relative Position Embedding, etc.)
- Integrated advanced encoder-decoder architectures
"""
from __future__ import annotations

# Core model generation
from pymodconn.model_gen import ModelGen

# Advanced attention mechanisms
from pymodconn.flash_attention_layer import (
    FlashAttentionLayer,
    FlashMultiHeadAttentionBlock
)

# TFT components
from pymodconn.tft_components import (
    VariableSelectionNetwork,
    StaticCovariateEncoder,
    TemporalSelfAttention,
    QuantileForecaster,
    TFTBlock
)

# Advanced normalization
from pymodconn.advanced_normalization import (
    RMSNorm,
    AdaptiveLayerNorm,
    GroupNorm,
    SpectralNorm,
    LayerScale
)

# Positional encodings
from pymodconn.positional_encodings import (
    RotaryPositionalEmbedding,
    RelativePositionalEmbedding,
    LearnablePositionalEmbedding,
    SinusoidalPositionalEmbedding,
    PositionalEncodingBlock
)

# Advanced integrated models
from pymodconn.advanced_layers_integration import (
    AdvancedEncoderBlock,
    AdvancedDecoderBlock,
    AdvancedSequenceToSequenceModel,
    create_advanced_model_config
)

__version__ = "2.1.0"
__all__ = [
    # Core
    "ModelGen",
    
    # Flash Attention
    "FlashAttentionLayer",
    "FlashMultiHeadAttentionBlock",
    
    # TFT Components
    "VariableSelectionNetwork",
    "StaticCovariateEncoder", 
    "TemporalSelfAttention",
    "QuantileForecaster",
    "TFTBlock",
    
    # Advanced Normalization
    "RMSNorm",
    "AdaptiveLayerNorm",
    "GroupNorm",
    "SpectralNorm",
    "LayerScale",
    
    # Positional Encodings
    "RotaryPositionalEmbedding",
    "RelativePositionalEmbedding",
    "LearnablePositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "PositionalEncodingBlock",
    
    # Advanced Integration
    "AdvancedEncoderBlock",
    "AdvancedDecoderBlock",
    "AdvancedSequenceToSequenceModel",
    "create_advanced_model_config"
]
