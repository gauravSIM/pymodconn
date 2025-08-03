"""Encoder class module for PyModConn."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from pymodconn.base_classes import BaseEncoder, ConfigValidator
from pymodconn.MHA_block_class_function import MHA_block_class
from pymodconn.RNN_block_class_function import RNN_block_class
from pymodconn.TCN_addnorm_class_function import TCN_addnorm_class
from pymodconn.utils_layers import positional_encoding

K = tf.keras.backend
logger = logging.getLogger(__name__)


class EncoderClass(BaseEncoder):
    """
    Encoder class for sequence-to-sequence neural networks.
    
    This class implements the encoder component of the neural network architecture,
    supporting TCN (Temporal Convolutional Network), RNN, and Multi-Head Attention
    layers with configurable parameters.
    
    Args:
        cfg: Configuration dictionary containing encoder parameters
        enc_or_dec_number: Encoder/decoder identifier string
        
    Attributes:
        cfg: Configuration dictionary
        enc_or_dec_number: Encoder/decoder identifier
        n_past: Number of past time steps
        n_future: Number of future time steps
        known_past_features: Number of known past features
        unknown_future_features: Number of unknown future features
        known_future_features: Number of known future features
        all_layers_neurons: Number of neurons in all layers
        all_layers_dropout: Dropout rate for all layers
    """

    def _validate_config(self) -> None:
        """Validate configuration parameters specific to encoder type."""
        required_keys = [
            "n_past", "n_future", "known_past_features", 
            "unknown_future_features", "known_future_features",
            "all_layers_neurons", "all_layers_dropout", "encoder"
        ]
        
        ConfigValidator.validate_required_keys(
            self.cfg, required_keys, "EncoderClass"
        )
        
        # Validate encoder-specific configuration
        encoder_config = self.cfg["encoder"]
        encoder_required_keys = ["self_MHA_block"]
        ConfigValidator.validate_required_keys(
            encoder_config, encoder_required_keys, "EncoderClass.encoder"
        )
        
        # Validate MHA configuration
        mha_config = encoder_config["self_MHA_block"]
        mha_required_keys = ["MHA_depth"]
        ConfigValidator.validate_required_keys(
            mha_config, mha_required_keys, "EncoderClass.encoder.self_MHA_block"
        )
        
        # Validate parameter ranges
        ConfigValidator.validate_positive_int(
            self.cfg["n_past"], "n_past", "EncoderClass"
        )
        ConfigValidator.validate_positive_int(
            self.cfg["n_future"], "n_future", "EncoderClass"
        )
        ConfigValidator.validate_positive_int(
            self.cfg["all_layers_neurons"], "all_layers_neurons", "EncoderClass"
        )
        ConfigValidator.validate_float_range(
            self.cfg["all_layers_dropout"], "all_layers_dropout", "EncoderClass"
        )

    def _setup_parameters(self) -> None:
        """Set up encoder-specific parameters from configuration."""
        self.n_past: int = self.cfg["n_past"]
        self.n_future: int = self.cfg["n_future"]
        self.known_past_features: int = self.cfg["known_past_features"]
        self.unknown_future_features: int = self.cfg["unknown_future_features"]
        self.known_future_features: int = self.cfg["known_future_features"]

        self.all_layers_neurons: int = self.cfg["all_layers_neurons"]
        self.all_layers_dropout: float = self.cfg["all_layers_dropout"]
        
        logger.debug(
            "EncoderClass initialized with %d neurons, %.3f dropout",
            self.all_layers_neurons, self.all_layers_dropout
        )

    def __call__(
        self, 
        input_tensor: tf.Tensor, 
        init_states: Optional[List[tf.Tensor]] = None
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Process input through encoder layers.
        
        Args:
            input_tensor: Input tensor with shape (batch_size, n_past, features)
            init_states: Optional initial states for RNN layers
            
        Returns:
            Tuple containing:
                - output: Processed sequence tensor
                - output_states: Final states from RNN layers
        """
        # Initial dense layer to standardize feature dimensions
        encoder_input = tf.keras.layers.Dense(self.all_layers_neurons)(input_tensor)

        # Light dropout layer
        encoder_input = tf.keras.layers.Dropout(self.all_layers_dropout / 5)(
            encoder_input
        )
        output_cell = encoder_input

        # TCN layer with additive normalization and GLU
        input_cell = output_cell
        output_cell = TCN_addnorm_class(
            self.cfg, "encoder", "input", self.enc_or_dec_number
        )(input_cell)

        # RNN layer with additive normalization and GLU
        input_cell = output_cell
        output_cell, output_states = RNN_block_class(
            self.cfg, "encoder", "input", self.enc_or_dec_number
        )(input_cell, init_states=init_states)

        # Multi-Head Attention layers with additive normalization and GLU
        input_cell = output_cell
        mha_depth = self.cfg["encoder"]["self_MHA_block"]["MHA_depth"]
        
        for i in range(mha_depth):
            output_cell = MHA_block_class(
                self.cfg,
                "encoder",
                self.enc_or_dec_number,
                "self",
                str(i + 1),
            )(input_cell, input_cell)

            input_cell = output_cell

        return output_cell, output_states
