"""Model generation module for PyModConn."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow import keras

import pymodconn.model_gen_utils as model_utils
from pymodconn.Decoder_class_layer import Decoder_class
from pymodconn.Encoder_class_layer import EncoderClass
from pymodconn.utils_layers import soft_relu

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
logger = logging.getLogger(__name__)


class ModelGen:
    """
    Main class for generating modular control-oriented neural networks.
    
    This class provides functionality to build sequence-to-sequence neural networks
    with configurable encoder-decoder architectures supporting various components
    like TCN, RNN, and Multi-Head Attention layers.
    
    Args:
        cfg: Configuration dictionary containing model parameters
        current_dt: Current datetime string for model identification
        
    Attributes:
        cfg: Configuration dictionary
        current_dt: Current datetime string
        model: The built Keras model (available after build_model() is called)
    """

    def __init__(self, cfg: Dict[str, Any], current_dt: str) -> None:
        """Initialize ModelGen with configuration and datetime."""
        self.cfg = cfg
        self.current_dt = current_dt

        if cfg["if_seed"]:
            np.random.seed(cfg["seed"])
            tf.random.set_seed(cfg["seed"])
            tf.keras.utils.set_random_seed(cfg["seed"])
            tf.config.experimental.enable_op_determinism()

        self.n_past: int = cfg["n_past"]
        self.n_future: int = cfg["n_future"]
        self.known_past_features: int = cfg["known_past_features"]
        self.unknown_future_features: int = cfg["unknown_future_features"]
        self.known_future_features: int = cfg["known_future_features"]

        self.all_layers_neurons: int = cfg["all_layers_neurons"]
        self.all_layers_dropout: float = cfg["all_layers_dropout"]

        self.model_type_prob: str = cfg["model_type_prob"]
        self.loss_prob: str = cfg["loss_prob"]
        self.q: List[float] = cfg["quantiles"]
        self.n_outputs_lastlayer: int = (
            2 if self.loss_prob == "parametric" else len(self.q)
        )

        self.save_models_dir: str = cfg["save_models_dir"]
        if not os.path.exists(cfg["save_models_dir"]):
            os.makedirs(cfg["save_models_dir"])

        self.model: Optional[Model] = None

    def build_model(self) -> None:
        """
        Build the neural network model according to configuration specifications.
        
        This method constructs a sequence-to-sequence neural network with configurable
        encoder-decoder architecture. The model supports various components including
        TCN, RNN, and Multi-Head Attention layers.
        
        The process involves:
        1. Creating encoder input layer and processing through EncoderClass
        2. Creating decoder input layer and processing through DecoderClass
        3. Applying probabilistic or non-probabilistic output transformations
        4. Compiling the model with appropriate loss functions and optimizers
        
        Raises:
            ValueError: If model_type_prob is not 'prob' or 'nonprob'
        """
        logger.info("Starting model compilation")
        timer = model_utils.Timer()
        timer.start()

        # Create encoder input layer
        encoder_inputs = tf.keras.layers.Input(
            shape=(self.n_past, self.known_past_features),
            name="encoder_past_inputs"
        )

        # Process through encoder
        encoder_outputs_seq, encoder_outputs_allstates = EncoderClass(
            self.cfg, "1"
        )(encoder_inputs, init_states=None)

        # Create decoder input layer
        decoder_inputs = tf.keras.layers.Input(
            shape=(self.n_future, self.known_future_features),
            name="decoder_inputs"
        )

        # Process through decoder
        decoder_outputs = Decoder_class(self.cfg, "1")(
            decoder_inputs, encoder_outputs_seq, encoder_states=encoder_outputs_allstates
        )

        # Apply output transformations based on model type
        if self.model_type_prob == "prob":
            # Probabilistic output: reshape for loss function requirements
            decoder_outputs3 = tf.keras.layers.Dense(
                units=self.unknown_future_features * self.n_outputs_lastlayer
            )(decoder_outputs)
            
            decoder_outputs4 = tf.keras.layers.Reshape(
                target_shape=(
                    self.n_future,
                    self.unknown_future_features,
                    self.n_outputs_lastlayer,
                )
            )(decoder_outputs3)
            
            # Apply soft ReLU for parametric loss to ensure positive std deviation
            if self.loss_prob == "parametric":
                decoder_outputs4 = tf.keras.layers.Lambda(
                    function=lambda x: tf.stack(
                        [x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1
                    )
                )(decoder_outputs4)
                
        elif self.model_type_prob == "nonprob":
            # Non-probabilistic output: clip values to [-1, 1]
            decoder_outputs4 = tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, -1, 1)
            )(decoder_outputs)
        else:
            raise ValueError(
                "model_type_prob should be either 'prob' or 'nonprob'"
            )

        # Create the final model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs4)

        # Post-process the model (compile, summary, etc.)
        model_utils.BuildUtils(self.cfg, self.current_dt).postbuild_model(self.model)

    def load_model(self, filepath: str) -> None:
        """
        Load model weights from a file.
        
        Args:
            filepath: Path to the model weights file
        """
        if self.model is None:
            raise ValueError("Model must be built before loading weights")
            
        logger.info("Loading model weights from file: %s", filepath)
        self.model.load_weights(filepath)

    def forget_model(self) -> None:
        """
        Delete the model and clear the Keras session.
        
        This method helps free up memory by deleting the model instance
        and clearing the Keras backend session.
        """
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        K.clear_session()
        logger.info("Model forgotten and Keras session cleared")
