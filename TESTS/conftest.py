"""Pytest configuration and fixtures for pymodconn tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def test_config_dir():
    """Provide path to test configuration directory."""
    return Path(__file__).parent.parent / "TESTS"


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a minimal valid configuration for testing."""
    return {
        # Basic model parameters
        "if_model_image": False,
        "if_model_summary": False,
        "if_seed": True,
        "seed": 42,
        
        # Model architecture
        "model_type_prob": "nonprob",
        "loss_prob": "nonparametric",
        "loss": "mse",
        "quantiles": [0.1, 0.5, 0.9],
        "metrics": ["mae"],
        
        # Optimizer settings
        "optimizer": "Adam",
        "Adam": {
            "lr": 0.001,
            "b1": 0.9,
            "b2": 0.999,
            "epsi": 1e-8
        },
        "SGD": {
            "lr": 0.01,
            "momentum": 0.0
        },
        
        # Data dimensions
        "batch_size": 32,
        "n_past": 10,
        "n_future": 5,
        "known_past_features": 2,
        "unknown_future_features": 1,
        "known_future_features": 1,
        
        # Network parameters
        "all_layers_neurons": 64,
        "all_layers_dropout": 0.1,
        
        # File paths
        "save_models_dir": "test_models",
        
        # Encoder configuration
        "encoder": {
            "TCN_input": {
                "IF_TCN": True,
                "IF_NONE_GLUADDNORM_ADDNORM_TCN": 1,
                "kernel_size": 3,
                "nb_stacks": 1,
                "dilations": [1, 2]
            },
            "RNN_block_input": {
                "IF_RNN": True,
                "IF_NONE_GLUADDNORM_ADDNORM_block": 1,
                "IF_GRN_block": False,
                "rnn_depth": 1,
                "rnn_type": "LSTM",
                "IF_birectionalRNN": False,
                "IF_NONE_GLUADDNORM_ADDNORM_deep": 1
            },
            "self_MHA_block": {
                "IF_MHA": True,
                "MHA_depth": 1,
                "MHA_head": 4,
                "IF_NONE_GLUADDNORM_ADDNORM_deep": 1,
                "IF_GRN_block": False
            }
        },
        
        # Decoder configuration
        "decoder": {
            "CIT_option": 1,
            "option_1_depth": 1,
            "IF_NONE_GLUADDNORM_ADDNORM_CIT_1": 1,
            "MERGE_STATES_METHOD": 4,
            "TCN_input": {
                "IF_TCN": True,
                "IF_NONE_GLUADDNORM_ADDNORM_TCN": 1,
                "kernel_size": 3,
                "nb_stacks": 1,
                "dilations": [1, 2]
            },
            "TCN_output": {
                "IF_TCN": True,
                "IF_NONE_GLUADDNORM_ADDNORM_TCN": 1,
                "kernel_size": 3,
                "nb_stacks": 1,
                "dilations": [1, 2]
            },
            "RNN_block_input": {
                "IF_RNN": True,
                "IF_NONE_GLUADDNORM_ADDNORM_block": 1,
                "IF_GRN_block": False,
                "rnn_depth": 1,
                "rnn_type": "LSTM",
                "IF_birectionalRNN": False,
                "IF_NONE_GLUADDNORM_ADDNORM_deep": 1
            },
            "RNN_block_output": {
                "IF_RNN": True,
                "IF_NONE_GLUADDNORM_ADDNORM_block": 1,
                "IF_GRN_block": False,
                "rnn_depth": 1,
                "rnn_type": "LSTM",
                "IF_birectionalRNN": False,
                "IF_NONE_GLUADDNORM_ADDNORM_deep": 1
            }
        }
    }


@pytest.fixture
def probabilistic_config(sample_config) -> Dict[str, Any]:
    """Provide a probabilistic model configuration."""
    config = sample_config.copy()
    config.update({
        "model_type_prob": "prob",
        "loss_prob": "parametric",
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9]
    })
    return config


@pytest.fixture
def sample_data(sample_config):
    """Generate sample training data for testing."""
    np.random.seed(42)
    
    batch_size = 100
    n_past = sample_config["n_past"]
    n_future = sample_config["n_future"]
    known_past_features = sample_config["known_past_features"]
    known_future_features = sample_config["known_future_features"]
    unknown_future_features = sample_config["unknown_future_features"]
    
    # Generate synthetic time series data
    x_past = np.random.randn(batch_size, n_past, known_past_features)
    x_future = np.random.randn(batch_size, n_future, known_future_features)
    y_future = np.random.randn(batch_size, n_future, unknown_future_features)
    
    return {
        "x_past": x_past,
        "x_future": x_future,
        "y_future": y_future
    }


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(autouse=True)
def clear_tf_session():
    """Clear TensorFlow session before and after each test."""
    tf.keras.backend.clear_session()
    yield
    tf.keras.backend.clear_session()


@pytest.fixture
def mock_datetime():
    """Provide a mock datetime string for consistent testing."""
    return "test_20240103_120000"


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    return len(tf.config.list_physical_devices('GPU')) > 0


@pytest.fixture
def config_with_temp_dir(sample_config, temp_model_dir):
    """Provide config with temporary directory for model saving."""
    config = sample_config.copy()
    config["save_models_dir"] = temp_model_dir
    return config
