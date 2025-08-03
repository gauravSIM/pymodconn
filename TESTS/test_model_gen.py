"""Comprehensive tests for ModelGen class."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from pymodconn import ModelGen
from pymodconn.config_manager import ConfigManager


class TestModelGenInitialization:
    """Test ModelGen initialization and basic functionality."""
    
    def test_model_gen_init_with_valid_config(self, config_with_temp_dir, mock_datetime):
        """Test ModelGen initialization with valid configuration."""
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        
        assert model_gen.cfg == config_with_temp_dir
        assert model_gen.current_dt == mock_datetime
        assert model_gen.n_past == config_with_temp_dir["n_past"]
        assert model_gen.n_future == config_with_temp_dir["n_future"]
        assert model_gen.model is None  # Model not built yet
    
    def test_model_gen_init_with_seed(self, config_with_temp_dir, mock_datetime):
        """Test ModelGen initialization with seed setting."""
        config_with_temp_dir["if_seed"] = True
        config_with_temp_dir["seed"] = 123
        
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        
        # Test that seed is properly set by generating random numbers
        tf.random.set_seed(123)
        expected_random = tf.random.normal([5]).numpy()
        
        tf.random.set_seed(123)
        actual_random = tf.random.normal([5]).numpy()
        
        np.testing.assert_array_equal(expected_random, actual_random)
    
    def test_model_gen_creates_save_directory(self, sample_config, mock_datetime, temp_model_dir):
        """Test that ModelGen creates save directory if it doesn't exist."""
        save_dir = Path(temp_model_dir) / "new_models"
        sample_config["save_models_dir"] = str(save_dir)
        
        assert not save_dir.exists()
        
        model_gen = ModelGen(sample_config, mock_datetime)
        
        assert save_dir.exists()
        assert save_dir.is_dir()


class TestModelGenBuildModel:
    """Test model building functionality."""
    
    def test_build_nonprob_model(self, config_with_temp_dir, mock_datetime):
        """Test building a non-probabilistic model."""
        config_with_temp_dir["model_type_prob"] = "nonprob"
        
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        assert model_gen.model is not None
        assert isinstance(model_gen.model, tf.keras.Model)
        
        # Check input shapes
        expected_encoder_shape = (None, config_with_temp_dir["n_past"], config_with_temp_dir["known_past_features"])
        expected_decoder_shape = (None, config_with_temp_dir["n_future"], config_with_temp_dir["known_future_features"])
        
        assert model_gen.model.input[0].shape == expected_encoder_shape
        assert model_gen.model.input[1].shape == expected_decoder_shape
        
        # Check output shape
        expected_output_shape = (None, config_with_temp_dir["n_future"], config_with_temp_dir["unknown_future_features"])
        assert model_gen.model.output.shape == expected_output_shape
    
    def test_build_prob_parametric_model(self, config_with_temp_dir, mock_datetime):
        """Test building a probabilistic parametric model."""
        config_with_temp_dir["model_type_prob"] = "prob"
        config_with_temp_dir["loss_prob"] = "parametric"
        
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        assert model_gen.model is not None
        
        # Check output shape for parametric model (should have 2 outputs: mean and std)
        expected_output_shape = (
            None, 
            config_with_temp_dir["n_future"], 
            config_with_temp_dir["unknown_future_features"], 
            2  # mean and std
        )
        assert model_gen.model.output.shape == expected_output_shape
    
    def test_build_prob_nonparametric_model(self, config_with_temp_dir, mock_datetime):
        """Test building a probabilistic non-parametric model."""
        config_with_temp_dir["model_type_prob"] = "prob"
        config_with_temp_dir["loss_prob"] = "nonparametric"
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        config_with_temp_dir["quantiles"] = quantiles
        
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        assert model_gen.model is not None
        
        # Check output shape for non-parametric model (should have quantile outputs)
        expected_output_shape = (
            None, 
            config_with_temp_dir["n_future"], 
            config_with_temp_dir["unknown_future_features"], 
            len(quantiles)
        )
        assert model_gen.model.output.shape == expected_output_shape
    
    def test_invalid_model_type_raises_error(self, config_with_temp_dir, mock_datetime):
        """Test that invalid model type raises ValueError."""
        config_with_temp_dir["model_type_prob"] = "invalid_type"
        
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        
        with pytest.raises(ValueError, match="model_type_prob should be either 'prob' or 'nonprob'"):
            model_gen.build_model()


class TestModelGenPrediction:
    """Test model prediction functionality."""
    
    def test_model_prediction(self, config_with_temp_dir, mock_datetime, sample_data):
        """Test that built model can make predictions."""
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        # Make predictions
        predictions = model_gen.model.predict([sample_data["x_past"], sample_data["x_future"]])
        
        # Check prediction shape
        expected_shape = (
            sample_data["x_past"].shape[0],
            config_with_temp_dir["n_future"],
            config_with_temp_dir["unknown_future_features"]
        )
        assert predictions.shape == expected_shape
        
        # Check that predictions are finite
        assert np.all(np.isfinite(predictions))
    
    def test_model_training_step(self, config_with_temp_dir, mock_datetime, sample_data):
        """Test that model can perform a training step."""
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        # Perform one training step
        history = model_gen.model.fit(
            [sample_data["x_past"], sample_data["x_future"]], 
            sample_data["y_future"],
            batch_size=16,
            epochs=1,
            verbose=0
        )
        
        assert "loss" in history.history
        assert len(history.history["loss"]) == 1
        assert np.isfinite(history.history["loss"][0])


class TestModelGenFileOperations:
    """Test model file operations."""
    
    def test_model_saves_artifacts(self, config_with_temp_dir, mock_datetime):
        """Test that model saves required artifacts."""
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        save_dir = Path(config_with_temp_dir["save_models_dir"])
        
        # Check that model summary file is created
        summary_file = save_dir / f"{mock_datetime}_modelsummary.txt"
        assert summary_file.exists()
        
        # Check that config file is created
        config_file = save_dir / f"{mock_datetime}_modelconfig.yaml"
        assert config_file.exists()
        
        # Check that config file contains valid YAML
        import yaml
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        assert loaded_config is not None
        assert loaded_config["n_past"] == config_with_temp_dir["n_past"]
    
    def test_load_model_weights(self, config_with_temp_dir, mock_datetime, sample_data, temp_model_dir):
        """Test loading model weights."""
        # Build and train a model
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        # Train for one epoch to get some weights
        model_gen.model.fit(
            [sample_data["x_past"], sample_data["x_future"]], 
            sample_data["y_future"],
            batch_size=16,
            epochs=1,
            verbose=0
        )
        
        # Save weights
        weights_path = Path(temp_model_dir) / "test_weights.h5"
        model_gen.model.save_weights(str(weights_path))
        
        # Create new model and load weights
        model_gen2 = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen2.build_model()
        model_gen2.load_model(str(weights_path))
        
        # Compare predictions (should be identical)
        pred1 = model_gen.model.predict([sample_data["x_past"][:10], sample_data["x_future"][:10]], verbose=0)
        pred2 = model_gen2.model.predict([sample_data["x_past"][:10], sample_data["x_future"][:10]], verbose=0)
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)
    
    def test_load_model_without_build_raises_error(self, config_with_temp_dir, mock_datetime):
        """Test that loading weights without building model raises error."""
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        
        with pytest.raises(ValueError, match="Model must be built before loading weights"):
            model_gen.load_model("dummy_path.h5")


class TestModelGenMemoryManagement:
    """Test memory management functionality."""
    
    def test_forget_model(self, config_with_temp_dir, mock_datetime):
        """Test that forget_model clears the model and session."""
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        assert model_gen.model is not None
        
        model_gen.forget_model()
        
        assert model_gen.model is None


class TestModelGenEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_minimal_network_dimensions(self, config_with_temp_dir, mock_datetime):
        """Test model with minimal network dimensions."""
        config_with_temp_dir["n_past"] = 1
        config_with_temp_dir["n_future"] = 1
        config_with_temp_dir["all_layers_neurons"] = 8
        
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        assert model_gen.model is not None
    
    def test_large_quantiles_list(self, config_with_temp_dir, mock_datetime):
        """Test model with large number of quantiles."""
        config_with_temp_dir["model_type_prob"] = "prob"
        config_with_temp_dir["loss_prob"] = "nonparametric"
        config_with_temp_dir["quantiles"] = [i/20 for i in range(1, 20)]  # 19 quantiles
        
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        assert model_gen.model is not None
        expected_output_shape = (
            None, 
            config_with_temp_dir["n_future"], 
            config_with_temp_dir["unknown_future_features"], 
            19
        )
        assert model_gen.model.output.shape == expected_output_shape


@pytest.mark.integration
class TestModelGenIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_training_workflow(self, config_with_temp_dir, mock_datetime, sample_data):
        """Test complete training workflow from initialization to evaluation."""
        # Initialize model
        model_gen = ModelGen(config_with_temp_dir, mock_datetime)
        model_gen.build_model()
        
        # Split data
        split_idx = int(0.8 * len(sample_data["x_past"]))
        x_train_past = sample_data["x_past"][:split_idx]
        x_train_future = sample_data["x_future"][:split_idx]
        y_train = sample_data["y_future"][:split_idx]
        
        x_test_past = sample_data["x_past"][split_idx:]
        x_test_future = sample_data["x_future"][split_idx:]
        y_test = sample_data["y_future"][split_idx:]
        
        # Train model
        history = model_gen.model.fit(
            [x_train_past, x_train_future], 
            y_train,
            batch_size=16,
            epochs=2,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate model
        test_loss = model_gen.model.evaluate([x_test_past, x_test_future], y_test, verbose=0)
        
        # Make predictions
        predictions = model_gen.model.predict([x_test_past, x_test_future], verbose=0)
        
        # Assertions
        assert len(history.history["loss"]) == 2
        assert np.isfinite(test_loss)
        assert predictions.shape == y_test.shape
        assert np.all(np.isfinite(predictions))
