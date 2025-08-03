"""Integration tests for PyModConn - modernized versions of old test files."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from pymodconn import ModelGen
from pymodconn.config_manager import ConfigManager


@pytest.mark.integration
class TestModelCompilation:
    """Integration tests for model compilation - modernized TEST_COMPILE.py."""
    
    def test_tcn_model_compilation(self, sample_config, temp_model_dir):
        """Test TCN-based model compilation."""
        # Configure for TCN model
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["encoder"]["TCN_input"]["IF_TCN"] = True
        config["encoder"]["RNN_block_input"]["IF_RNN"] = False
        config["encoder"]["self_MHA_block"]["IF_MHA"] = False
        
        current_dt = "TCN_test_compile"
        
        # Clear any existing session
        tf.keras.backend.clear_session()
        
        # Build model
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Verify model was built
        assert model_gen.model is not None
        assert isinstance(model_gen.model, tf.keras.Model)
        
        # Verify model can make predictions
        batch_size = 4
        x_past = np.random.randn(batch_size, config["n_past"], config["known_past_features"])
        x_future = np.random.randn(batch_size, config["n_future"], config["known_future_features"])
        
        predictions = model_gen.model.predict([x_past, x_future], verbose=0)
        
        expected_shape = (batch_size, config["n_future"], config["unknown_future_features"])
        assert predictions.shape == expected_shape
        assert np.all(np.isfinite(predictions))
    
    def test_mha_model_compilation(self, sample_config, temp_model_dir):
        """Test Multi-Head Attention model compilation."""
        # Configure for MHA model
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["encoder"]["TCN_input"]["IF_TCN"] = False
        config["encoder"]["RNN_block_input"]["IF_RNN"] = False
        config["encoder"]["self_MHA_block"]["IF_MHA"] = True
        config["encoder"]["self_MHA_block"]["MHA_depth"] = 2
        
        current_dt = "MHA_test_compile"
        
        # Clear any existing session
        tf.keras.backend.clear_session()
        
        # Build model
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Verify model was built
        assert model_gen.model is not None
        assert isinstance(model_gen.model, tf.keras.Model)
        
        # Verify model structure
        assert len(model_gen.model.inputs) == 2  # encoder and decoder inputs
        assert len(model_gen.model.outputs) == 1  # single output
    
    def test_rnn_model_compilation(self, sample_config, temp_model_dir):
        """Test RNN (LSTM) model compilation."""
        # Configure for RNN model
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["encoder"]["TCN_input"]["IF_TCN"] = False
        config["encoder"]["RNN_block_input"]["IF_RNN"] = True
        config["encoder"]["RNN_block_input"]["rnn_type"] = "LSTM"
        config["encoder"]["RNN_block_input"]["IF_birectionalRNN"] = True
        config["encoder"]["self_MHA_block"]["IF_MHA"] = False
        
        current_dt = "RNN_test_compile"
        
        # Clear any existing session
        tf.keras.backend.clear_session()
        
        # Build model
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Verify model was built
        assert model_gen.model is not None
        assert isinstance(model_gen.model, tf.keras.Model)
        
        # Test model summary generation
        summary_file = Path(temp_model_dir) / f"{current_dt}_modelsummary.txt"
        assert summary_file.exists()
        
        # Verify summary file has content
        with open(summary_file, 'r') as f:
            summary_content = f.read()
        assert len(summary_content) > 0
        assert "Total params" in summary_content


@pytest.mark.integration
class TestModelTraining:
    """Integration tests for model training - modernized TEST_COMPILE_TRAIN.py."""
    
    def generate_synthetic_data(self, config, num_samples=200):
        """Generate synthetic time series data for training."""
        np.random.seed(42)  # For reproducible results
        
        n_past = config["n_past"]
        n_future = config["n_future"]
        known_past_features = config["known_past_features"]
        known_future_features = config["known_future_features"]
        unknown_future_features = config["unknown_future_features"]
        
        # Generate synthetic time series with some temporal patterns
        time_steps = np.arange(num_samples * (n_past + n_future))
        
        # Create base signals with different frequencies
        signal1 = np.sin(time_steps * 0.1) + 0.5 * np.sin(time_steps * 0.05)
        signal2 = np.cos(time_steps * 0.08) + 0.3 * np.cos(time_steps * 0.12)
        
        # Add noise
        signal1 += np.random.normal(0, 0.1, len(signal1))
        signal2 += np.random.normal(0, 0.1, len(signal2))
        
        # Create sequences
        x_past = []
        x_future = []
        y_future = []
        
        for i in range(num_samples):
            start_idx = i * (n_past + n_future)
            
            # Past data (known features)
            past_data = np.column_stack([
                signal1[start_idx:start_idx + n_past],
                signal2[start_idx:start_idx + n_past]
            ])[:, :known_past_features]
            
            # Future data (known features)
            future_data = np.column_stack([
                signal1[start_idx + n_past:start_idx + n_past + n_future],
                signal2[start_idx + n_past:start_idx + n_past + n_future]
            ])[:, :known_future_features]
            
            # Target data (unknown features to predict)
            target_data = np.column_stack([
                signal1[start_idx + n_past:start_idx + n_past + n_future],
                signal2[start_idx + n_past:start_idx + n_past + n_future]
            ])[:, :unknown_future_features]
            
            x_past.append(past_data)
            x_future.append(future_data)
            y_future.append(target_data)
        
        return {
            "x_past": np.array(x_past),
            "x_future": np.array(x_future),
            "y_future": np.array(y_future)
        }
    
    def test_lstm_model_training(self, sample_config, temp_model_dir):
        """Test complete training workflow with LSTM model."""
        # Configure for LSTM model
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["encoder"]["RNN_block_input"]["IF_RNN"] = True
        config["encoder"]["RNN_block_input"]["rnn_type"] = "LSTM"
        config["encoder"]["RNN_block_input"]["IF_birectionalRNN"] = True
        config["batch_size"] = 16
        
        # Generate synthetic data
        data = self.generate_synthetic_data(config, num_samples=100)
        
        # Split data
        split_idx = int(0.8 * len(data["x_past"]))
        x_train_past = data["x_past"][:split_idx]
        x_train_future = data["x_future"][:split_idx]
        y_train = data["y_future"][:split_idx]
        
        x_test_past = data["x_past"][split_idx:]
        x_test_future = data["x_future"][split_idx:]
        y_test = data["y_future"][split_idx:]
        
        # Build model
        current_dt = "LSTM_training_test"
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Train model
        history = model_gen.model.fit(
            [x_train_past, x_train_future],
            y_train,
            batch_size=config["batch_size"],
            epochs=3,
            validation_split=0.2,
            verbose=0
        )
        
        # Verify training worked
        assert "loss" in history.history
        assert len(history.history["loss"]) == 3
        assert all(np.isfinite(loss) for loss in history.history["loss"])
        
        # Evaluate model
        test_loss = model_gen.model.evaluate(
            [x_test_past, x_test_future], 
            y_test, 
            verbose=0
        )
        assert np.isfinite(test_loss)
        
        # Make predictions
        predictions = model_gen.model.predict(
            [x_test_past, x_test_future], 
            verbose=0
        )
        
        # Verify predictions
        assert predictions.shape == y_test.shape
        assert np.all(np.isfinite(predictions))
        
        # Save model
        model_path = Path(temp_model_dir) / f"{current_dt}_model.h5"
        model_gen.model.save(str(model_path))
        assert model_path.exists()
    
    def test_mha_model_training(self, sample_config, temp_model_dir):
        """Test complete training workflow with Multi-Head Attention model."""
        # Configure for MHA model
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["encoder"]["self_MHA_block"]["IF_MHA"] = True
        config["encoder"]["self_MHA_block"]["MHA_depth"] = 1
        config["encoder"]["self_MHA_block"]["MHA_head"] = 2
        config["batch_size"] = 8
        
        # Generate synthetic data
        data = self.generate_synthetic_data(config, num_samples=80)
        
        # Build model
        current_dt = "MHA_training_test"
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Train model (fewer epochs for faster testing)
        history = model_gen.model.fit(
            [data["x_past"], data["x_future"]],
            data["y_future"],
            batch_size=config["batch_size"],
            epochs=2,
            validation_split=0.2,
            verbose=0
        )
        
        # Verify training worked
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2
        assert all(np.isfinite(loss) for loss in history.history["loss"])
        
        # Test model can make predictions
        predictions = model_gen.model.predict(
            [data["x_past"][:10], data["x_future"][:10]], 
            verbose=0
        )
        
        assert predictions.shape == data["y_future"][:10].shape
        assert np.all(np.isfinite(predictions))
    
    def test_tcn_model_training(self, sample_config, temp_model_dir):
        """Test complete training workflow with TCN model."""
        # Configure for TCN model
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["encoder"]["TCN_input"]["IF_TCN"] = True
        config["encoder"]["TCN_input"]["kernel_size"] = 3
        config["encoder"]["TCN_input"]["nb_stacks"] = 1
        config["encoder"]["TCN_input"]["dilations"] = [1, 2]
        config["batch_size"] = 8
        
        # Generate synthetic data
        data = self.generate_synthetic_data(config, num_samples=80)
        
        # Build model
        current_dt = "TCN_training_test"
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Train model
        history = model_gen.model.fit(
            [data["x_past"], data["x_future"]],
            data["y_future"],
            batch_size=config["batch_size"],
            epochs=2,
            validation_split=0.2,
            verbose=0
        )
        
        # Verify training worked
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2
        
        # Test model performance didn't degrade significantly
        final_loss = history.history["loss"][-1]
        initial_loss = history.history["loss"][0]
        
        # Loss should either improve or stay reasonable
        assert final_loss < initial_loss * 2  # Allow some variance


@pytest.mark.integration
class TestModelPersistence:
    """Test model saving and loading functionality."""
    
    def test_model_save_and_load_weights(self, sample_config, temp_model_dir):
        """Test saving and loading model weights."""
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        
        # Build and train original model
        current_dt = "save_load_test"
        model_gen1 = ModelGen(config, current_dt)
        model_gen1.build_model()
        
        # Generate some data and train briefly
        data = np.random.randn(20, config["n_past"], config["known_past_features"])
        future_data = np.random.randn(20, config["n_future"], config["known_future_features"])
        target_data = np.random.randn(20, config["n_future"], config["unknown_future_features"])
        
        model_gen1.model.fit([data, future_data], target_data, epochs=1, verbose=0)
        
        # Save weights
        weights_path = Path(temp_model_dir) / "test_weights.h5"
        model_gen1.model.save_weights(str(weights_path))
        
        # Create new model and load weights
        model_gen2 = ModelGen(config, current_dt)
        model_gen2.build_model()
        model_gen2.load_model(str(weights_path))
        
        # Compare predictions (should be identical)
        test_data = data[:5]
        test_future = future_data[:5]
        
        pred1 = model_gen1.model.predict([test_data, test_future], verbose=0)
        pred2 = model_gen2.model.predict([test_data, test_future], verbose=0)
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)
    
    def test_model_artifacts_creation(self, sample_config, temp_model_dir):
        """Test that all expected model artifacts are created."""
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["if_model_image"] = True  # Enable model image saving
        config["if_model_summary"] = True
        
        current_dt = "artifacts_test"
        
        # Build model
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        save_dir = Path(temp_model_dir)
        
        # Check that all expected files are created
        expected_files = [
            f"{current_dt}_modelsummary.txt",
            f"{current_dt}_modelconfig.yaml"
        ]
        
        for filename in expected_files:
            file_path = save_dir / filename
            assert file_path.exists(), f"Expected file not found: {filename}"
            
            # Verify file has content
            assert file_path.stat().st_size > 0, f"File is empty: {filename}"
        
        # Verify config file contains expected content
        config_file = save_dir / f"{current_dt}_modelconfig.yaml"
        import yaml
        with open(config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config["n_past"] == config["n_past"]
        assert saved_config["n_future"] == config["n_future"]
        assert saved_config["optimizer"] == config["optimizer"]


@pytest.mark.integration
@pytest.mark.slow
class TestModelPerformance:
    """Performance and stress tests for models."""
    
    def test_model_memory_usage(self, sample_config, temp_model_dir):
        """Test model memory usage calculation."""
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["all_layers_neurons"] = 128  # Larger model for memory test
        
        current_dt = "memory_test"
        
        # Build model
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Check that memory usage was calculated and logged
        summary_file = Path(temp_model_dir) / f"{current_dt}_modelsummary.txt"
        assert summary_file.exists()
        
        with open(summary_file, 'r') as f:
            summary_content = f.read()
        
        assert "Model size in GB" in summary_content
        
        # Extract memory usage value
        import re
        memory_match = re.search(r"Model size in GB : ([\d.]+)", summary_content)
        assert memory_match is not None
        
        memory_gb = float(memory_match.group(1))
        assert memory_gb > 0
        assert memory_gb < 10  # Reasonable upper bound for test models
    
    def test_model_with_large_dimensions(self, sample_config, temp_model_dir):
        """Test model with larger dimensions."""
        config = sample_config.copy()
        config["save_models_dir"] = temp_model_dir
        config["n_past"] = 50
        config["n_future"] = 20
        config["all_layers_neurons"] = 256
        config["known_past_features"] = 5
        config["known_future_features"] = 3
        config["unknown_future_features"] = 2
        
        current_dt = "large_dims_test"
        
        # Build model
        model_gen = ModelGen(config, current_dt)
        model_gen.build_model()
        
        # Test with appropriately sized data
        batch_size = 4
        x_past = np.random.randn(batch_size, config["n_past"], config["known_past_features"])
        x_future = np.random.randn(batch_size, config["n_future"], config["known_future_features"])
        
        # Model should handle larger dimensions
        predictions = model_gen.model.predict([x_past, x_future], verbose=0)
        
        expected_shape = (batch_size, config["n_future"], config["unknown_future_features"])
        assert predictions.shape == expected_shape
        assert np.all(np.isfinite(predictions))
