"""Tests for configuration management functionality."""

import tempfile
from pathlib import Path

import pytest
import yaml

from pymodconn.config_manager import ConfigManager, load_default_config, create_config_from_template
from pymodconn.base_classes import ConfigValidator


class TestConfigValidator:
    """Test ConfigValidator utility class."""
    
    def test_validate_required_keys_success(self):
        """Test successful validation of required keys."""
        config = {"key1": "value1", "key2": "value2", "key3": "value3"}
        required_keys = ["key1", "key2"]
        
        # Should not raise any exception
        ConfigValidator.validate_required_keys(config, required_keys, "TestComponent")
    
    def test_validate_required_keys_missing(self):
        """Test validation failure with missing keys."""
        config = {"key1": "value1"}
        required_keys = ["key1", "key2", "key3"]
        
        with pytest.raises(ValueError, match="TestComponent missing required configuration keys: \\['key2', 'key3'\\]"):
            ConfigValidator.validate_required_keys(config, required_keys, "TestComponent")
    
    def test_validate_positive_int_success(self):
        """Test successful positive integer validation."""
        ConfigValidator.validate_positive_int(5, "test_param", "TestComponent")
        ConfigValidator.validate_positive_int(1, "test_param", "TestComponent")
        ConfigValidator.validate_positive_int(100, "test_param", "TestComponent")
    
    def test_validate_positive_int_failure(self):
        """Test positive integer validation failures."""
        with pytest.raises(ValueError, match="TestComponent.test_param must be a positive integer, got 0"):
            ConfigValidator.validate_positive_int(0, "test_param", "TestComponent")
        
        with pytest.raises(ValueError, match="TestComponent.test_param must be a positive integer, got -5"):
            ConfigValidator.validate_positive_int(-5, "test_param", "TestComponent")
        
        with pytest.raises(ValueError, match="TestComponent.test_param must be a positive integer, got 3.14"):
            ConfigValidator.validate_positive_int(3.14, "test_param", "TestComponent")
    
    def test_validate_float_range_success(self):
        """Test successful float range validation."""
        ConfigValidator.validate_float_range(0.5, "test_param", "TestComponent", 0.0, 1.0)
        ConfigValidator.validate_float_range(0.0, "test_param", "TestComponent", 0.0, 1.0)
        ConfigValidator.validate_float_range(1.0, "test_param", "TestComponent", 0.0, 1.0)
    
    def test_validate_float_range_failure(self):
        """Test float range validation failures."""
        with pytest.raises(ValueError, match="TestComponent.test_param must be between 0.0 and 1.0, got 1.5"):
            ConfigValidator.validate_float_range(1.5, "test_param", "TestComponent", 0.0, 1.0)
        
        with pytest.raises(ValueError, match="TestComponent.test_param must be between 0.0 and 1.0, got -0.1"):
            ConfigValidator.validate_float_range(-0.1, "test_param", "TestComponent", 0.0, 1.0)
    
    def test_validate_choice_success(self):
        """Test successful choice validation."""
        ConfigValidator.validate_choice("option1", "test_param", "TestComponent", ["option1", "option2", "option3"])
        ConfigValidator.validate_choice(2, "test_param", "TestComponent", [1, 2, 3])
    
    def test_validate_choice_failure(self):
        """Test choice validation failures."""
        with pytest.raises(ValueError, match="TestComponent.test_param must be one of \\['option1', 'option2'\\], got invalid"):
            ConfigValidator.validate_choice("invalid", "test_param", "TestComponent", ["option1", "option2"])


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_init(self, temp_model_dir):
        """Test ConfigManager initialization."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        config_manager = ConfigManager(config_path)
        
        assert config_manager.config_path == config_path
        assert config_manager.config is None
    
    def test_load_config_success(self, sample_config, temp_model_dir):
        """Test successful configuration loading."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        # Save sample config to file
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        config_manager = ConfigManager(config_path)
        loaded_config = config_manager.load_config()
        
        assert loaded_config == sample_config
        assert config_manager.config == sample_config
    
    def test_load_config_file_not_found(self, temp_model_dir):
        """Test loading non-existent configuration file."""
        config_path = Path(temp_model_dir) / "nonexistent.yaml"
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            config_manager.load_config()
    
    def test_load_config_invalid_yaml(self, temp_model_dir):
        """Test loading invalid YAML configuration."""
        config_path = Path(temp_model_dir) / "invalid.yaml"
        
        # Write invalid YAML
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(yaml.YAMLError, match="Failed to parse YAML configuration file"):
            config_manager.load_config()
    
    def test_validate_config_missing_keys(self, temp_model_dir):
        """Test configuration validation with missing keys."""
        config_path = Path(temp_model_dir) / "incomplete_config.yaml"
        incomplete_config = {"n_past": 10}  # Missing many required keys
        
        with open(config_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="Configuration missing required configuration keys"):
            config_manager.load_config()
    
    def test_validate_config_invalid_values(self, sample_config, temp_model_dir):
        """Test configuration validation with invalid values."""
        config_path = Path(temp_model_dir) / "invalid_config.yaml"
        
        # Create config with invalid values
        invalid_config = sample_config.copy()
        invalid_config["n_past"] = -5  # Invalid negative value
        
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="Configuration.n_past must be a positive integer"):
            config_manager.load_config()
    
    def test_validate_config_invalid_quantiles(self, sample_config, temp_model_dir):
        """Test configuration validation with invalid quantiles."""
        config_path = Path(temp_model_dir) / "invalid_quantiles.yaml"
        
        # Create config with invalid quantiles
        invalid_config = sample_config.copy()
        invalid_config["quantiles"] = [0.1, 1.5, 0.9]  # 1.5 is invalid
        
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="Quantile 1 must be between 0 and 1"):
            config_manager.load_config()
    
    def test_get_config_success(self, sample_config, temp_model_dir):
        """Test getting loaded configuration."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        config_manager = ConfigManager(config_path)
        config_manager.load_config()
        
        retrieved_config = config_manager.get_config()
        assert retrieved_config == sample_config
    
    def test_get_config_not_loaded(self, temp_model_dir):
        """Test getting configuration before loading."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            config_manager.get_config()
    
    def test_save_config(self, sample_config, temp_model_dir):
        """Test saving configuration to file."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        output_path = Path(temp_model_dir) / "saved_config.yaml"
        
        # Load config first
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        config_manager = ConfigManager(config_path)
        config_manager.load_config()
        
        # Save to new location
        config_manager.save_config(output_path)
        
        # Verify saved file
        assert output_path.exists()
        with open(output_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config == sample_config
    
    def test_save_config_not_loaded(self, temp_model_dir):
        """Test saving configuration before loading."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        output_path = Path(temp_model_dir) / "output.yaml"
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            config_manager.save_config(output_path)
    
    def test_update_config(self, sample_config, temp_model_dir):
        """Test updating configuration."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        config_manager = ConfigManager(config_path)
        config_manager.load_config()
        
        # Update configuration
        updates = {
            "n_past": 20,
            "encoder": {
                "self_MHA_block": {
                    "MHA_depth": 3
                }
            }
        }
        
        config_manager.update_config(updates)
        updated_config = config_manager.get_config()
        
        assert updated_config["n_past"] == 20
        assert updated_config["encoder"]["self_MHA_block"]["MHA_depth"] == 3
        # Other values should remain unchanged
        assert updated_config["n_future"] == sample_config["n_future"]
    
    def test_update_config_invalid_values(self, sample_config, temp_model_dir):
        """Test updating configuration with invalid values."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        config_manager = ConfigManager(config_path)
        config_manager.load_config()
        
        # Try to update with invalid values
        invalid_updates = {"n_past": -10}
        
        with pytest.raises(ValueError, match="Configuration.n_past must be a positive integer"):
            config_manager.update_config(invalid_updates)
    
    def test_update_config_not_loaded(self, temp_model_dir):
        """Test updating configuration before loading."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            config_manager.update_config({"n_past": 20})


class TestConfigManagerValidation:
    """Test specific validation scenarios in ConfigManager."""
    
    def test_validate_encoder_config(self, sample_config, temp_model_dir):
        """Test encoder configuration validation."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        # Create config with missing encoder MHA depth
        invalid_config = sample_config.copy()
        del invalid_config["encoder"]["self_MHA_block"]["MHA_depth"]
        
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="encoder.self_MHA_block missing required configuration keys"):
            config_manager.load_config()
    
    def test_validate_decoder_config(self, sample_config, temp_model_dir):
        """Test decoder configuration validation."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        # Create config with invalid CIT option
        invalid_config = sample_config.copy()
        invalid_config["decoder"]["CIT_option"] = 5  # Invalid option
        
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="decoder.CIT_option must be one of \\[1, 2, 3\\]"):
            config_manager.load_config()
    
    def test_validate_optimizer_config(self, sample_config, temp_model_dir):
        """Test optimizer configuration validation."""
        config_path = Path(temp_model_dir) / "test_config.yaml"
        
        # Create config with invalid optimizer
        invalid_config = sample_config.copy()
        invalid_config["optimizer"] = "InvalidOptimizer"
        
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        with pytest.raises(ValueError, match="Configuration.optimizer must be one of \\['Adam', 'SGD'\\]"):
            config_manager.load_config()


@pytest.mark.integration
class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""
    
    def test_complete_config_workflow(self, sample_config, temp_model_dir):
        """Test complete configuration management workflow."""
        config_path = Path(temp_model_dir) / "original_config.yaml"
        updated_path = Path(temp_model_dir) / "updated_config.yaml"
        
        # Save original config
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Load, update, and save
        config_manager = ConfigManager(config_path)
        config_manager.load_config()
        
        updates = {
            "n_past": 15,
            "all_layers_neurons": 128,
            "encoder": {
                "self_MHA_block": {
                    "MHA_head": 8
                }
            }
        }
        
        config_manager.update_config(updates)
        config_manager.save_config(updated_path)
        
        # Load updated config with new manager
        new_manager = ConfigManager(updated_path)
        final_config = new_manager.load_config()
        
        # Verify updates
        assert final_config["n_past"] == 15
        assert final_config["all_layers_neurons"] == 128
        assert final_config["encoder"]["self_MHA_block"]["MHA_head"] == 8
        
        # Verify unchanged values
        assert final_config["n_future"] == sample_config["n_future"]
        assert final_config["optimizer"] == sample_config["optimizer"]


class TestConfigUtilityFunctions:
    """Test utility functions for configuration management."""
    
    @pytest.mark.skip(reason="Requires default config file to exist")
    def test_load_default_config(self):
        """Test loading default configuration."""
        # This test is skipped as it requires the actual default config file
        # In a real scenario, you would test this with the actual default config
        pass
    
    @pytest.mark.skip(reason="Requires template config files to exist")
    def test_create_config_from_template(self, temp_model_dir):
        """Test creating configuration from template."""
        # This test is skipped as it requires actual template files
        # In a real scenario, you would test this with actual templates
        pass
