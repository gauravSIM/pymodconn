"""Configuration management module for PyModConn."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from pymodconn.base_classes import ConfigValidator

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Enhanced configuration manager with validation and error handling.
    
    This class provides comprehensive configuration management including
    loading, validation, and error reporting with detailed messages.
    
    Args:
        config_path: Path to configuration file
        
    Attributes:
        config_path: Path to the configuration file
        config: Loaded and validated configuration dictionary
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        """Initialize ConfigManager with configuration file path."""
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load and validate configuration from file.
        
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            logger.info("Configuration loaded from: %s", self.config_path)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse YAML configuration file {self.config_path}: {e}"
            ) from e
            
        # Validate the loaded configuration
        self._validate_config()
        
        return self.config
    
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration.
        
        Raises:
            ValueError: If configuration validation fails
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")
            
        # Validate top-level required keys
        top_level_keys = [
            "if_model_image", "if_model_summary", "if_seed",
            "model_type_prob", "loss_prob", "loss", "quantiles",
            "metrics", "optimizer", "save_models_dir", "batch_size",
            "n_past", "n_future", "known_past_features",
            "unknown_future_features", "known_future_features",
            "all_layers_neurons", "all_layers_dropout",
            "encoder", "decoder"
        ]
        
        ConfigValidator.validate_required_keys(
            self.config, top_level_keys, "Configuration"
        )
        
        # Validate parameter types and ranges
        self._validate_basic_parameters()
        self._validate_encoder_config()
        self._validate_decoder_config()
        
        logger.info("Configuration validation completed successfully")
    
    def _validate_basic_parameters(self) -> None:
        """Validate basic configuration parameters."""
        # Validate boolean parameters
        bool_params = ["if_model_image", "if_model_summary", "if_seed"]
        for param in bool_params:
            if not isinstance(self.config[param], (bool, int)):
                raise ValueError(f"Parameter {param} must be boolean, got {type(self.config[param])}")
        
        # Validate positive integers
        int_params = [
            "batch_size", "n_past", "n_future", "known_past_features",
            "unknown_future_features", "known_future_features", "all_layers_neurons"
        ]
        for param in int_params:
            ConfigValidator.validate_positive_int(
                self.config[param], param, "Configuration"
            )
        
        # Validate dropout rate
        ConfigValidator.validate_float_range(
            self.config["all_layers_dropout"], "all_layers_dropout", "Configuration"
        )
        
        # Validate model type
        ConfigValidator.validate_choice(
            self.config["model_type_prob"], "model_type_prob", "Configuration",
            ["prob", "nonprob"]
        )
        
        # Validate loss type for probabilistic models
        if self.config["model_type_prob"] == "prob":
            ConfigValidator.validate_choice(
                self.config["loss_prob"], "loss_prob", "Configuration",
                ["parametric", "nonparametric"]
            )
        
        # Validate optimizer
        ConfigValidator.validate_choice(
            self.config["optimizer"], "optimizer", "Configuration",
            ["Adam", "SGD"]
        )
        
        # Validate quantiles
        quantiles = self.config["quantiles"]
        if not isinstance(quantiles, list) or not quantiles:
            raise ValueError("Quantiles must be a non-empty list")
            
        for i, q in enumerate(quantiles):
            if not isinstance(q, (int, float)) or not (0 < q < 1):
                raise ValueError(f"Quantile {i} must be between 0 and 1, got {q}")
    
    def _validate_encoder_config(self) -> None:
        """Validate encoder configuration."""
        encoder_config = self.config["encoder"]
        
        # Validate encoder components
        encoder_components = ["TCN_input", "RNN_block_input", "self_MHA_block"]
        for component in encoder_components:
            if component not in encoder_config:
                logger.warning("Encoder component %s not found in configuration", component)
                continue
                
            component_config = encoder_config[component]
            
            if component == "self_MHA_block":
                ConfigValidator.validate_required_keys(
                    component_config, ["MHA_depth"], f"encoder.{component}"
                )
                ConfigValidator.validate_positive_int(
                    component_config["MHA_depth"], "MHA_depth", f"encoder.{component}"
                )
    
    def _validate_decoder_config(self) -> None:
        """Validate decoder configuration."""
        decoder_config = self.config["decoder"]
        
        # Validate CIT option
        if "CIT_option" in decoder_config:
            ConfigValidator.validate_choice(
                decoder_config["CIT_option"], "CIT_option", "decoder",
                [1, 2, 3]
            )
        
        # Validate merge states method
        if "MERGE_STATES_METHOD" in decoder_config:
            ConfigValidator.validate_choice(
                decoder_config["MERGE_STATES_METHOD"], "MERGE_STATES_METHOD", "decoder",
                list(range(1, 9))  # Methods 1-8
            )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration.
        
        Returns:
            Configuration dictionary
            
        Raises:
            RuntimeError: If configuration hasn't been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self.config
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to a file.
        
        Args:
            output_path: Path where to save the configuration
            
        Raises:
            RuntimeError: If configuration hasn't been loaded
            IOError: If file writing fails
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
            
        output_path = Path(output_path)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
                
            logger.info("Configuration saved to: %s", output_path)
            
        except IOError as e:
            raise IOError(f"Failed to save configuration to {output_path}: {e}") from e
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Raises:
            RuntimeError: If configuration hasn't been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
            
        # Deep update the configuration
        self._deep_update(self.config, updates)
        
        # Re-validate after updates
        self._validate_config()
        
        logger.info("Configuration updated with %d changes", len(updates))
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionaries.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Updates to apply
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


def load_default_config() -> Dict[str, Any]:
    """
    Load the default configuration.
    
    Returns:
        Default configuration dictionary
        
    Raises:
        FileNotFoundError: If default config file not found
    """
    default_config_path = Path(__file__).parent / "configs" / "default_config.yaml"
    
    config_manager = ConfigManager(default_config_path)
    return config_manager.load_config()


def create_config_from_template(
    template_name: str, 
    output_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a configuration file from a template.
    
    Args:
        template_name: Name of the template configuration
        output_path: Path where to save the new configuration
        overrides: Optional configuration overrides
        
    Returns:
        Created configuration dictionary
        
    Raises:
        FileNotFoundError: If template not found
    """
    template_path = Path(__file__).parent / "configs" / f"{template_name}.yaml"
    
    config_manager = ConfigManager(template_path)
    config = config_manager.load_config()
    
    if overrides:
        config_manager.update_config(overrides)
        config = config_manager.get_config()
    
    config_manager.save_config(output_path)
    
    logger.info("Configuration created from template %s at %s", template_name, output_path)
    
    return config
