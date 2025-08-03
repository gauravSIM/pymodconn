"""Abstract base classes for PyModConn architecture components."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseEncoder(ABC):
    """
    Abstract base class for encoder components.
    
    This class defines the interface that all encoder implementations
    must follow, ensuring consistency across different encoder types.
    
    Args:
        cfg: Configuration dictionary containing encoder parameters
        enc_or_dec_number: Encoder/decoder identifier string
    """

    def __init__(self, cfg: Dict[str, Any], enc_or_dec_number: str) -> None:
        """Initialize base encoder with configuration validation."""
        self.cfg = cfg
        self.enc_or_dec_number = enc_or_dec_number
        self._validate_config()
        self._setup_parameters()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration parameters specific to encoder type."""
        pass

    @abstractmethod
    def _setup_parameters(self) -> None:
        """Set up encoder-specific parameters from configuration."""
        pass

    @abstractmethod
    def __call__(
        self, 
        input_tensor: tf.Tensor, 
        init_states: Optional[List[tf.Tensor]] = None
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Process input through encoder layers.
        
        Args:
            input_tensor: Input tensor with shape (batch_size, sequence_length, features)
            init_states: Optional initial states for RNN layers
            
        Returns:
            Tuple containing:
                - output: Processed sequence tensor
                - output_states: Final states from RNN layers
        """
        pass


class BaseDecoder(ABC):
    """
    Abstract base class for decoder components.
    
    This class defines the interface that all decoder implementations
    must follow, ensuring consistency across different decoder types.
    
    Args:
        cfg: Configuration dictionary containing decoder parameters
        enc_or_dec_number: Encoder/decoder identifier string
    """

    def __init__(self, cfg: Dict[str, Any], enc_or_dec_number: str) -> None:
        """Initialize base decoder with configuration validation."""
        self.cfg = cfg
        self.enc_or_dec_number = enc_or_dec_number
        self._validate_config()
        self._setup_parameters()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration parameters specific to decoder type."""
        pass

    @abstractmethod
    def _setup_parameters(self) -> None:
        """Set up decoder-specific parameters from configuration."""
        pass

    @abstractmethod
    def __call__(
        self,
        input_tensor: tf.Tensor,
        encoder_outputs: tf.Tensor,
        encoder_states: Optional[List[tf.Tensor]] = None
    ) -> tf.Tensor:
        """
        Process input through decoder layers.
        
        Args:
            input_tensor: Input tensor with shape (batch_size, sequence_length, features)
            encoder_outputs: Encoder output sequence
            encoder_states: Optional encoder final states
            
        Returns:
            Processed decoder output tensor
        """
        pass


class BaseBlock(ABC):
    """
    Abstract base class for neural network blocks.
    
    This class defines the interface for reusable neural network
    components like attention blocks, RNN blocks, etc.
    
    Args:
        cfg: Configuration dictionary containing block parameters
        block_type: Type identifier for the block
        block_number: Block identifier string
    """

    def __init__(
        self, 
        cfg: Dict[str, Any], 
        block_type: str, 
        block_number: str
    ) -> None:
        """Initialize base block with configuration validation."""
        self.cfg = cfg
        self.block_type = block_type
        self.block_number = block_number
        self._validate_config()
        self._setup_parameters()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration parameters specific to block type."""
        pass

    @abstractmethod
    def _setup_parameters(self) -> None:
        """Set up block-specific parameters from configuration."""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tf.Tensor:
        """
        Process input through block layers.
        
        Args and returns depend on specific block implementation.
        """
        pass


class ConfigValidator:
    """
    Utility class for configuration validation.
    
    This class provides common validation methods that can be used
    across different components to ensure configuration consistency.
    """

    @staticmethod
    def validate_required_keys(
        config: Dict[str, Any], 
        required_keys: List[str], 
        component_name: str
    ) -> None:
        """
        Validate that all required keys are present in configuration.
        
        Args:
            config: Configuration dictionary to validate
            required_keys: List of required key names
            component_name: Name of component for error messages
            
        Raises:
            ValueError: If any required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(
                f"{component_name} missing required configuration keys: {missing_keys}"
            )

    @staticmethod
    def validate_positive_int(
        value: Any, 
        param_name: str, 
        component_name: str
    ) -> None:
        """
        Validate that a parameter is a positive integer.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            component_name: Component name for error messages
            
        Raises:
            ValueError: If value is not a positive integer
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{component_name}.{param_name} must be a positive integer, got {value}"
            )

    @staticmethod
    def validate_float_range(
        value: Any, 
        param_name: str, 
        component_name: str,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> None:
        """
        Validate that a parameter is a float within specified range.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            component_name: Component name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Raises:
            ValueError: If value is not within the specified range
        """
        if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
            raise ValueError(
                f"{component_name}.{param_name} must be between {min_val} and {max_val}, got {value}"
            )

    @staticmethod
    def validate_choice(
        value: Any, 
        param_name: str, 
        component_name: str,
        valid_choices: List[Any]
    ) -> None:
        """
        Validate that a parameter value is one of the valid choices.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            component_name: Component name for error messages
            valid_choices: List of valid choices
            
        Raises:
            ValueError: If value is not in valid choices
        """
        if value not in valid_choices:
            raise ValueError(
                f"{component_name}.{param_name} must be one of {valid_choices}, got {value}"
            )
