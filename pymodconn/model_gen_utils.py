"""Model generation utilities for PyModConn."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.utils.layer_utils import count_params
from keras.utils.vis_utils import plot_model

from pymodconn.configs.configs_init import read_write_yaml

K = tf.keras.backend

# Set up logging
logger = logging.getLogger(__name__)


class Timer:
    """
    Timer utility for measuring execution time.
    
    This class provides a simple interface for timing operations
    with proper error handling and logging.
    """

    def __init__(self) -> None:
        """Initialize Timer with no start time."""
        self.start_dt: Optional[datetime] = None

    def start(self) -> None:
        """Start the timer."""
        self.start_dt = datetime.now()
        logger.debug("Timer started at %s", self.start_dt)

    def stop(self) -> datetime:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time as a timedelta object
            
        Raises:
            RuntimeError: If timer was not started
        """
        if self.start_dt is None:
            raise RuntimeError("Timer was not started. Call start() first.")
            
        end_dt = datetime.now()
        elapsed_time = end_dt - self.start_dt
        
        logger.info("Time taken: %s", elapsed_time)
        print(f"Time taken: {elapsed_time}")
        print()
        
        return elapsed_time

    def __enter__(self) -> Timer:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

class BuildUtils:
    """
    Utility class for building, compiling, and saving neural network models.
    
    This class handles model compilation, summary generation, visualization,
    and configuration saving with proper error handling and modern Python features.
    
    Args:
        cfg: Configuration dictionary containing model parameters
        current_dt: Current datetime string for file naming
    """

    def __init__(self, cfg: Dict[str, Any], current_dt: str) -> None:
        """Initialize BuildUtils with configuration and datetime."""
        self.cfg = cfg
        self.current_dt = current_dt
        
        # Validate required configuration keys
        self._validate_config()
        
        # Model parameters
        self.batch_size: int = cfg["batch_size"]
        self.optimizer_name: str = cfg["optimizer"]
        self.loss_func: str = cfg["loss"]
        self.model_type_prob: str = cfg["model_type_prob"]
        self.loss_prob: str = cfg["loss_prob"]
        self.metrics: List[str] = cfg["metrics"]
        
        # Optimizer parameters
        self.sgd_params = cfg.get("SGD", {})
        self.adam_params = cfg.get("Adam", {})
        
        # Quantiles processing
        self.q = self._process_quantiles(cfg["quantiles"])
        self.n_outputs_lastlayer = (
            2 if self.loss_prob == "parametric" else len(self.q)
        )
        
        # Output settings
        self.if_save_model_image: bool = cfg["if_model_image"]
        self.if_model_summary: bool = cfg["if_model_summary"]
        
        # File paths using pathlib for better path handling
        self.save_models_dir = Path(cfg["save_models_dir"])
        self.save_models_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_modelimage_name = (
            self.save_models_dir / f"{self.current_dt}_modelimage.png"
        )
        self.save_modelsummary_name = (
            self.save_models_dir / f"{self.current_dt}_modelsummary.txt"
        )
        self.save_modelconfig = (
            self.save_models_dir / f"{self.current_dt}_modelconfig.yaml"
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_keys = [
            "batch_size", "optimizer", "loss", "model_type_prob", 
            "loss_prob", "quantiles", "metrics", "if_model_image", 
            "if_model_summary", "save_models_dir"
        ]
        
        missing_keys = [key for key in required_keys if key not in self.cfg]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
            
        # Validate optimizer
        if self.cfg["optimizer"] not in ["Adam", "SGD"]:
            raise ValueError(f"Unsupported optimizer: {self.cfg['optimizer']}")
            
        # Validate model type
        if self.cfg["model_type_prob"] not in ["prob", "nonprob"]:
            raise ValueError(f"Invalid model_type_prob: {self.cfg['model_type_prob']}")

    def _process_quantiles(self, quantiles: List[float]) -> np.ndarray:
        """
        Process and validate quantiles.
        
        Args:
            quantiles: List of quantile values
            
        Returns:
            Processed quantiles array with 0.5 included
        """
        q_array = np.unique(np.array(quantiles))
        
        # Ensure 0.5 (median) is included
        if 0.5 not in q_array:
            q_array = np.sort(np.append(0.5, q_array))
            
        # Validate quantile range
        if np.any((q_array <= 0) | (q_array >= 1)):
            raise ValueError("Quantiles must be between 0 and 1 (exclusive)")
            
        return q_array

    def cvrmse_q50_prob_nonparametric(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate CVRMSE for Q50 in non-parametric probabilistic models.
        
        Args:
            y_true: True values tensor
            y_pred: Predicted values tensor
            
        Returns:
            CVRMSE metric value
        """
        return K.sqrt(K.mean(K.square(y_pred[:, :, :, 3] - y_true))) / K.mean(y_true)

    def cvrmse_q50_prob_parametric(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate CVRMSE for Q50 in parametric probabilistic models.
        
        Args:
            y_true: True values tensor
            y_pred: Predicted values tensor
            
        Returns:
            CVRMSE metric value
        """
        return K.sqrt(K.mean(K.square(y_pred[:, :, :, 0] - y_true))) / K.mean(y_true)

    def cvrmse_q50_nonprob(
        self, y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate CVRMSE for non-probabilistic models.
        
        Args:
            y_true: True values tensor
            y_pred: Predicted values tensor
            
        Returns:
            CVRMSE metric value
        """
        return K.sqrt(K.mean(K.square(y_pred - y_true))) / K.mean(y_true)

    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Create optimizer based on configuration.
        
        Returns:
            Configured optimizer instance
            
        Raises:
            ValueError: If optimizer type is not supported
        """
        if self.optimizer_name == "Adam":
            return tf.keras.optimizers.Adam(
                learning_rate=self.adam_params.get("lr", 0.001),
                beta_1=self.adam_params.get("b1", 0.9),
                beta_2=self.adam_params.get("b2", 0.999),
                epsilon=self.adam_params.get("epsi", 1e-8),
            )
        elif self.optimizer_name == "SGD":
            return tf.keras.optimizers.SGD(
                learning_rate=self.sgd_params.get("lr", 0.01),
                momentum=self.sgd_params.get("momentum", 0.0),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _compile_model(self, model: tf.keras.Model) -> None:
        """
        Compile the model with appropriate loss function and metrics.
        
        Args:
            model: Keras model to compile
        """
        optimizer = self._create_optimizer()
        
        if self.model_type_prob == "prob":
            if self.loss_prob == "parametric":
                # Maximum likelihood estimation for mean and std deviation
                model.compile(
                    loss=parametric_loss,
                    optimizer=optimizer,
                    metrics=[self.cvrmse_q50_prob_parametric],
                    run_eagerly=True,
                )
            elif self.loss_prob == "nonparametric":
                # Non-parametric approach with quantile prediction
                model.compile(
                    loss=lambda y_true, y_pred: nonparametric_loss(y_true, y_pred, self.q),
                    optimizer=optimizer,
                    metrics=[self.cvrmse_q50_prob_nonparametric],
                    run_eagerly=True,
                )
        elif self.model_type_prob == "nonprob":
            # Non-probabilistic model
            metrics = self.metrics.copy()
            metrics.append(self.cvrmse_q50_nonprob)
            model.compile(
                loss=self.loss_func,
                optimizer=optimizer,
                metrics=metrics,
                run_eagerly=True,
            )

    def _save_model_summary(self, model: tf.keras.Model, model_size_gb: float) -> None:
        """
        Save model summary to file.
        
        Args:
            model: Keras model
            model_size_gb: Model size in GB
        """
        try:
            with open(self.save_modelsummary_name, "w", encoding="utf-8") as f:
                model.summary(
                    print_fn=lambda x: f.write(x + "\n"),
                    line_length=250,
                    expand_nested=True,
                    show_trainable=True,
                )

            with open(self.save_modelsummary_name, "a", encoding="utf-8") as f:
                f.write("_" * 25 + "\n")
                f.write(f"Model size in GB: {model_size_gb:.6f}\n")
                
            logger.info("Model summary saved to %s", self.save_modelsummary_name)
        except IOError as e:
            logger.error("Failed to save model summary: %s", e)
            raise

    def _save_model_visualization(self, model: tf.keras.Model) -> None:
        """
        Save model visualization as PNG.
        
        Args:
            model: Keras model to visualize
        """
        if not self.if_save_model_image:
            return
            
        try:
            logger.info("Saving model visualization to %s", self.save_modelimage_name)
            plot_model(
                model,
                to_file=str(self.save_modelimage_name),
                show_shapes=True,
                show_layer_names=True,
                dpi=600,
                expand_nested=True,
            )
            print(f"Model visualization saved as {self.save_modelimage_name}")
        except Exception as e:
            logger.warning("Failed to save model visualization: %s", e)
            print(f"Warning: Could not save model visualization: {e}")

    def _save_model_config(self) -> None:
        """Save model configuration to YAML file."""
        try:
            read_write_yaml(str(self.save_modelconfig), "w", self.cfg)
            logger.info("Model configuration saved to %s", self.save_modelconfig)
        except Exception as e:
            logger.error("Failed to save model configuration: %s", e)
            raise

    def postbuild_model(self, model: tf.keras.Model) -> Tuple[float, int]:
        """
        Post-process model after building: compile, summarize, and save.
        
        This method handles the complete post-build workflow including:
        - Model compilation with appropriate optimizer and loss
        - Model summary generation and saving
        - Model visualization (if enabled)
        - Configuration saving
        
        Args:
            model: Built Keras model to post-process
            
        Returns:
            Tuple of (model_size_gb, trainable_parameters_count)
            
        Raises:
            ValueError: If model is None or configuration is invalid
            IOError: If file operations fail
        """
        if model is None:
            raise ValueError("Model cannot be None")
            
        self.model = model
        
        try:
            # Compile the model
            self._compile_model(model)
            
            # Display summary if requested
            if self.if_model_summary:
                model.summary()
            
            # Calculate model statistics
            trainable_count = count_params(model.trainable_weights)
            model_size_gb = get_model_memory_usage(self.batch_size, model)
            
            print(f"Trainable parameters in the model: {trainable_count:,}")
            
            # Save model artifacts
            self._save_model_summary(model, model_size_gb)
            self._save_model_visualization(model)
            self._save_model_config()
            
            logger.info("Model post-build processing completed successfully")
            print("[ModelClass] Building postmodel DONE!! Model can be used as self.model")
            
            return model_size_gb, trainable_count
            
        except Exception as e:
            logger.error("Failed to post-process model: %s", e)
            raise


def get_model_memory_usage(batch_size: int, model: tf.keras.Model) -> float:
    """
    Calculate memory usage of a Keras model in GB.
    
    This function estimates the memory requirements for a given model
    including parameters, activations, and intermediate computations.
    
    Args:
        batch_size: Batch size for memory calculation
        model: Keras model to analyze
        
    Returns:
        Estimated memory usage in GB
        
    Raises:
        ValueError: If batch_size is not positive or model is None
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if model is None:
        raise ValueError("Model cannot be None")
    
    shapes_mem_count = 0
    internal_model_mem_count = 0.0
    
    # Calculate memory for each layer
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        
        # Handle nested models recursively
        if layer_type == "Model":
            internal_model_mem_count += get_model_memory_usage(batch_size, layer)
            continue
            
        # Calculate memory for layer outputs
        single_layer_mem = 1
        out_shape = layer.output_shape
        
        # Handle multiple outputs
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
            
        # Calculate total elements in output shape
        for dim_size in out_shape:
            if dim_size is not None:
                single_layer_mem *= dim_size
                
        shapes_mem_count += single_layer_mem

    # Calculate parameter counts
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    # Determine number size based on precision
    precision_map = {
        "float16": 2.0,
        "float32": 4.0,
        "float64": 8.0,
    }
    number_size = precision_map.get(K.floatx(), 4.0)

    # Calculate total memory
    total_memory = number_size * (
        batch_size * shapes_mem_count + trainable_count + non_trainable_count
    )
    
    # Convert to GB
    gbytes = np.round(total_memory / (1024.0**3), 3) + internal_model_mem_count
    
    logger.info("Memory usage for model: %.3f GB", gbytes)
    print(f"Memory usage for model: {gbytes:.3f} GB")
    
    return gbytes


def nonparametric_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, quantiles: Union[np.ndarray, tf.Tensor]
) -> tf.Tensor:
    """
    Nonparametric quantile loss function.
    
    Implementation of the quantile loss function as described in Section 3.2.1
    of the DeepTCN paper. This loss function is used for non-parametric
    probabilistic forecasting where quantiles are directly predicted.
    
    Args:
        y_true: Actual values of target time series with shape 
                (n_samples, n_forecast, n_targets)
        y_pred: Predicted quantiles with shape 
                (n_samples, n_forecast, n_targets, n_quantiles)
        quantiles: Quantile values as 1D array with length n_quantiles
        
    Returns:
        Scalar tensor representing the quantile loss value
        
    Raises:
        ValueError: If tensor shapes are incompatible
    """
    # Ensure proper tensor types and shapes
    y_true = tf.cast(tf.expand_dims(y_true, axis=3), dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    # Reshape quantiles for broadcasting
    q = tf.cast(tf.reshape(quantiles, shape=(1, len(quantiles))), dtype=tf.float32)
    
    # Calculate prediction errors
    errors = tf.subtract(y_true, y_pred)
    
    # Apply quantile loss formula: q * max(0, e) + (1-q) * max(0, -e)
    positive_errors = tf.maximum(0.0, errors)
    negative_errors = tf.maximum(0.0, -errors)
    
    quantile_loss = (
        tf.multiply(q, positive_errors) + tf.multiply(1.0 - q, negative_errors)
    )
    
    # Reduce across all dimensions
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(quantile_loss, axis=-1), axis=-1))


def parametric_loss(y_true: tf.Tensor, params: tf.Tensor) -> tf.Tensor:
    """
    Parametric loss function for Gaussian distribution.
    
    Implementation of the negative log-likelihood loss for a Gaussian distribution
    as described in Section 3.2.2 of the DeepTCN paper. This loss function is used
    for parametric probabilistic forecasting where mean and standard deviation
    are predicted.
    
    Args:
        y_true: Actual values of target time series with shape 
                (n_samples, n_forecast, n_targets)
        params: Predicted parameters (mean, std) with shape 
                (n_samples, n_forecast, n_targets, 2)
                
    Returns:
        Scalar tensor representing the negative log-likelihood
        
    Raises:
        ValueError: If tensor shapes are incompatible
    """
    # Ensure proper tensor types
    y_true = tf.cast(y_true, dtype=tf.float32)
    params = tf.cast(params, dtype=tf.float32)
    
    # Extract mean and standard deviation
    mu = params[:, :, :, 0]
    sigma = params[:, :, :, 1]
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    sigma = tf.maximum(sigma, epsilon)
    
    # Calculate negative log-likelihood for Gaussian distribution
    # L = 0.5 * log(2π) + log(σ) + (y - μ)² / (2σ²)
    log_likelihood = (
        0.5 * tf.math.log(2 * np.pi)
        + tf.math.log(sigma)
        + tf.math.divide(
            tf.math.pow(y_true - mu, 2), 2 * tf.math.pow(sigma, 2)
        )
    )
    
    # Handle NaN values and reduce across dimensions
    return tf.reduce_mean(
        tf.where(
            tf.math.is_nan(log_likelihood),
            tf.zeros_like(log_likelihood),
            log_likelihood
        )
    )
