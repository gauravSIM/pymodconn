"""Basic tests for pymodconn package structure and imports."""

import importlib
import sys
from pathlib import Path

import pytest

import pymodconn


class TestPackageStructure:
    """Test basic package structure and imports."""
    
    def test_package_import(self):
        """Test that pymodconn package can be imported."""
        assert pymodconn is not None
        assert hasattr(pymodconn, "__name__")
        assert pymodconn.__name__ == "pymodconn"
    
    def test_version_exists(self):
        """Test that version is accessible."""
        assert hasattr(pymodconn, "__version__")
        assert isinstance(pymodconn.__version__, str)
        assert len(pymodconn.__version__) > 0
    
    def test_main_exports(self):
        """Test that main classes are exported."""
        assert hasattr(pymodconn, "__all__")
        assert "ModelGen" in pymodconn.__all__
    
    def test_model_gen_import(self):
        """Test that ModelGen can be imported."""
        from pymodconn import ModelGen
        
        assert ModelGen is not None
        assert callable(ModelGen)
        assert hasattr(ModelGen, "__init__")
        assert hasattr(ModelGen, "build_model")


class TestModuleImports:
    """Test that all modules can be imported without errors."""
    
    def test_import_model_gen(self):
        """Test importing model_gen module."""
        from pymodconn import model_gen
        assert model_gen is not None
    
    def test_import_utils_layers(self):
        """Test importing utils_layers module."""
        from pymodconn import utils_layers
        assert utils_layers is not None
    
    def test_import_config_manager(self):
        """Test importing config_manager module."""
        from pymodconn import config_manager
        assert config_manager is not None
    
    def test_import_base_classes(self):
        """Test importing base_classes module."""
        from pymodconn import base_classes
        assert base_classes is not None
    
    def test_import_model_gen_utils(self):
        """Test importing model_gen_utils module."""
        from pymodconn import model_gen_utils
        assert model_gen_utils is not None


class TestCoreClasses:
    """Test that core classes exist and have expected methods."""
    
    def test_model_gen_class(self):
        """Test ModelGen class structure."""
        from pymodconn import ModelGen
        
        # Check class exists and is callable
        assert ModelGen is not None
        assert callable(ModelGen)
        
        # Check required methods exist
        required_methods = ["build_model", "load_model", "forget_model"]
        for method in required_methods:
            assert hasattr(ModelGen, method), f"ModelGen missing method: {method}"
    
    def test_config_manager_class(self):
        """Test ConfigManager class structure."""
        from pymodconn.config_manager import ConfigManager
        
        assert ConfigManager is not None
        assert callable(ConfigManager)
        
        # Check required methods exist
        required_methods = ["load_config", "save_config", "get_config", "update_config"]
        for method in required_methods:
            assert hasattr(ConfigManager, method), f"ConfigManager missing method: {method}"
    
    def test_base_classes_exist(self):
        """Test that base classes exist."""
        from pymodconn.base_classes import BaseEncoder, BaseDecoder, BaseBlock, ConfigValidator
        
        assert BaseEncoder is not None
        assert BaseDecoder is not None
        assert BaseBlock is not None
        assert ConfigValidator is not None
    
    def test_utility_layers_exist(self):
        """Test that utility layers exist."""
        from pymodconn.utils_layers import (
            LinearLayer, GRNLayer, GLUWithAddNorm, 
            AddNorm, GLULayer, soft_relu
        )
        
        assert LinearLayer is not None
        assert GRNLayer is not None
        assert GLUWithAddNorm is not None
        assert AddNorm is not None
        assert GLULayer is not None
        assert soft_relu is not None


class TestPackageMetadata:
    """Test package metadata and structure."""
    
    def test_package_directory_structure(self):
        """Test that expected package directories exist."""
        package_path = Path(pymodconn.__file__).parent
        
        expected_files = [
            "__init__.py",
            "model_gen.py",
            "utils_layers.py",
            "config_manager.py",
            "base_classes.py",
            "model_gen_utils.py"
        ]
        
        for file_name in expected_files:
            file_path = package_path / file_name
            assert file_path.exists(), f"Expected file not found: {file_name}"
    
    def test_configs_directory_exists(self):
        """Test that configs directory exists."""
        package_path = Path(pymodconn.__file__).parent
        configs_path = package_path / "configs"
        
        assert configs_path.exists(), "configs directory not found"
        assert configs_path.is_dir(), "configs is not a directory"
    
    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        # Package should work with Python 3.8+
        assert sys.version_info >= (3, 8), "Python 3.8+ required"


class TestImportPerformance:
    """Test import performance and circular dependencies."""
    
    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # Try importing all modules in different orders
        import_sequences = [
            ["pymodconn.model_gen", "pymodconn.utils_layers", "pymodconn.config_manager"],
            ["pymodconn.config_manager", "pymodconn.base_classes", "pymodconn.model_gen"],
            ["pymodconn.utils_layers", "pymodconn.model_gen_utils", "pymodconn.model_gen"]
        ]
        
        for sequence in import_sequences:
            # Clear modules from cache
            for module_name in sequence:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            
            # Try importing in sequence
            for module_name in sequence:
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    pytest.fail(f"Circular import detected in {module_name}: {e}")
    
    def test_import_speed(self):
        """Test that imports complete in reasonable time."""
        import time
        
        start_time = time.time()
        
        # Re-import the package
        if "pymodconn" in sys.modules:
            del sys.modules["pymodconn"]
        
        import pymodconn
        
        import_time = time.time() - start_time
        
        # Import should complete in less than 5 seconds
        assert import_time < 5.0, f"Import took too long: {import_time:.2f}s"


@pytest.mark.integration
class TestBasicIntegration:
    """Basic integration tests to ensure components work together."""
    
    def test_model_gen_with_config_manager(self, sample_config, mock_datetime):
        """Test that ModelGen works with ConfigManager."""
        from pymodconn import ModelGen
        from pymodconn.config_manager import ConfigManager
        
        # This is a basic integration test - just ensure no import errors
        assert ModelGen is not None
        assert ConfigManager is not None
        
        # Test basic initialization (without building model)
        model_gen = ModelGen(sample_config, mock_datetime)
        assert model_gen.cfg == sample_config
        assert model_gen.current_dt == mock_datetime
    
    def test_utility_layers_integration(self):
        """Test that utility layers can be imported and used together."""
        from pymodconn.utils_layers import LinearLayer, AddNorm, soft_relu
        import tensorflow as tf
        
        # Test basic functionality
        linear = LinearLayer(hidden_layer_size=16)
        add_norm = AddNorm()
        
        x = tf.random.normal([2, 8])
        y = tf.random.normal([2, 8])
        
        # Test that layers can be called
        linear_out = linear(x)
        add_norm_out = add_norm(x, y)
        soft_relu_out = soft_relu(x)
        
        assert linear_out.shape == (2, 16)
        assert add_norm_out.shape == (2, 8)
        assert soft_relu_out.shape == (2, 8)
