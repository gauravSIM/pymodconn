"""Basic tests for pymodconn package."""

import pytest

import pymodconn


def test_version():
    """Test that version is accessible and correct."""
    assert hasattr(pymodconn, "__version__")
    assert pymodconn.__version__ == "2.0.0"


def test_model_gen_import():
    """Test that Model_Gen can be imported."""
    from pymodconn import Model_Gen
    
    assert Model_Gen is not None


def test_package_structure():
    """Test basic package structure."""
    assert hasattr(pymodconn, "__all__")
    assert "Model_Gen" in pymodconn.__all__


class TestModelGen:
    """Test cases for Model_Gen class."""
    
    def test_model_gen_exists(self):
        """Test that Model_Gen class exists."""
        from pymodconn import Model_Gen
        
        assert Model_Gen is not None
        assert callable(Model_Gen)
    
    @pytest.mark.skip(reason="Requires configuration setup")
    def test_model_gen_initialization(self):
        """Test Model_Gen initialization with basic config."""
        # This test is skipped for now as it requires a proper config
        # Will be implemented in later phases
        pass
