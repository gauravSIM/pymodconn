#!/usr/bin/env python3
"""Check that all layers have corresponding tests and configuration validation."""

import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple


def extract_class_names(file_path: Path, filter_func=None) -> Set[str]:
    """Extract class names from a Python file."""
    if not file_path.exists():
        return set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return set()
    
    class_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if filter_func is None or filter_func(node.name):
                class_names.add(node.name)
    
    return class_names


def extract_function_names(file_path: Path, filter_func=None) -> Set[str]:
    """Extract function/method names from a Python file."""
    if not file_path.exists():
        return set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return set()
    
    function_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if filter_func is None or filter_func(node.name):
                function_names.add(node.name)
    
    return function_names


def is_layer_class(class_name: str) -> bool:
    """Check if a class name represents a layer."""
    return (
        class_name.endswith('Layer') or 
        class_name in ['AddNorm', 'STATES_MANIPULATION_BLOCK', 'MERGE_LIST'] or
        class_name.startswith('GLU') or
        class_name.startswith('GRN')
    )


def is_test_class(class_name: str) -> bool:
    """Check if a class name represents a test class."""
    return class_name.startswith('Test') and class_name != 'Test'


def is_validation_method(method_name: str) -> bool:
    """Check if a method name represents a validation method."""
    return method_name.startswith('_validate_') and method_name.endswith('_config')


def check_layer_test_coverage() -> Tuple[List[str], List[str]]:
    """Check that all layers have corresponding test classes."""
    print("🔍 Checking layer test coverage...")
    
    # Find all layer classes
    utils_layers_path = Path("pymodconn/utils_layers.py")
    layer_classes = extract_class_names(utils_layers_path, is_layer_class)
    
    if not layer_classes:
        print("⚠️  No layer classes found in pymodconn/utils_layers.py")
        return [], []
    
    print(f"Found {len(layer_classes)} layer classes: {sorted(layer_classes)}")
    
    # Find all test classes
    test_file_path = Path("tests/test_utils_layers.py")
    test_classes = extract_class_names(test_file_path, is_test_class)
    
    print(f"Found {len(test_classes)} test classes: {sorted(test_classes)}")
    
    # Check coverage
    missing_tests = []
    extra_tests = []
    
    for layer_class in layer_classes:
        expected_test_class = f"Test{layer_class}"
        if expected_test_class not in test_classes:
            missing_tests.append(layer_class)
    
    # Check for test classes without corresponding layers
    for test_class in test_classes:
        if test_class.startswith('Test'):
            layer_name = test_class[4:]  # Remove 'Test' prefix
            if layer_name not in layer_classes and is_layer_class(layer_name):
                extra_tests.append(test_class)
    
    return missing_tests, extra_tests


def check_config_validation_coverage() -> Tuple[List[str], List[str]]:
    """Check that all layers have corresponding configuration validation."""
    print("\n🔍 Checking configuration validation coverage...")
    
    # Find all layer classes
    utils_layers_path = Path("pymodconn/utils_layers.py")
    layer_classes = extract_class_names(utils_layers_path, is_layer_class)
    
    # Find all validation methods
    config_manager_path = Path("pymodconn/config_manager.py")
    validation_methods = extract_function_names(config_manager_path, is_validation_method)
    
    print(f"Found {len(validation_methods)} validation methods: {sorted(validation_methods)}")
    
    # Check coverage
    missing_validation = []
    extra_validation = []
    
    for layer_class in layer_classes:
        expected_validation_method = f"_validate_{layer_class.lower()}_config"
        if expected_validation_method not in validation_methods:
            missing_validation.append(layer_class)
    
    # Check for validation methods without corresponding layers
    for validation_method in validation_methods:
        if validation_method.startswith('_validate_') and validation_method.endswith('_config'):
            # Extract layer name from method name
            layer_part = validation_method[10:-7]  # Remove '_validate_' and '_config'
            
            # Try to find corresponding layer class
            found_layer = None
            for layer_class in layer_classes:
                if layer_class.lower() == layer_part:
                    found_layer = layer_class
                    break
            
            if not found_layer and layer_part not in ['encoder', 'decoder', 'optimizer', 'quantiles']:
                extra_validation.append(validation_method)
    
    return missing_validation, extra_validation


def check_config_test_coverage() -> Tuple[List[str], List[str]]:
    """Check that all layers have corresponding configuration tests."""
    print("\n🔍 Checking configuration test coverage...")
    
    # Find all layer classes
    utils_layers_path = Path("pymodconn/utils_layers.py")
    layer_classes = extract_class_names(utils_layers_path, is_layer_class)
    
    # Find all config test classes
    config_test_path = Path("tests/test_config_manager.py")
    
    def is_config_test_class(class_name: str) -> bool:
        return class_name.startswith('Test') and 'Config' in class_name
    
    config_test_classes = extract_class_names(config_test_path, is_config_test_class)
    
    print(f"Found {len(config_test_classes)} config test classes: {sorted(config_test_classes)}")
    
    # Check coverage
    missing_config_tests = []
    extra_config_tests = []
    
    for layer_class in layer_classes:
        expected_config_test_class = f"Test{layer_class}ConfigValidation"
        if expected_config_test_class not in config_test_classes:
            missing_config_tests.append(layer_class)
    
    return missing_config_tests, extra_config_tests


def check_fixture_coverage() -> List[str]:
    """Check that layers have corresponding test fixtures."""
    print("\n🔍 Checking test fixture coverage...")
    
    # Find all layer classes
    utils_layers_path = Path("pymodconn/utils_layers.py")
    layer_classes = extract_class_names(utils_layers_path, is_layer_class)
    
    # Find all fixtures
    conftest_path = Path("tests/conftest.py")
    
    def is_layer_fixture(func_name: str) -> bool:
        return func_name.startswith('sample_config_with_') and func_name != 'sample_config_with_temp_dir'
    
    fixtures = extract_function_names(conftest_path, is_layer_fixture)
    
    print(f"Found {len(fixtures)} layer fixtures: {sorted(fixtures)}")
    
    # Check coverage
    missing_fixtures = []
    
    for layer_class in layer_classes:
        expected_fixture = f"sample_config_with_{layer_class.lower()}"
        if expected_fixture not in fixtures:
            missing_fixtures.append(layer_class)
    
    return missing_fixtures


def generate_coverage_report() -> bool:
    """Generate a comprehensive coverage report."""
    print("=" * 80)
    print("🧪 PyModConn Layer Coverage Report")
    print("=" * 80)
    
    all_good = True
    
    # Check test coverage
    missing_tests, extra_tests = check_layer_test_coverage()
    
    if missing_tests:
        print("\n❌ Missing unit tests for the following layers:")
        for layer in sorted(missing_tests):
            print(f"  - {layer} (expected: Test{layer})")
        all_good = False
    else:
        print("\n✅ All layers have unit tests")
    
    if extra_tests:
        print(f"\n⚠️  Found {len(extra_tests)} test classes without corresponding layers:")
        for test_class in sorted(extra_tests):
            print(f"  - {test_class}")
    
    # Check config validation coverage
    missing_validation, extra_validation = check_config_validation_coverage()
    
    if missing_validation:
        print("\n❌ Missing configuration validation for the following layers:")
        for layer in sorted(missing_validation):
            print(f"  - {layer} (expected: _validate_{layer.lower()}_config)")
        all_good = False
    else:
        print("\n✅ All layers have configuration validation")
    
    if extra_validation:
        print(f"\n⚠️  Found {len(extra_validation)} validation methods without corresponding layers:")
        for method in sorted(extra_validation):
            print(f"  - {method}")
    
    # Check config test coverage
    missing_config_tests, extra_config_tests = check_config_test_coverage()
    
    if missing_config_tests:
        print("\n❌ Missing configuration tests for the following layers:")
        for layer in sorted(missing_config_tests):
            print(f"  - {layer} (expected: Test{layer}ConfigValidation)")
        all_good = False
    else:
        print("\n✅ All layers have configuration tests")
    
    # Check fixture coverage
    missing_fixtures = check_fixture_coverage()
    
    if missing_fixtures:
        print("\n⚠️  Missing test fixtures for the following layers:")
        for layer in sorted(missing_fixtures):
            print(f"  - {layer} (expected: sample_config_with_{layer.lower()})")
        print("Note: Fixtures are optional but recommended for integration tests")
    else:
        print("\n✅ All layers have test fixtures")
    
    # Summary
    print("\n" + "=" * 80)
    if all_good:
        print("🎉 All required coverage checks passed!")
        print("Your layer implementation is properly tested and validated.")
    else:
        print("❌ Some coverage checks failed.")
        print("Please add the missing tests and validation methods.")
        print("\n💡 Use 'python scripts/generate_layer_template.py LayerName' to generate boilerplate code.")
    
    print("=" * 80)
    
    return all_good


def main():
    """Main entry point for coverage checking."""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python scripts/check_layer_coverage.py")
        print("\nThis script checks that all layers have corresponding:")
        print("  - Unit tests (Test<LayerName> classes)")
        print("  - Configuration validation (_validate_<layer>_config methods)")
        print("  - Configuration tests (Test<LayerName>ConfigValidation classes)")
        print("  - Test fixtures (sample_config_with_<layer> functions)")
        print("\nRun this script before committing new layers to ensure completeness.")
        return
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if project_root.name != 'pymodconn' and not (project_root / 'pymodconn').exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run coverage checks
    success = generate_coverage_report()
    
    if not success:
        sys.exit(1)
    
    print("\n🚀 Ready to commit your changes!")


if __name__ == "__main__":
    main()
