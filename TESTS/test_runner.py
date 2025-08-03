"""Modern test runner for PyModConn with comprehensive reporting."""

import argparse
import sys
import time
from pathlib import Path

import pytest


def run_tests(
    test_path: str = "tests/",
    verbose: bool = True,
    coverage: bool = True,
    markers: str = None,
    parallel: bool = False,
    html_report: bool = False
) -> int:
    """
    Run tests with modern pytest configuration.
    
    Args:
        test_path: Path to test directory or specific test file
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        markers: Run only tests with specific markers (e.g., 'integration', 'unit')
        parallel: Run tests in parallel using pytest-xdist
        html_report: Generate HTML coverage report
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = []
    
    # Basic pytest arguments
    if verbose:
        args.extend(["-v", "--tb=short"])
    
    # Coverage arguments
    if coverage:
        args.extend([
            "--cov=pymodconn",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ])
        
        if html_report:
            args.append("--cov-report=html")
    
    # Marker filtering
    if markers:
        args.extend(["-m", markers])
    
    # Parallel execution
    if parallel:
        args.extend(["-n", "auto"])
    
    # Additional useful options
    args.extend([
        "--strict-markers",  # Ensure all markers are defined
        "--disable-warnings",  # Reduce noise in output
        "--color=yes",  # Colored output
        test_path
    ])
    
    print(f"Running tests with command: pytest {' '.join(args)}")
    print("-" * 60)
    
    start_time = time.time()
    exit_code = pytest.main(args)
    end_time = time.time()
    
    print("-" * 60)
    print(f"Tests completed in {end_time - start_time:.2f} seconds")
    
    return exit_code


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run PyModConn tests with modern configuration"
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        default="tests/",
        help="Path to test directory or specific test file (default: tests/)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "-m", "--markers",
        help="Run only tests with specific markers (e.g., 'integration', 'unit')"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (excludes integration tests)"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )
    
    args = parser.parse_args()
    
    # Handle quick/integration flags
    markers = args.markers
    if args.quick:
        markers = "not integration"
    elif args.integration:
        markers = "integration"
    
    exit_code = run_tests(
        test_path=args.path,
        verbose=args.verbose,
        coverage=not args.no_coverage,
        markers=markers,
        parallel=args.parallel,
        html_report=args.html
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
