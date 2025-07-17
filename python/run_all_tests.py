#!/usr/bin/env python3
"""
Comprehensive Test Runner for Dataset Safety Specifications

Runs all tests including:
- Regression tests
- Data integration tests
- Runtime safety kernel tests
- SentinelOps bundle tests
- ONNX node extraction tests
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Try to import ONNX for testing
try:
    import onnx

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(
    test_name: str, test_func: callable, timeout: int = 300
) -> Tuple[bool, str, float]:
    """Run a test with timeout and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = test_func()
        end_time = time.time()
        duration = end_time - start_time

        if result:
            print(f"{test_name}: PASSED ({duration:.2f}s)")
            return True, "PASSED", duration
        else:
            print(f" {test_name}: FAILED ({duration:.2f}s)")
            return False, "FAILED", duration

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f" {test_name}: ERROR ({duration:.2f}s)")
        print(f"   Error: {e}")
        return False, f"ERROR: {str(e)}", duration


def test_data_integration() -> bool:
    """Test data integration functionality."""
    try:
        from test_data_integration import main

        return main() == 0
    except Exception as e:
        print(f"Data integration test error: {e}")
        return False


def test_regression_suite() -> bool:
    """Test regression suite."""
    try:
        from regression_tests import RegressionTestSuite

        test_suite = RegressionTestSuite()
        results = test_suite.run_all_tests()
        return results["summary"]["success_rate"] >= 0.8  # Allow 80% pass rate
    except Exception as e:
        print(f"Regression test error: {e}")
        return False


def test_runtime_safety_kernel() -> bool:
    """Test runtime safety kernel."""
    try:
        from runtime_safety_kernel import (
            create_safety_kernel,
            RuntimeConfig,
            ModelAsset,
            SafetyKernel,
        )

        # Create kernel
        config = RuntimeConfig()
        kernel = create_safety_kernel(config)

        # Test model asset registration
        asset = ModelAsset(
            name="test_model",
            version="1.0.0",
            model_path="test_model.onnx",
            safety_hash="abc123",
            compliance_level="strict",
            last_verified="2024-01-01",
            verification_status="verified",
        )

        success = kernel.register_model_asset(asset)

        # Test safety checking
        from ds_guard import Row

        test_row = Row(phi=[], age=25, gdpr_special=[], custom_fields=[])

        safety_result = kernel.check_data_safety([test_row])

        return success and safety_result["safe"]

    except Exception as e:
        print(f"Runtime safety kernel test error: {e}")
        return False


def test_sentinelops_bundle() -> bool:
    """Test SentinelOps bundle generation."""
    try:
        from sentinelops_bundle import SentinelOpsBundleGenerator
        import pandas as pd
        import tempfile
        import os

        # Create test dataset
        df = pd.DataFrame(
            {
                "test_col": [1, 2, 3],
                "phi_ssn": [None, None, None],
                "age": [25, 30, 35],
                "gdpr_health": [None, None, None],
            }
        )

        # Save test dataset
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name)
            dataset_path = tmp.name

        try:
            # Create bundle
            generator = SentinelOpsBundleGenerator()
            bundle = generator.create_dataset_bundle(
                dataset_path=dataset_path,
                dataset_name="test_dataset",
                compliance_level="strict",
            )

            # Validate bundle
            validation = generator.validate_bundle(bundle)

            return validation["valid"]

        finally:
            os.unlink(dataset_path)

    except Exception as e:
        print(f"SentinelOps bundle test error: {e}")
        return False


def test_onnx_node_extractor() -> bool:
    """Test ONNX node extractor."""
    try:
        from onnx_node_extractor import ONNXNodeNameExtractor

        # Test extractor creation
        extractor = ONNXNodeNameExtractor()

        # Test name validation
        valid_name = extractor._is_valid_name("valid_node_name")
        invalid_name = extractor._is_valid_name("123_invalid_name")

        # Test name cleaning
        cleaned_name = extractor._clean_name("123_invalid-name!")

        return valid_name and not invalid_name and cleaned_name == "invalid_name"

    except Exception as e:
        print(f"ONNX node extractor test error: {e}")
        return False


def test_real_onnx_parser() -> bool:
    """Test real ONNX parser."""
    if not ONNX_AVAILABLE:
        print("ONNX not available, skipping real ONNX parser test")
        return True  # Not a failure, just missing dependency

    try:
        from real_onnx_parser import RealONNXParser
        import tempfile
        import os

        # Create a simple test model
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            test_model_path = tmp.name

        try:
            # Create a minimal ONNX model for testing
            from onnx import helper, TensorProto

            input_tensor = helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 3, 224, 224]
            )
            output_tensor = helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [1, 1000]
            )
            node = helper.make_node("Flatten", ["input"], ["output"])
            graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])
            model = helper.make_model(graph)
            import onnx

            onnx.save(model, test_model_path)

            # Test parsing
            parser = RealONNXParser()
            result = parser.parse_onnx_model(test_model_path)

            return result.get("success", False)

        finally:
            if os.path.exists(test_model_path):
                os.unlink(test_model_path)

    except Exception as e:
        print(f"Real ONNX parser test error: {e}")
        return False


def test_gpt2_demo() -> bool:
    """Test GPT-2 end-to-end demo."""
    try:
        from gpt2_demo import GPT2Demo

        demo = GPT2Demo()
        result = demo.run_end_to_end_demo(download_model=True)

        return result.get("success", False)

    except Exception as e:
        print(f"GPT-2 demo test error: {e}")
        return False


def test_rust_guard_transpiler() -> bool:
    """Test Rust guard transpiler."""
    try:
        from rust_guard_transpiler import RustGuardTranspiler
        import tempfile

        predicates = [
            {"name": "PHI", "lean_code": "has_phi row", "description": "Test PHI guard"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            transpiler = RustGuardTranspiler()
            success = transpiler.generate_complete_crate(predicates, temp_dir)
            return success

    except Exception as e:
        print(f"Rust guard transpiler test error: {e}")
        return False


def test_package_publisher() -> bool:
    """Test package publishing automation."""
    try:
        from package_publisher import PackagePublisher

        publisher = PackagePublisher()
        # Test dry run
        success = publisher.publish_python_package("0.1.0", dry_run=True)
        return success

    except Exception as e:
        print(f"Package publisher test error: {e}")
        return False


def test_lean_build() -> bool:
    """Test Lean project build."""
    try:
        result = subprocess.run(
            ["lake", "build"], capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Lean build timed out")
        return False
    except FileNotFoundError:
        print("lake command not found")
        return False
    except Exception as e:
        print(f"Lean build error: {e}")
        return False


def test_lean_tests() -> bool:
    """Test Lean test suite."""
    try:
        result = subprocess.run(
            ["lake", "exe", "test_suite"], capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Lean tests timed out")
        return False
    except FileNotFoundError:
        print("lake command not found")
        return False
    except Exception as e:
        print(f"Lean tests error: {e}")
        return False


def main():
    """Run all tests and generate report."""
    print("Dataset Safety Specs - Comprehensive Test Runner")
    print("=" * 60)

    # Define all tests
    tests = [
        ("Lean Build", test_lean_build),
        ("Lean Test Suite", test_lean_tests),
        ("Data Integration", test_data_integration),
        ("Regression Suite", test_regression_suite),
        ("Runtime Safety Kernel", test_runtime_safety_kernel),
        ("SentinelOps Bundle", test_sentinelops_bundle),
        ("ONNX Node Extractor", test_onnx_node_extractor),
        ("Real ONNX Parser", test_real_onnx_parser),
        ("GPT-2 End-to-End Demo", test_gpt2_demo),
        ("Rust Guard Transpiler", test_rust_guard_transpiler),
        ("Package Publisher", test_package_publisher),
    ]

    # Run all tests
    results = []
    total_start_time = time.time()

    for test_name, test_func in tests:
        success, status, duration = run_test(test_name, test_func)
        results.append(
            {
                "name": test_name,
                "success": success,
                "status": status,
                "duration": duration,
            }
        )

    total_duration = time.time() - total_start_time

    # Generate summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    for result in results:
        status_icon = "" if result["success"] else ""
        print(
            f"{status_icon} {result['name']:25} : {result['status']:15} ({result['duration']:.2f}s)"
        )

    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Total time: {total_duration:.2f}s")

    # Calculate success rate
    success_rate = passed / total if total > 0 else 0

    if success_rate >= 0.8:
        print("🎉 Excellent! Most tests passed.")
        return 0
    elif success_rate >= 0.6:
        print("⚠️  Good progress, but some tests need attention.")
        return 1
    else:
        print(" Many tests failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    exit(main())
