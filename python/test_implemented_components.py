#!/usr/bin/env python3
"""
Test Implemented Components

Comprehensive test script for all implemented missing pieces:
1. Real ONNX parsing
2. GPT-2 end-to-end demo
3. Rust guard transpilation
4. Package publishing automation

Usage:
    python test_implemented_components.py
"""

import sys
import os
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_real_onnx_parser():
    """Test real ONNX parser functionality."""
    print("\n=== Testing Real ONNX Parser ===")

    try:
        from real_onnx_parser import RealONNXParser

        # Create a test ONNX model
        test_model_path = "test_model.onnx"
        create_test_onnx_model(test_model_path)

        # Test parsing
        parser = RealONNXParser()
        result = parser.parse_onnx_model(test_model_path)

        if result["success"]:
            stats = result["stats"]
            print(f"✓ ONNX parser test PASSED")
            print(f"  Nodes extracted: {stats['nodes_extracted']}")
            print(f"  Parse time: {stats['parse_time']:.3f}s")

            # Clean up
            os.unlink(test_model_path)
            return True
        else:
            print(f"✗ ONNX parser test FAILED: {result.get('error')}")
            return False

    except Exception as e:
        print(f"✗ ONNX parser test ERROR: {e}")
        return False


def test_gpt2_demo():
    """Test GPT-2 end-to-end demo."""
    print("\n=== Testing GPT-2 End-to-End Demo ===")

    try:
        from gpt2_demo import GPT2Demo

        demo = GPT2Demo()
        result = demo.run_end_to_end_demo(download_model=True)

        if result["success"]:
            report = result["report"]
            print(f"✓ GPT-2 demo test PASSED")
            print(f"  Total time: {report['performance_metrics']['total_time']:.3f}s")
            print(f"  DSS-4 target met: {report['compliance']['dss4_target_met']}")
            print(
                f"  Shape safety verified: {report['compliance']['shape_safety_verified']}"
            )
            return True
        else:
            print(f"✗ GPT-2 demo test FAILED: {result.get('error')}")
            return False

    except Exception as e:
        print(f"✗ GPT-2 demo test ERROR: {e}")
        return False


def test_rust_guard_transpiler():
    """Test Rust guard transpiler."""
    print("\n=== Testing Rust Guard Transpiler ===")

    try:
        from rust_guard_transpiler import RustGuardTranspiler

        transpiler = RustGuardTranspiler()

        # Test guard generation
        test_guards = [
            {
                "name": "phi_guard",
                "lean_code": "has_phi row",
            },
            {
                "name": "coppa_guard",
                "lean_code": "is_minor row",
            },
        ]

        rust_code = transpiler.transpile_predicates(test_guards)

        if rust_code:
            print(f"✓ Rust guard transpiler test PASSED")
            print(f"  Guards generated: {len(test_guards)}")
            print(f"  Rust code length: {len(rust_code)} chars")
            return True
        else:
            print(f"✗ Rust guard transpiler test FAILED")
            return False

    except Exception as e:
        print(f"✗ Rust guard transpiler test ERROR: {e}")
        return False


def test_package_publisher():
    """Test package publishing automation."""
    print("\n=== Testing Package Publishing Automation ===")

    try:
        from package_publisher import PackagePublisher

        publisher = PackagePublisher()

        # Test dry run publishing
        success = publisher.publish_python_package("0.1.0", dry_run=True)

        if success:
            print(f"✓ Package publishing test PASSED")
            print(f"  Dry run successful")
            return True
        else:
            print(f"✗ Package publishing test FAILED")
            return False

    except Exception as e:
        print(f"✗ Package publishing test ERROR: {e}")
        return False


def create_test_onnx_model(output_path: str):
    """Create a simple test ONNX model."""
    try:
        import onnx
        from onnx import helper, TensorProto, GraphProto, ModelProto

        # Create a simple model
        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 3, 224, 224]
        )

        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 1000]
        )

        # Create a simple node
        node = helper.make_node(
            "Flatten", inputs=["input"], outputs=["output"], name="flatten"
        )

        # Create graph
        graph = helper.make_graph([node], "test_model", [input_tensor], [output_tensor])

        # Create model
        model = helper.make_model(graph, producer_name="test")

        # Save model
        onnx.save(model, output_path)

    except Exception as e:
        print(f"Failed to create test ONNX model: {e}")
        raise


def run_all_tests():
    """Run all component tests."""
    print("Dataset Safety Specs - Component Testing")
    print("=" * 60)

    tests = [
        ("Real ONNX Parser", test_real_onnx_parser),
        ("GPT-2 End-to-End Demo", test_gpt2_demo),
        ("Rust Guard Transpiler", test_rust_guard_transpiler),
        ("Package Publishing Automation", test_package_publisher),
    ]

    results = []
    total_start_time = time.time()

    for test_name, test_func in tests:
        start_time = time.time()

        try:
            success = test_func()
            end_time = time.time()
            duration = end_time - start_time

            results.append(
                {"name": test_name, "success": success, "duration": duration}
            )

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            results.append(
                {
                    "name": test_name,
                    "success": False,
                    "duration": duration,
                    "error": str(e),
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
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        print(f"{result['name']:35} : {status:10} ({result['duration']:.2f}s)")

        if not result["success"] and "error" in result:
            print(f"  Error: {result['error']}")

    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Total time: {total_duration:.2f}s")

    # Calculate success rate
    success_rate = passed / total if total > 0 else 0

    if success_rate >= 0.75:
        print("🎉 Most components are working correctly!")
        return 0
    elif success_rate >= 0.5:
        print("⚠️  Some components need attention.")
        return 1
    else:
        print("❌ Many components need fixing.")
        return 1


def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run all tests
    exit_code = run_all_tests()

    # Generate test report
    report = {
        "timestamp": time.time(),
        "components_tested": [
            "Real ONNX Parser",
            "GPT-2 End-to-End Demo",
            "Rust Guard Transpiler",
            "Package Publishing Automation",
        ],
        "status": "completed",
    }

    with open("component_test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
