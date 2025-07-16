#!/usr/bin/env python3
"""
Simple Component Test

Simplified test script for core functionality without external dependencies.
"""

import sys
import tempfile
import os
from pathlib import Path


def test_real_onnx_parser():
    """Test real ONNX parser import and basic functionality."""
    print("Testing Real ONNX Parser...")
    try:
        from real_onnx_parser import RealONNXParser

        parser = RealONNXParser()
        print("✓ Real ONNX Parser imported successfully")
        return True
    except Exception as e:
        print(f"✗ Real ONNX Parser test failed: {e}")
        return False


def test_gpt2_demo():
    """Test GPT-2 demo import and basic functionality."""
    print("Testing GPT-2 Demo...")
    try:
        from gpt2_demo import GPT2Demo

        demo = GPT2Demo()
        print("✓ GPT-2 Demo imported successfully")
        return True
    except Exception as e:
        print(f"✗ GPT-2 Demo test failed: {e}")
        return False


def test_rust_guard_transpiler():
    """Test Rust guard transpiler with simple predicates."""
    print("Testing Rust Guard Transpiler...")
    try:
        from rust_guard_transpiler import RustGuardTranspiler

        predicates = [
            {"name": "PHI", "lean_code": "has_phi row", "description": "Test PHI guard"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            transpiler = RustGuardTranspiler()
            success = transpiler.generate_complete_crate(predicates, temp_dir)

            if success:
                print("✓ Rust Guard Transpiler test passed")
                return True
            else:
                print("✗ Rust Guard Transpiler test failed")
                return False

    except Exception as e:
        print(f"✗ Rust Guard Transpiler test failed: {e}")
        return False


def test_package_publisher():
    """Test package publisher import and basic functionality."""
    print("Testing Package Publisher...")
    try:
        from package_publisher import PackagePublisher

        publisher = PackagePublisher()
        print("✓ Package Publisher imported successfully")
        return True
    except Exception as e:
        print(f"✗ Package Publisher test failed: {e}")
        return False


def main():
    """Run all simple tests."""
    print("Simple Component Testing")
    print("=" * 40)

    tests = [
        ("Real ONNX Parser", test_real_onnx_parser),
        ("GPT-2 Demo", test_gpt2_demo),
        ("Rust Guard Transpiler", test_rust_guard_transpiler),
        ("Package Publisher", test_package_publisher),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test error: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*40}")
    print("TEST SUMMARY")
    print(f"{'='*40}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:25} : {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All components are working correctly!")
        return 0
    else:
        print("⚠️  Some components need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
