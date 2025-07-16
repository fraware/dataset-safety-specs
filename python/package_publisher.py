#!/usr/bin/env python3
"""
Package Publishing Automation

Automates publishing of Python and Rust packages to PyPI and crates.io.
This implements the package publishing part of DSS-5.

Usage:
    python package_publisher.py --publish-python --publish-rust --version 0.1.0
"""

import sys
import os
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class PackagePublisher:
    """Automates package publishing to PyPI and crates.io."""

    def __init__(self):
        self.logger = logging.getLogger("PackagePublisher")
        self.publishing_stats = {
            "python_published": False,
            "rust_published": False,
            "python_version": None,
            "rust_version": None,
            "errors": [],
            "warnings": [],
        }

        # Package configurations
        self.package_configs = {
            "python": {
                "name": "ds-guard",
                "description": "Dataset safety guards generated from Lean predicates",
                "author": "Dataset Safety Specs",
                "license": "MIT",
                "python_requires": ">=3.8",
                "classifiers": [
                    "Development Status :: 4 - Beta",
                    "Intended Audience :: Developers",
                    "License :: OSI Approved :: MIT License",
                    "Programming Language :: Python :: 3",
                    "Programming Language :: Python :: 3.8",
                    "Programming Language :: Python :: 3.9",
                    "Programming Language :: Python :: 3.10",
                    "Programming Language :: Python :: 3.11",
                    "Topic :: Software Development :: Libraries :: Python Modules",
                    "Topic :: Security",
                ],
            },
            "rust": {
                "name": "ds-guard",
                "description": "Dataset safety guards generated from Lean predicates",
                "license": "MIT",
                "authors": ["Dataset Safety Specs"],
                "edition": "2021",
                "repository": "https://github.com/your-org/dataset-safety-specs",
                "keywords": ["dataset", "safety", "guards", "compliance"],
                "categories": ["security", "data-processing"],
            },
        }

    def publish_python_package(self, version: str, dry_run: bool = False) -> bool:
        """Publish Python package to PyPI."""
        try:
            self.logger.info(f"Publishing Python package version {version}...")

            # Check if Python package directory exists
            python_dir = Path("extracted/python")
            if not python_dir.exists():
                error_msg = "Python package directory not found. Run 'lake exe extract_guard' first."
                self.logger.error(error_msg)
                self.publishing_stats["errors"].append(error_msg)
                return False

            # Update version in setup.py
            self._update_python_version(python_dir, version)

            # Build package
            self.logger.info("Building Python package...")
            build_result = self._build_python_package(python_dir)
            if not build_result:
                return False

            # Check if package already exists
            if not dry_run and self._package_exists_pypi("ds-guard", version):
                warning_msg = (
                    f"Package ds-guard version {version} already exists on PyPI"
                )
                self.logger.warning(warning_msg)
                self.publishing_stats["warnings"].append(warning_msg)
                return True

            # Publish to PyPI
            if not dry_run:
                self.logger.info("Publishing to PyPI...")
                publish_result = self._publish_to_pypi(python_dir)
                if not publish_result:
                    return False
            else:
                self.logger.info("DRY RUN: Would publish to PyPI")

            self.publishing_stats["python_published"] = True
            self.publishing_stats["python_version"] = version

            self.logger.info(f"✓ Python package {version} published successfully!")
            return True

        except Exception as e:
            error_msg = f"Failed to publish Python package: {str(e)}"
            self.logger.error(error_msg)
            self.publishing_stats["errors"].append(error_msg)
            return False

    def publish_rust_package(self, version: str, dry_run: bool = False) -> bool:
        """Publish Rust package to crates.io."""
        try:
            self.logger.info(f"Publishing Rust package version {version}...")

            # Check if Rust package directory exists
            rust_dir = Path("extracted/rust")
            if not rust_dir.exists():
                error_msg = "Rust package directory not found. Run 'lake exe extract_guard' first."
                self.logger.error(error_msg)
                self.publishing_stats["errors"].append(error_msg)
                return False

            # Update version in Cargo.toml
            self._update_rust_version(rust_dir, version)

            # Build package
            self.logger.info("Building Rust package...")
            build_result = self._build_rust_package(rust_dir)
            if not build_result:
                return False

            # Check if package already exists
            if not dry_run and self._package_exists_crates("ds-guard", version):
                warning_msg = (
                    f"Package ds-guard version {version} already exists on crates.io"
                )
                self.logger.warning(warning_msg)
                self.publishing_stats["warnings"].append(warning_msg)
                return True

            # Publish to crates.io
            if not dry_run:
                self.logger.info("Publishing to crates.io...")
                publish_result = self._publish_to_crates(rust_dir)
                if not publish_result:
                    return False
            else:
                self.logger.info("DRY RUN: Would publish to crates.io")

            self.publishing_stats["rust_published"] = True
            self.publishing_stats["rust_version"] = version

            self.logger.info(f"✓ Rust package {version} published successfully!")
            return True

        except Exception as e:
            error_msg = f"Failed to publish Rust package: {str(e)}"
            self.logger.error(error_msg)
            self.publishing_stats["errors"].append(error_msg)
            return False

    def _update_python_version(self, package_dir: Path, version: str):
        """Update version in Python package files."""
        setup_py_path = package_dir / "setup.py"
        if setup_py_path.exists():
            with open(setup_py_path, "r") as f:
                content = f.read()

            # Update version in setup.py
            import re

            content = re.sub(r'version="[^"]*"', f'version="{version}"', content)

            with open(setup_py_path, "w") as f:
                f.write(content)

        # Update pyproject.toml if it exists
        pyproject_path = package_dir / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "r") as f:
                content = f.read()

            import re

            content = re.sub(r'version = "[^"]*"', f'version = "{version}"', content)

            with open(pyproject_path, "w") as f:
                f.write(content)

    def _update_rust_version(self, package_dir: Path, version: str):
        """Update version in Rust package files."""
        cargo_toml_path = package_dir / "Cargo.toml"
        if cargo_toml_path.exists():
            with open(cargo_toml_path, "r") as f:
                content = f.read()

            import re

            content = re.sub(r'version = "[^"]*"', f'version = "{version}"', content)

            with open(cargo_toml_path, "w") as f:
                f.write(content)

    def _build_python_package(self, package_dir: Path) -> bool:
        """Build Python package."""
        try:
            # Change to package directory
            original_dir = os.getcwd()
            os.chdir(package_dir)

            # Clean previous builds
            subprocess.run(
                ["python", "setup.py", "clean", "--all"],
                capture_output=True,
                check=False,
            )

            # Try modern build first
            try:
                result = subprocess.run(
                    ["python", "-m", "build"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fallback to setup.py
                result = subprocess.run(
                    ["python", "setup.py", "sdist", "bdist_wheel"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

            # Restore original directory
            os.chdir(original_dir)

            if result.returncode != 0:
                self.logger.error(f"Python build failed: {result.stderr}")
                return False

            self.logger.info("✓ Python package built successfully")
            return True

        except Exception as e:
            self.logger.error(f"Python build error: {e}")
            return False

    def _build_rust_package(self, package_dir: Path) -> bool:
        """Build Rust package."""
        try:
            # Change to package directory
            original_dir = os.getcwd()
            os.chdir(package_dir)

            # Check if Cargo.toml exists
            if not (package_dir / "Cargo.toml").exists():
                self.logger.error("Cargo.toml not found")
                return False

            # Build package with timeout
            result = subprocess.run(
                ["cargo", "build", "--release"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            # Restore original directory
            os.chdir(original_dir)

            if result.returncode != 0:
                self.logger.error(f"Rust build failed: {result.stderr}")
                return False

            self.logger.info("✓ Rust package built successfully")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("Rust build timed out")
            return False
        except Exception as e:
            self.logger.error(f"Rust build error: {e}")
            return False

    def _package_exists_pypi(self, package_name: str, version: str) -> bool:
        """Check if package version exists on PyPI."""
        if not REQUESTS_AVAILABLE:
            return False

        try:
            url = f"https://pypi.org/pypi/{package_name}/{version}/json"
            response = requests.get(url)
            return response.status_code == 200
        except Exception:
            return False

    def _package_exists_crates(self, package_name: str, version: str) -> bool:
        """Check if package version exists on crates.io."""
        if not REQUESTS_AVAILABLE:
            return False

        try:
            url = f"https://crates.io/api/v1/crates/{package_name}/{version}"
            response = requests.get(url)
            return response.status_code == 200
        except Exception:
            return False

    def _publish_to_pypi(self, package_dir: Path) -> bool:
        """Publish Python package to PyPI."""
        try:
            # Change to package directory
            original_dir = os.getcwd()
            os.chdir(package_dir)

            # Publish using twine
            result = subprocess.run(
                ["twine", "upload", "dist/*"], capture_output=True, text=True
            )

            # Restore original directory
            os.chdir(original_dir)

            if result.returncode != 0:
                self.logger.error(f"PyPI upload failed: {result.stderr}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"PyPI upload error: {e}")
            return False

    def _publish_to_crates(self, package_dir: Path) -> bool:
        """Publish Rust package to crates.io."""
        try:
            # Change to package directory
            original_dir = os.getcwd()
            os.chdir(package_dir)

            # Publish using cargo
            result = subprocess.run(
                ["cargo", "publish"], capture_output=True, text=True
            )

            # Restore original directory
            os.chdir(original_dir)

            if result.returncode != 0:
                self.logger.error(f"crates.io upload failed: {result.stderr}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"crates.io upload error: {e}")
            return False

    def generate_publishing_report(self) -> Dict[str, Any]:
        """Generate publishing report."""
        return {
            "timestamp": time.time(),
            "python_published": self.publishing_stats["python_published"],
            "rust_published": self.publishing_stats["rust_published"],
            "python_version": self.publishing_stats["python_version"],
            "rust_version": self.publishing_stats["rust_version"],
            "errors": self.publishing_stats["errors"],
            "warnings": self.publishing_stats["warnings"],
            "success": (
                self.publishing_stats["python_published"]
                or self.publishing_stats["rust_published"]
            ),
        }

    def create_github_release(self, version: str, dry_run: bool = False) -> bool:
        """Create GitHub release with package artifacts."""
        try:
            self.logger.info(f"Creating GitHub release for version {version}...")

            # This would integrate with GitHub API to create releases
            # For now, just log the action
            if not dry_run:
                self.logger.info("Would create GitHub release with:")
                self.logger.info(f"  - Python package: ds-guard-{version}")
                self.logger.info(f"  - Rust package: ds-guard-{version}")
            else:
                self.logger.info("DRY RUN: Would create GitHub release")

            return True

        except Exception as e:
            self.logger.error(f"GitHub release creation failed: {e}")
            return False


def publish_packages(
    version: str,
    publish_python: bool = False,
    publish_rust: bool = False,
    dry_run: bool = False,
) -> bool:
    """Convenience function to publish packages."""
    publisher = PackagePublisher()

    success = True

    if publish_python:
        if not publisher.publish_python_package(version, dry_run):
            success = False

    if publish_rust:
        if not publisher.publish_rust_package(version, dry_run):
            success = False

    # Generate report
    report = publisher.generate_publishing_report()

    if success:
        print("✓ Package publishing completed successfully!")
        if report["python_published"]:
            print(f"  Python: ds-guard {report['python_version']} published to PyPI")
        if report["rust_published"]:
            print(f"  Rust: ds-guard {report['rust_version']} published to crates.io")
    else:
        print("✗ Package publishing failed!")
        for error in report["errors"]:
            print(f"  Error: {error}")

    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  Warning: {warning}")

    return success


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Package Publishing Automation")
    parser.add_argument(
        "--version", type=str, required=True, help="Package version to publish"
    )
    parser.add_argument(
        "--publish-python", action="store_true", help="Publish Python package to PyPI"
    )
    parser.add_argument(
        "--publish-rust", action="store_true", help="Publish Rust package to crates.io"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run (don't actually publish)"
    )
    parser.add_argument(
        "--github-release", action="store_true", help="Create GitHub release"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.publish_python and not args.publish_rust:
        print("Error: Must specify --publish-python and/or --publish-rust")
        sys.exit(1)

    # Publish packages
    success = publish_packages(
        version=args.version,
        publish_python=args.publish_python,
        publish_rust=args.publish_rust,
        dry_run=args.dry_run,
    )

    # Create GitHub release if requested
    if args.github_release and success:
        publisher = PackagePublisher()
        publisher.create_github_release(args.version, args.dry_run)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
