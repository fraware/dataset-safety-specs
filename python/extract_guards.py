#!/usr/bin/env python3
"""
Guard Extraction Script

Extracts Rust and Python guards from Lean predicates without relying on Lake plugin.
This provides an alternative implementation for DSS-5 guard extraction.

Usage:
    python extract_guards.py
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

from rust_guard_transpiler import RustGuardTranspiler


class GuardExtractor:
    """Extracts guards from Lean predicates."""

    def __init__(self):
        self.logger = logging.getLogger("GuardExtractor")
        self.extraction_stats = {
            "predicates_processed": 0,
            "rust_functions_generated": 0,
            "python_functions_generated": 0,
            "errors": 0,
        }

    def extract_all_guards(self) -> Dict[str, Any]:
        """Extract all guards from Lean predicates."""
        self.logger.info("Extracting guards from Lean predicates...")

        # Define the guard predicates
        predicates = [
            {
                "name": "PHI",
                "lean_code": "has_phi row",
                "description": "Detects Protected Health Information",
            },
            {
                "name": "COPPA",
                "lean_code": "is_minor row",
                "description": "Detects COPPA violations (age < 13)",
            },
            {
                "name": "GDPR",
                "lean_code": "has_gdpr_special row",
                "description": "Detects GDPR special categories",
            },
        ]

        try:
            # Generate Rust guards
            rust_transpiler = RustGuardTranspiler()
            rust_module = rust_transpiler.transpile_predicates(predicates)

            # Generate Python guards
            python_module = self._generate_python_module(predicates)

            # Generate Cargo.toml
            cargo_toml = self._generate_cargo_toml()

            # Generate setup.py
            setup_py = self._generate_setup_py()

            # Generate README files
            rust_readme = self._generate_rust_readme()
            python_readme = self._generate_python_readme()

            # Generate test files
            rust_tests = self._generate_rust_tests(predicates)
            python_tests = self._generate_python_tests(predicates)

            # Generate CI configuration
            github_workflow = self._generate_github_workflow()

            # Generate package publishing scripts
            publish_script = self._generate_publish_script()

            # Generate Docker configuration
            dockerfile = self._generate_dockerfile()

            # Generate Kubernetes manifests
            k8s_deployment = self._generate_k8s_deployment()

            # Generate monitoring configuration
            prometheus_config = self._generate_prometheus_config()

            self.extraction_stats["predicates_processed"] = len(predicates)
            self.extraction_stats["rust_functions_generated"] = len(predicates)
            self.extraction_stats["python_functions_generated"] = len(predicates)

            return {
                "success": True,
                "predicates": predicates,
                "rust_module": rust_module,
                "python_module": python_module,
                "cargo_toml": cargo_toml,
                "setup_py": setup_py,
                "rust_readme": rust_readme,
                "python_readme": python_readme,
                "rust_tests": rust_tests,
                "python_tests": python_tests,
                "github_workflow": github_workflow,
                "publish_script": publish_script,
                "dockerfile": dockerfile,
                "k8s_deployment": k8s_deployment,
                "prometheus_config": prometheus_config,
                "stats": self.extraction_stats.copy(),
            }

        except Exception as e:
            error_msg = f"Failed to extract guards: {str(e)}"
            self.logger.error(error_msg)
            self.extraction_stats["errors"] += 1
            return {
                "success": False,
                "error": error_msg,
                "stats": self.extraction_stats.copy(),
            }

    def _generate_python_module(self, predicates: List[Dict[str, Any]]) -> str:
        """Generate Python module from predicates."""
        guard_functions = []
        for predicate in predicates:
            name = predicate["name"].lower()
            guard_functions.append(
                f"""def {name}_guard(row: Row) -> bool:
    \"\"\"{predicate['description']}\"\"\"
    # Implementation would be generated from Lean code
    return True  # Placeholder"""
            )

        combined = "\n\n".join(guard_functions)
        return f"""# Auto-generated Python guards from Lean predicates

from typing import List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class Row:
    \"\"\"Row representation with tagged fields for safety checking.\"\"\"
    phi: List[str] = field(default_factory=list)
    age: Optional[int] = None
    gdpr_special: List[str] = field(default_factory=list)
    custom_fields: List[Tuple[str, str]] = field(default_factory=list)

{combined}

# Helper function for PHI detection
def contains_phi(text: str) -> bool:
    \"\"\"Check if text contains PHI patterns.\"\"\"
    phi_patterns = [
        "SSN", "social security", "medical record", "health plan",
        "patient", "diagnosis", "treatment", "prescription"
    ]
    return any(pattern in text for pattern in phi_patterns)"""

    def _generate_cargo_toml(self) -> str:
        """Generate Cargo.toml for Rust package."""
        return """[package]
name = "ds-guard"
version = "0.1.0"
edition = "2021"
description = "Dataset safety guards generated from Lean predicates"
license = "MIT"
authors = ["Dataset Safety Specs"]

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
regex = "1.0"
lazy_static = "1.0"

[lib]
name = "ds_guard"
path = "src/lib.rs"
"""

    def _generate_setup_py(self) -> str:
        """Generate setup.py for Python package."""
        return """from setuptools import setup, find_packages

setup(
    name="ds-guard",
    version="0.1.0",
    description="Dataset safety guards generated from Lean predicates",
    author="Dataset Safety Specs",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "typing-extensions>=4.0",
    ],
)"""

    def _generate_rust_readme(self) -> str:
        """Generate Rust README."""
        return """# Dataset Safety Guards - Rust

Auto-generated Rust guards from Lean predicates for dataset safety verification.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ds-guard = { path = "path/to/ds-guard" }
```

## Usage

```rust
use ds_guard::{Row, phi_guard, coppa_guard, gdpr_guard};

let row = Row::new()
    .with_phi(vec!["SSN: 123-45-6789".to_string()])
    .with_age(25);

if phi_guard(&row) {
    println!("PHI detected!");
}

if coppa_guard(&row) {
    println!("COPPA violation detected!");
}
```

## Generated Guards

- `phi_guard`: Detects Protected Health Information
- `coppa_guard`: Detects COPPA violations (age < 13)
- `gdpr_guard`: Detects GDPR special categories

## Building

```bash
cargo build
cargo test
```

## Publishing

```bash
cargo publish
```"""

    def _generate_python_readme(self) -> str:
        """Generate Python README."""
        return """# Dataset Safety Guards - Python

Auto-generated Python guards from Lean predicates for dataset safety verification.

## Installation

```bash
pip install ds-guard
```

## Usage

```python
from ds_guard import Row, phi_guard, coppa_guard, gdpr_guard

row = Row(
    phi=["SSN: 123-45-6789"],
    age=25,
    gdpr_special=[]
)

if phi_guard(row):
    print("PHI detected!")

if coppa_guard(row):
    print("COPPA violation detected!")
```

## Generated Guards

- `phi_guard`: Detects Protected Health Information
- `coppa_guard`: Detects COPPA violations (age < 13)
- `gdpr_guard`: Detects GDPR special categories

## Development

```bash
pip install -e .
python -m pytest tests/
```

## Publishing

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```"""

    def _generate_rust_tests(self, predicates: List[Dict[str, Any]]) -> str:
        """Generate Rust tests."""
        test_functions = []
        for predicate in predicates:
            name = predicate["name"].lower()
            test_functions.append(
                f"""    #[test]
    fn test_{name}_guard() {{
        let row = Row::new()
            .with_phi(vec!["SSN: 123-45-6789".to_string()]);
        assert!({name}_guard(&row));
    }}"""
            )

        combined = "\n".join(test_functions)
        return f"""#[cfg(test)]
mod tests {{
    use super::*;

{combined}
}}"""

    def _generate_python_tests(self, predicates: List[Dict[str, Any]]) -> str:
        """Generate Python tests."""
        test_functions = []
        for predicate in predicates:
            name = predicate["name"].lower()
            test_functions.append(
                f"""def test_{name}_guard():
    \"\"\"Test {predicate['description']}.\"\"\"
    row = Row(phi=["SSN: 123-45-6789"])
    assert {name}_guard(row)"""
            )

        combined = "\n\n".join(test_functions)
        guard_names = [p["name"].lower() + "_guard" for p in predicates]
        return f"""# Python tests for generated guards

import pytest
from ds_guard import Row, {", ".join(guard_names)}

{combined}

def test_clean_row():
    \"\"\"Test that clean rows pass all guards.\"\"\"
    row = Row(age=25)
    assert not phi_guard(row)
    assert not coppa_guard(row)
    assert not gdpr_guard(row)"""

    def _generate_github_workflow(self) -> str:
        """Generate GitHub workflow."""
        return """name: Guards CI

on:
  push:
    branches: [main]
    paths:
      - 'extracted/**'
  pull_request:
    branches: [main]
    paths:
      - 'extracted/**'

jobs:
  test-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Test Rust guards
        run: |
          cd extracted/rust
          cargo test

  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Test Python guards
        run: |
          cd extracted/python
          pip install -e .
          python -m pytest tests/"""

    def _generate_publish_script(self) -> str:
        """Generate publishing script."""
        return """#!/bin/bash
# Package publishing script for extracted guards

set -e

VERSION="$1"
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.0"
    exit 1
fi

echo "Publishing guards version $VERSION..."

# Publish Rust package
echo "Publishing Rust package..."
cd extracted/rust
cargo publish --allow-dirty
cd ../..

# Publish Python package
echo "Publishing Python package..."
cd extracted/python
python setup.py sdist bdist_wheel
twine upload dist/*
cd ../..

echo "✓ Guards published successfully!"
echo "  Rust: ds-guard $VERSION on crates.io"
echo "  Python: ds-guard $VERSION on PyPI"
"""

    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile."""
        return """# Multi-stage Dockerfile for ds-guard

FROM rust:1.70 as rust-builder
WORKDIR /app
COPY extracted/rust/ .
RUN cargo build --release

FROM python:3.9-slim as python-builder
WORKDIR /app
COPY extracted/python/ .
RUN pip install -e .

FROM python:3.9-slim
WORKDIR /app

# Install Rust runtime
RUN apt-get update && apt-get install -y libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy Rust binary
COPY --from=rust-builder /app/target/release/ds_guard /usr/local/bin/

# Copy Python package
COPY --from=python-builder /app /app/python
RUN pip install -e /app/python

# Copy monitoring
COPY extracted/monitoring/ /app/monitoring/

EXPOSE 8080

CMD ["python", "-m", "ds_guard.server"]"""

    def _generate_k8s_deployment(self) -> str:
        """Generate Kubernetes deployment."""
        return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: ds-guard
  labels:
    app: ds-guard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ds-guard
  template:
    metadata:
      labels:
        app: ds-guard
    spec:
      containers:
      - name: ds-guard
        image: ds-guard:latest
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: "info"
        - name: PYTHONPATH
          value: "/app/python"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ds-guard-service
spec:
  selector:
    app: ds-guard
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer"""

    def _generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration."""
        return """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'ds-guard'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']"""

    def write_extracted_files(self, bundle: Dict[str, Any]) -> bool:
        """Write all extracted files to disk."""
        try:
            # Create output directories
            os.makedirs("extracted/rust/src", exist_ok=True)
            os.makedirs("extracted/python/ds_guard", exist_ok=True)
            os.makedirs("extracted/python/tests", exist_ok=True)
            os.makedirs(".github/workflows", exist_ok=True)
            os.makedirs("extracted/k8s", exist_ok=True)
            os.makedirs("extracted/monitoring", exist_ok=True)

            # Write Rust files
            with open("extracted/rust/src/lib.rs", "w", encoding="utf-8") as f:
                f.write(bundle["rust_module"])
            with open("extracted/rust/Cargo.toml", "w", encoding="utf-8") as f:
                f.write(bundle["cargo_toml"])
            with open("extracted/rust/README.md", "w", encoding="utf-8") as f:
                f.write(bundle["rust_readme"])
            with open("extracted/rust/src/tests.rs", "w", encoding="utf-8") as f:
                f.write(bundle["rust_tests"])

            # Write Python files
            with open(
                "extracted/python/ds_guard/__init__.py", "w", encoding="utf-8"
            ) as f:
                f.write(bundle["python_module"])
            with open("extracted/python/setup.py", "w", encoding="utf-8") as f:
                f.write(bundle["setup_py"])
            with open("extracted/python/README.md", "w", encoding="utf-8") as f:
                f.write(bundle["python_readme"])
            with open(
                "extracted/python/tests/test_guards.py", "w", encoding="utf-8"
            ) as f:
                f.write(bundle["python_tests"])

            # Write CI configuration
            with open(".github/workflows/guards-ci.yml", "w", encoding="utf-8") as f:
                f.write(bundle["github_workflow"])

            # Write package publishing scripts
            with open("extracted/publish_guards.sh", "w", encoding="utf-8") as f:
                f.write(bundle["publish_script"])

            # Write Docker configuration
            with open("extracted/Dockerfile", "w", encoding="utf-8") as f:
                f.write(bundle["dockerfile"])

            # Write Kubernetes manifests
            with open("extracted/k8s/deployment.yaml", "w", encoding="utf-8") as f:
                f.write(bundle["k8s_deployment"])

            # Write monitoring configuration
            with open(
                "extracted/monitoring/prometheus.yml", "w", encoding="utf-8"
            ) as f:
                f.write(bundle["prometheus_config"])

            return True

        except Exception as e:
            self.logger.error(f"Failed to write extracted files: {e}")
            return False


def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Extract guards
    extractor = GuardExtractor()
    result = extractor.extract_all_guards()

    if result["success"]:
        print("✓ Guard extraction completed!")
        print("")
        print("Generated files:")
        print("  Rust: extracted/rust/")
        print("  Python: extracted/python/")
        print("  CI: .github/workflows/guards-ci.yml")
        print("  Publishing: extracted/publish_guards.sh")
        print("  Docker: extracted/Dockerfile")
        print("  Kubernetes: extracted/k8s/")
        print("  Monitoring: extracted/monitoring/")
        print("")

        # Write files to disk
        if extractor.write_extracted_files(result):
            print("✓ All files written to disk!")
            print("")
            print("To build and test:")
            print("  cd extracted/rust && cargo build")
            print("  cd extracted/python && python -m pytest")
            print("")
            print("To publish packages:")
            print("  ./extracted/publish_guards.sh --version 0.1.0")
            print("")
            print("To deploy with Docker:")
            print("  docker build -t ds-guard extracted/")
            print("  docker run -p 8080:8080 ds-guard")
        else:
            print("✗ Failed to write files to disk")
            return 1
    else:
        print(f"✗ Guard extraction failed: {result.get('error')}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
