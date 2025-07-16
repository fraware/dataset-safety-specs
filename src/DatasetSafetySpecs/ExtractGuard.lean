/-
# Guard Extractor Lake Plugin

Lake plugin for extracting Rust and Python guards from Lean predicates.
Implements DSS-5: ETL Guard Extractor.

Usage:
    lake exe extract_guard
-/

import DatasetSafetySpecs.Guard
import System.IO
import System.Environment
import System.FilePath
import System.Directory

def main : IO Unit := do
  IO.println "Dataset Safety Specs - Guard Extractor"
  IO.println "====================================="

  -- Create output directories
  IO.println "Creating output directories..."
  IO.FS.createDirAll "extracted/rust/src"
  IO.FS.createDirAll "extracted/python/ds_guard"
  IO.FS.createDirAll "extracted/python/tests"

  -- Extract guard bundle
  IO.println "Extracting guards from Lean predicates..."
  let bundle := Guard.extract_all_guards

  -- Generate Rust files
  IO.println "Generating Rust guard module..."
  IO.FS.writeFile "extracted/rust/src/lib.rs" bundle.rust_module
  IO.FS.writeFile "extracted/rust/Cargo.toml" bundle.cargo_toml

  -- Generate Python files
  IO.println "Generating Python guard module..."
  IO.FS.writeFile "extracted/python/ds_guard/__init__.py" bundle.python_module
  IO.FS.writeFile "extracted/python/setup.py" bundle.setup_py

  -- Generate Rust types
  IO.println "Generating Rust type definitions..."
  let rust_types := generate_rust_types
  IO.FS.writeFile "extracted/rust/src/types.rs" rust_types

  -- Generate Python types
  IO.println "Generating Python type definitions..."
  let python_types := generate_python_types
  IO.FS.writeFile "extracted/python/ds_guard/types.py" python_types

  -- Generate README files
  IO.println "Generating documentation..."
  let rust_readme := generate_rust_readme
  IO.FS.writeFile "extracted/rust/README.md" rust_readme

  let python_readme := generate_python_readme
  IO.FS.writeFile "extracted/python/README.md" python_readme

  -- Generate test files
  IO.println "Generating test files..."
  let rust_tests := generate_rust_tests bundle.predicates
  IO.FS.writeFile "extracted/rust/src/tests.rs" rust_tests

  let python_tests := generate_python_tests bundle.predicates
  IO.FS.writeFile "extracted/python/tests/test_guards.py" python_tests

  -- Generate CI configuration
  IO.println "Generating CI configuration..."
  let github_workflow := generate_github_workflow
  IO.FS.createDirAll ".github/workflows"
  IO.FS.writeFile ".github/workflows/guards-ci.yml" github_workflow

  -- Generate package publishing scripts
  IO.println "Generating publishing scripts..."
  let publish_script := generate_publish_script
  IO.FS.writeFile "extracted/publish_guards.sh" publish_script

  -- Generate Docker configuration
  IO.println "Generating Docker configuration..."
  let dockerfile := generate_dockerfile
  IO.FS.writeFile "extracted/Dockerfile" dockerfile

  -- Generate Kubernetes manifests
  IO.println "Generating Kubernetes manifests..."
  let k8s_deployment := generate_k8s_deployment
  IO.FS.createDirAll "extracted/k8s"
  IO.FS.writeFile "extracted/k8s/deployment.yaml" k8s_deployment

  -- Generate monitoring configuration
  IO.println "Generating monitoring configuration..."
  let prometheus_config := generate_prometheus_config
  IO.FS.writeFile "extracted/monitoring/prometheus.yml" prometheus_config

  IO.println "✓ Guard extraction completed!"
  IO.println ""
  IO.println "Generated files:"
  IO.println "  Rust: extracted/rust/"
  IO.println "  Python: extracted/python/"
  IO.println "  CI: .github/workflows/guards-ci.yml"
  IO.println "  Publishing: extracted/publish_guards.sh"
  IO.println "  Docker: extracted/Dockerfile"
  IO.println "  Kubernetes: extracted/k8s/"
  IO.println "  Monitoring: extracted/monitoring/"
  IO.println ""
  IO.println "To build and test:"
  IO.println "  cd extracted/rust && cargo build"
  IO.println "  cd extracted/python && python -m pytest"
  IO.println ""
  IO.println "To publish packages:"
  IO.println "  ./extracted/publish_guards.sh --version 0.1.0"
  IO.println ""
  IO.println "To deploy with Docker:"
  IO.println "  docker build -t ds-guard extracted/"
  IO.println "  docker run -p 8080:8080 ds-guard"

def generate_rust_types : String :=
"-- Rust type definitions for dataset safety guards

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub phi: Vec<String>,
    pub age: Option<u32>,
    pub gdpr_special: Vec<String>,
    pub custom_fields: Vec<(String, String)>,
}

impl Row {
    pub fn new() -> Self {
        Self {
            phi: Vec::new(),
            age: None,
            gdpr_special: Vec::new(),
            custom_fields: Vec::new(),
        }
    }

    pub fn with_phi(mut self, phi: Vec<String>) -> Self {
        self.phi = phi;
        self
    }

    pub fn with_age(mut self, age: u32) -> Self {
        self.age = Some(age);
        self
    }

    pub fn with_gdpr_special(mut self, gdpr: Vec<String>) -> Self {
        self.gdpr_special = gdpr;
        self
    }

    pub fn with_custom_fields(mut self, fields: Vec<(String, String)>) -> Self {
        self.custom_fields = fields;
        self
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}"

def generate_python_types : String :=
"""# Python type definitions for dataset safety guards

from typing import List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class Row:
    \"\"\"Row representation with tagged fields for safety checking.\"\"\"

    phi: List[str] = field(default_factory=list)
    age: Optional[int] = None
    gdpr_special: List[str] = field(default_factory=list)
    custom_fields: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        \"\"\"Validate row data after initialization.\"\"\"
        if self.age is not None and self.age < 0:
            raise ValueError(\"Age cannot be negative\")

    def has_phi(self) -> bool:
        \"\"\"Check if row contains PHI data.\"\"\"
        return len(self.phi) > 0

    def is_minor(self) -> bool:
        \"\"\"Check if row contains minor data (COPPA).\"\"\"
        return self.age is not None and self.age < 13

    def has_gdpr_special(self) -> bool:
        \"\"\"Check if row contains GDPR special categories.\"\"\"
        return len(self.gdpr_special) > 0"""

def generate_rust_readme : String :=
"# Dataset Safety Guards - Rust

Auto-generated Rust guards from Lean predicates for dataset safety verification.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ds-guard = { path = \"path/to/ds-guard\" }
```

## Usage

```rust
use ds_guard::{Row, phi_guard, coppa_guard, gdpr_guard};

let row = Row::new()
    .with_phi(vec![\"SSN: 123-45-6789\".to_string()])
    .with_age(25);

if phi_guard(&row) {
    println!(\"PHI detected!\");
}

if coppa_guard(&row) {
    println!(\"COPPA violation detected!\");
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
```"

def generate_python_readme : String :=
"""# Dataset Safety Guards - Python

Auto-generated Python guards from Lean predicates for dataset safety verification.

## Installation

```bash
pip install ds-guard
```

## Usage

```python
from ds_guard import Row, phi_guard, coppa_guard, gdpr_guard

row = Row(
    phi=[\"SSN: 123-45-6789\"],
    age=25,
    gdpr_special=[]
)

if phi_guard(row):
    print(\"PHI detected!\")

if coppa_guard(row):
    print(\"COPPA violation detected!\")
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

def generate_rust_tests (predicates : List Guard.Predicate) : String :=
"-- Rust tests for generated guards

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_guard() {
        let row = Row::new()
            .with_phi(vec![\"SSN: 123-45-6789\".to_string()]);
        assert!(phi_guard(&row));
    }

    #[test]
    fn test_coppa_guard() {
        let row = Row::new().with_age(12);
        assert!(coppa_guard(&row));
    }

    #[test]
    fn test_gdpr_guard() {
        let row = Row::new()
            .with_gdpr_special(vec![\"health\".to_string()]);
        assert!(gdpr_guard(&row));
    }
}"

def generate_python_tests (predicates : List Guard.Predicate) : String :=
"""# Python tests for generated guards

import pytest
from ds_guard import Row, phi_guard, coppa_guard, gdpr_guard

def test_phi_guard():
    \"\"\"Test PHI detection.\"\"\"
    row = Row(phi=[\"SSN: 123-45-6789\"])
    assert phi_guard(row)

def test_coppa_guard():
    \"\"\"Test COPPA violation detection.\"\"\"
    row = Row(age=12)
    assert coppa_guard(row)

def test_gdpr_guard():
    \"\"\"Test GDPR special category detection.\"\"\"
    row = Row(gdpr_special=[\"health\"])
    assert gdpr_guard(row)

def test_clean_row():
    \"\"\"Test that clean rows pass all guards.\"\"\"
    row = Row(age=25)
    assert not phi_guard(row)
    assert not coppa_guard(row)
    assert not gdpr_guard(row)"""

def generate_github_workflow : String :=
"name: Guards CI

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
          python -m pytest tests/"

def generate_publish_script : String :=
"#!/bin/bash
# Package publishing script for extracted guards

set -e

VERSION=\"$1\"
if [ -z \"$VERSION\" ]; then
    echo \"Usage: $0 <version>\"
    echo \"Example: $0 0.1.0\"
    exit 1
fi

echo \"Publishing guards version $VERSION...\"

# Publish Rust package
echo \"Publishing Rust package...\"
cd extracted/rust
cargo publish --allow-dirty
cd ../..

# Publish Python package
echo \"Publishing Python package...\"
cd extracted/python
python setup.py sdist bdist_wheel
twine upload dist/*
cd ../..

echo \"✓ Guards published successfully!\"
echo \"  Rust: ds-guard $VERSION on crates.io\"
echo \"  Python: ds-guard $VERSION on PyPI\""

def generate_dockerfile : String :=
"# Multi-stage Dockerfile for ds-guard

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

CMD [\"python\", \"-m\", \"ds_guard.server\"]"

def generate_k8s_deployment : String :=
"apiVersion: apps/v1
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
          value: \"info\"
        - name: PYTHONPATH
          value: \"/app/python\"
        resources:
          requests:
            memory: \"256Mi\"
            cpu: \"250m\"
          limits:
            memory: \"512Mi\"
            cpu: \"500m\"
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
  type: LoadBalancer"

def generate_prometheus_config : String :=
"global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - \"first_rules.yml\"
  # - \"second_rules.yml\"

scrape_configs:
  - job_name: 'ds-guard'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']"

end ExtractGuard
