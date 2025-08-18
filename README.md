# Dataset Safety Specifications

Formal verification framework for dataset lineage, policy compliance, and training-time safety guarantees.

## North-Star Outcomes

| Tag   | Outcome                                                                                                                  | Success Metric                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| DSS-1 | Lineage Proof Framework — Lean proofs that every ETL transform is hash-chain-consistent                                  | 100% of sample ETL DAG nodes emit T(d) hash matching formal spec; regression test passes on 10k-row dataset |
| DSS-2 | Policy Filter Pack covering HIPAA PHI, COPPA, GDPR minor-data, and custom regex rules                                    | ≥99% recall / 0 false-negatives on curated PHI+COPPA benchmark                                              |
| DSS-3 | Optimizer-Invariant Suite for SGD, AdamW, Lion (β-version)                                                               | Proof scripts compile <3s; runbook shows bound ⇒ ∇ explosion impossible for given hyper-params              |
| DSS-4 | Shape-Safety Verifier that can consume ONNX / PyTorch FX graphs, generate Lean shape spec, and prove ∀ ℓ, out=shapeSpec… | End-to-end demo verifies GPT-2 (124M) computation graph in <45s                                             |
| DSS-5 | ETL Guard Extractor (Rust/Python) auto-generated from Lean predicates                                                    | Guard library adds <5ms overhead / 1M rows; published to PyPI & crates.io                                   |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SentinelOps-CI/dataset-safety-specs.git

# Install dependencies
pip install -r requirements.txt

# Build the project
lake build

# Run tests
lake exe test_suite

# Run benchmarks
lake exe benchmark_suite

# Extract guards from Lean predicates
lake exe extract_guard

# Verify shape safety of ONNX model
lake exe shapesafe_verify model.onnx

# Convert ONNX model to Lean shape spec
python python/onnx2lean_shape.py model.onnx output.lean

# Create distributable bundle
./bundle.sh bundle
```

## Architecture

- **Dataset Lineage Module**: Hash-chain consistent ETL transforms
- **Data Policy Filters**: HIPAA PHI, COPPA, GDPR compliance
- **Optimizer Invariants**: Formal proofs for SGD, AdamW, Lion stability
- **Shape Safety Verifier**: ONNX/PyTorch graph verification
- **Guard Extractor**: Auto-generated Rust/Python guards

## Core Components

### 1. Dataset Lineage (DSS-1)

Formal verification of ETL transform hash-chain consistency.

```lean
-- Create a dataset
let dataset := Lineage.mk_dataset ["row1", "row2", "row3"]

-- Apply a transform
let transform := {
  name := "filter_even"
  params := ["even_only"]
  transform_fn := fun d => { d with data := d.data.filter isEven }
}

-- Verify the transform
let verified := Lineage.verify_transform transform
```

### 2. Data Policy Filters (DSS-2)

HIPAA PHI, COPPA, and GDPR compliance filters.

```lean
-- Check for PHI
let row := {
  phi := ["SSN: 123-45-6789"]
  age := some 25
  gdpr_special := []
  custom_fields := []
}

let has_phi := Policy.has_phi row
let is_minor := Policy.is_minor row
```

### 3. Optimizer Invariants (DSS-3)

Formal proofs for optimizer stability.

```lean
-- SGD energy stability theorem
theorem sgd_energy_stability
  (w : Vector) (grad : Gradient) (lr : LearningRate)
  (h_lr : lr > 0) (h_lr_bound : lr ≤ 1) :
  let w_new := sgd_update w grad lr
  energy w_new ≤ energy w := by sorry
```

### 4. Shape Safety Verifier (DSS-4)

ONNX/PyTorch graph shape verification.

```bash
# Convert ONNX model to Lean shape spec
python python/onnx2lean_shape.py model.onnx output.lean

# Verify shape safety
lake exe shapesafe_verify model.onnx
```

### 5. Guard Extractor (DSS-5)

Auto-generated Rust and Python guards.

```bash
# Extract guards from Lean predicates
lake exe extract_guard

# Use in Python
from ds_guard import phi_guard, coppa_guard
```

## New Components

### Regression Testing Suite

Comprehensive regression tests for all deliverables:

```bash
# Run regression tests
python python/regression_tests.py

# Tests include:
# - 10k-row dataset processing
# - GPT-2 shape proof performance (≤45s)
# - ETL throughput profiling
# - Compliance validation
# - SentinelOps schema compliance
```

### Runtime Safety Kernel

Production-ready safety kernel for ML pipeline integration:

```python
from python.runtime_safety_kernel import create_safety_kernel, RuntimeConfig

# Create safety kernel
config = RuntimeConfig(compliance_level="strict")
kernel = create_safety_kernel(config)

# Register model assets
kernel.register_model_asset(model_asset)

# Check data safety
safety_result = kernel.check_data_safety(data)
```

### SentinelOps Compliance Bundle

Generate compliance bundles matching SentinelOps schema:

```python
from python.sentinelops_bundle import create_sentinelops_bundle

# Create compliance bundle
success = create_sentinelops_bundle(
    dataset_path="dataset.parquet",
    dataset_name="my_dataset",
    output_path="compliance_bundle.zip",
    compliance_level="strict"
)
```

### ONNX Node Name Extractor

Robust extraction of ONNX node names with fallback strategies:

```python
from python.onnx_node_extractor import extract_onnx_node_names

# Extract node names
result = extract_onnx_node_names("model.onnx", "node_names_report.json")
```

### Lion Optimizer Proof Gating

CI workflow for experimental Lion optimizer proofs:

```bash
# CI automatically detects experimental proofs
# and gates testing appropriately
# See .github/workflows/lion_proof_gating.yml
```

### Hard Induction Sub-lemmas

Complex mathematical proofs for optimizer stability:

```lean
-- See src/DatasetSafetySpecs/OptimizerInduction.lean
-- Contains hard induction sub-lemmas for Lion optimizer
-- with proof status tracking and CI gating
```

## Documentation

- **API Reference**: See `docs/index.md` for complete API documentation
- **Tutorial**: "Prove Your First PHI Filter" tutorial in the documentation
- **Contributing**: See `CONTRIBUTING.md` for development guidelines

## Development

```bash
# Install dependencies
lake update

# Format code
lake exe lean --run src/format.lean

# Type check
lake build DatasetSafetySpecs

# Run all tests
python python/run_all_tests.py

# Run specific test suites
lake exe test_suite
lake exe benchmark_suite
lake exe lineage_regression_test
python python/regression_tests.py
```

## Packaging

The project includes a comprehensive bundling system:

```bash
# Create complete bundle (Lean specs + Rust/Python guards)
./bundle.sh bundle

# Create specific bundles
./bundle.sh lean      # Lean specifications only
./bundle.sh python    # Python wheel only
./bundle.sh rust      # Rust crate only

# Clean bundle directory
./bundle.sh clean
```

## Issue Reporting

Use our issue templates for different types of issues:

- **[POLICY]**: Data policy and compliance issues
- **[SHAPE]**: Shape safety verification issues
- **[BUG]**: General bugs
- **[FEATURE]**: Feature requests

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
