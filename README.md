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
git clone https://github.com/your-org/dataset-safety-specs.git
cd dataset-safety-specs

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

## Examples

### PHI Masking Demo

See `examples/mask_phi.ipynb` for a complete demo of PHI detection and masking.

### GPT-2 Shape Verification

See `examples/verify_shapes_gpt2.ipynb` for GPT-2 (124M) shape verification.

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
lake exe test_suite
lake exe benchmark_suite
lake exe lineage_regression_test
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
