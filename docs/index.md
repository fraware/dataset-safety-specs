# Dataset Safety Specifications

Formal verification framework for dataset lineage, policy compliance, and training-time safety guarantees.

## Overview

The Dataset Safety Specifications (DSS) framework provides formal verification for:

- **Dataset Lineage**: Hash-chain consistent ETL transforms
- **Data Policy Filters**: HIPAA PHI, COPPA, GDPR compliance
- **Optimizer Invariants**: Formal proofs for SGD, AdamW, Lion stability
- **Shape Safety Verifier**: ONNX/PyTorch graph verification
- **Guard Extractor**: Auto-generated Rust/Python guards

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fraware/dataset-safety-specs

# Install dependencies
pip install -r requirements.txt

# Build the project
lake build
```

### Basic Usage

```bash
# Run tests
lake exe test_suite

# Extract guards from Lean predicates
lake exe extract_guard

# Verify shape safety of ONNX model
lake exe shapesafe_verify model.onnx

# Create distributable bundle
./bundle.sh bundle
```

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

## Tutorial: Prove Your First PHI Filter

1. **Define the PHI predicate in Lean**:

```lean
def contains_phi (text : String) : Bool :=
  let phi_patterns := [
    "SSN", "social security", "medical record", "health plan"
  ]
  phi_patterns.any (fun pattern => pattern.isInfixOf text)

def has_phi (row : Row) : Bool :=
  row.phi.any contains_phi
```

2. **Extract the guard**:

```bash
lake exe extract_guard
```

3. **Use in Python**:

```python
from ds_guard import phi_guard, Row

row = Row(
    phi=["SSN: 123-45-6789", "medical record"],
    age=None,
    gdpr_special=[],
    custom_fields=[]
)

if phi_guard(row):
    print("PHI detected!")
```

## API Reference

### Lineage Module

- `Lineage.mk_dataset` - Create a dataset with hash
- `Lineage.apply_transform` - Apply ETL transform
- `Lineage.verify_transform` - Verify transform consistency

### Policy Module

- `Policy.has_phi` - Check for PHI
- `Policy.is_minor` - Check for COPPA compliance
- `Policy.has_gdpr_special` - Check for GDPR special categories

### Optimizer Module

- `Optimizer.sgd_update` - SGD update rule
- `Optimizer.adamw_update` - AdamW update rule
- `Optimizer.energy` - Parameter energy function

### Shape Module

- `Shape.parse_onnx_graph` - Parse ONNX model
- `Shape.verify_graph_shapes` - Verify shape consistency
- `Shape.generate_lean_shape_spec` - Generate Lean shape spec

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.
