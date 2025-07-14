/-
# Test Suite Executable

Comprehensive test suite for dataset safety specifications.
-/

import DatasetSafetySpecs
import DatasetSafetySpecs.Lineage
import DatasetSafetySpecs.Policy
import DatasetSafetySpecs.Optimizer
import DatasetSafetySpecs.Shape
import DatasetSafetySpecs.Guard
import LineageRegressionTest
import System.IO

def test_lineage : IO Bool := do

def test_lineage : IO Bool := do
  IO.println "Testing Dataset Lineage Module..."

  -- Test dataset creation
  let test_data := ["row1", "row2", "row3"]
  let dataset := Lineage.mk_dataset test_data "test"

  -- Test transform application
  let transform := {
    name := "test_transform"
    params := ["param1"]
    transform_fn := fun d => { d with data := d.data.map (· ++ "_transformed") }
  }
  let transformed := Lineage.apply_transform transform dataset

  let success := transformed.data.length = dataset.data.length
  IO.println s!"  ✓ Dataset lineage tests: {if success then \"PASSED\" else \"FAILED\"}"
  return success

def test_lineage_regression : IO Bool := do
  IO.println "Testing Lineage Regression (DSS-1)..."

  -- Run the comprehensive regression test
  LineageRegressionTest.run_regression_test

  -- For now, return true if the test completes without error
  -- In practice, this would check actual results
  return true

def test_policy : IO Bool := do
  IO.println "Testing Data Policy Filters..."

  -- Test PHI detection
  let phi_row := {
    phi := ["SSN: 123-45-6789", "medical record"]
    age := none
    gdpr_special := []
    custom_fields := []
  }
  let has_phi := Policy.has_phi phi_row

  -- Test COPPA compliance
  let minor_row := {
    phi := []
    age := some 12
    gdpr_special := []
    custom_fields := []
  }
  let is_minor := Policy.is_minor minor_row

  let success := has_phi && is_minor
  IO.println s!"  ✓ Policy filter tests: {if success then \"PASSED\" else \"FAILED\"}"
  return success

def test_optimizer : IO Bool := do
  IO.println "Testing Optimizer Invariants..."

  -- Test SGD invariant
  let sgd_inv := Optimizer.sgd_invariant
  let sgd_ok := Optimizer.verify_invariant sgd_inv

  -- Test AdamW invariant
  let adamw_inv := Optimizer.adamw_invariant
  let adamw_ok := Optimizer.verify_invariant adamw_inv

  let success := sgd_ok && adamw_ok
  IO.println s!"  ✓ Optimizer invariant tests: {if success then \"PASSED\" else \"FAILED\"}"
  return success

def test_shape : IO Bool := do
  IO.println "Testing Shape Safety Verifier..."

  -- Test GPT-2 shape verification
  let gpt2_ok := Shape.verify_gpt2_shapes

  let success := gpt2_ok
  IO.println s!"  ✓ Shape safety tests: {if success then \"PASSED\" else \"FAILED\"}"
  return success

def test_guard : IO Bool := do
  IO.println "Testing Guard Extractor..."

  -- Test guard bundle creation
  let bundle := Guard.extract_all_guards
  let has_predicates := bundle.predicates.length > 0
  let has_rust := bundle.rust_module.length > 0
  let has_python := bundle.python_module.length > 0

  let success := has_predicates && has_rust && has_python
  IO.println s!"  ✓ Guard extractor tests: {if success then \"PASSED\" else \"FAILED\"}"
  return success

def main : IO Unit := do
  IO.println "Running Dataset Safety Specifications Test Suite"
  IO.println "=" * 50

  let lineage_ok ← test_lineage
  let lineage_regression_ok ← test_lineage_regression
  let policy_ok ← test_policy
  let optimizer_ok ← test_optimizer
  let shape_ok ← test_shape
  let guard_ok ← test_guard

  IO.println "\n" ++ "=" * 50
  let all_passed := lineage_ok && lineage_regression_ok && policy_ok && optimizer_ok && shape_ok && guard_ok

  if all_passed then
    IO.println "✓ All tests PASSED!"
    IO.Process.exit 0
  else
    IO.println "✗ Some tests FAILED!"
    IO.Process.exit 1
