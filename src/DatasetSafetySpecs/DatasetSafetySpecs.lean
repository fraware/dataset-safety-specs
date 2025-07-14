/-
# Dataset Safety Specifications

Main module for the dataset safety specifications framework.
This module provides formal verification for dataset lineage, policy compliance,
and training-time safety guarantees.
-/

import DatasetSafetySpecs.Lineage
import DatasetSafetySpecs.Policy
import DatasetSafetySpecs.Optimizer
import DatasetSafetySpecs.Shape
import DatasetSafetySpecs.Guard

/-- Main namespace for dataset safety specifications -/
namespace DatasetSafetySpecs

/-- Version information -/
def version := "0.1.0"

/-- Core safety guarantees provided by this framework -/
inductive SafetyGuarantee where
  | LineageProof : Lineage.Transform → SafetyGuarantee
  | PolicyCompliance : Policy.Filter → SafetyGuarantee
  | OptimizerStability : Optimizer.Invariant → SafetyGuarantee
  | ShapeSafety : Shape.Verification → SafetyGuarantee

/-- Bundle containing all safety guarantees for a dataset -/
structure SafetyBundle where
  dataset_hash : String
  guarantees : List SafetyGuarantee
  metadata : String

/-- Verify all safety guarantees in a bundle -/
def verify_bundle (bundle : SafetyBundle) : Bool :=
  bundle.guarantees.all (fun g =>
    match g with
    | .LineageProof t => Lineage.verify_transform t
    | .PolicyCompliance f => Policy.verify_filter f
    | .OptimizerStability i => Optimizer.verify_invariant i
    | .ShapeSafety v => Shape.verify v
  )

end DatasetSafetySpecs
