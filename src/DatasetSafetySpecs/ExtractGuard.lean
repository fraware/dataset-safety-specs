/-
# Guard Extractor Executable

Command-line tool for extracting Rust and Python guards from Lean predicates.
-/

import DatasetSafetySpecs.Guard
import System.IO

def main : IO Unit := do

def main : IO Unit := do
  IO.println "Extracting guards from Lean predicates..."

  let bundle := Guard.extract_all_guards

  -- Create output directories
  IO.FS.createDirAll "extracted/rust/src"
  IO.FS.createDirAll "extracted/python/ds_guard"

  -- Write Rust files
  IO.FS.writeFile "extracted/rust/Cargo.toml" bundle.cargo_toml
  IO.FS.writeFile "extracted/rust/src/lib.rs" bundle.rust_module

  -- Write Python files
  IO.FS.writeFile "extracted/python/setup.py" bundle.setup_py
  IO.FS.writeFile "extracted/python/ds_guard/__init__.py" bundle.python_module

  IO.println "✓ Guards extracted successfully!"
  IO.println "  - Rust guards: extracted/rust/"
  IO.println "  - Python guards: extracted/python/"
