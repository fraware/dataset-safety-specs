# Contributing to Dataset Safety Specifications

Thank you for your interest in contributing to the Dataset Safety Specifications (DSS) framework! This document provides guidelines for contributing to the project.

## Overview

The Dataset Safety Specs framework provides formal verification for:

- Dataset lineage with hash-chain consistency
- Data policy filters (HIPAA PHI, COPPA, GDPR)
- Optimizer invariants (SGD, AdamW, Lion)
- Shape safety verification (ONNX/PyTorch)
- Auto-generated guards (Rust/Python)

## Development Setup

### Prerequisites

1. **Lean 4**: Install Lean 4.7.0 or later

   ```bash
   # Using elan
   elan install leanprover/lean4:v4.7.0
   elan default leanprover/lean4:v4.7.0
   ```

2. **Python 3.8+**: For Python components and ONNX parsing

   ```bash
   pip install -r requirements.txt
   ```

3. **Rust**: For Rust guard generation
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

### Building the Project

```bash
# Clone the repository
git clone https://github.com/your-org/dataset-safety-specs.git
cd dataset-safety-specs

# Install dependencies
pip install -r requirements.txt

# Build Lean project
lake build

# Run tests
lake exe test_suite
```

## Project Structure

```
dataset-safety-specs/
├── src/DatasetSafetySpecs/          # Lean specifications
│   ├── Lineage.lean                 # Dataset lineage module
│   ├── Policy.lean                  # Data policy filters
│   ├── Optimizer.lean               # Optimizer invariants
│   ├── Shape.lean                   # Shape safety verifier
│   └── Guard.lean                   # Guard extractor
├── python/                          # Python components
│   ├── ds_guard/                    # Python guard library
│   └── onnx2lean_shape.py          # ONNX to Lean converter
├── extracted/                       # Generated guards
│   ├── rust/                        # Rust guards
│   └── python/                      # Python guards
├── docs/                           # Documentation
└── .github/                        # GitHub workflows and templates
```

## Development Workflow

### 1. Making Changes

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes**:

   ```bash
   # Run Lean tests
   lake exe test_suite

   # Run Python tests
   python -m pytest tests/

   # Run benchmarks
   lake exe benchmark_suite
   ```

4. **Format your code**:

   ```bash
   # Format Lean code
   lake exe lean --run src/format.lean

   # Format Python code
   black python/
   isort python/
   ```

### 2. Submitting Changes

1. **Commit your changes** with clear commit messages:

   ```bash
   git commit -m "feat: add new PHI detection pattern"
   ```

2. **Push to your branch**:

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request** with:
   - Clear description of changes
   - Link to related issues
   - Test results
   - Documentation updates

## Coding Standards

### Lean Code

- **File organization**: One module per file
- **Naming**: Use snake_case for functions, PascalCase for types
- **Documentation**: Use docstrings for all public functions
- **Proofs**: Use `sorry` for incomplete proofs, add TODO comments

```lean
/-- Example function with documentation -/
def example_function (input : String) : Bool :=
  -- Implementation here
  true

/-- Example theorem with proof -/
theorem example_theorem (x : Nat) : x + 0 = x := by
  -- Proof here
  sorry  -- TODO: Complete proof
```

### Python Code

- **Style**: Follow PEP 8, use Black for formatting
- **Type hints**: Use type hints for all functions
- **Documentation**: Use docstrings for all functions
- **Testing**: Write unit tests for new functionality

```python
def example_function(input_data: str) -> bool:
    """Example function with documentation.

    Args:
        input_data: Input string to process

    Returns:
        True if processing successful
    """
    # Implementation here
    return True
```

### Rust Code

- **Style**: Use rustfmt for formatting
- **Documentation**: Use doc comments for public items
- **Error handling**: Use Result types appropriately

```rust
/// Example function with documentation
pub fn example_function(input: &str) -> bool {
    // Implementation here
    true
}
```

## Testing Guidelines

### Lean Tests

- Write tests for all public functions
- Test both positive and negative cases
- Use descriptive test names

```lean
def test_example_function : IO Bool := do
  let result := example_function "test"
  return result = true
```

### Python Tests

- Use pytest for testing
- Test edge cases and error conditions
- Mock external dependencies

```python
def test_example_function():
    """Test example function."""
    assert example_function("test") == True
    assert example_function("") == False
```

## Documentation

### Adding Documentation

1. **Update README.md** for user-facing changes
2. **Update docs/index.md** for API changes
3. **Add docstrings** to all new functions
4. **Create examples** in the examples/ directory

### Documentation Standards

- Use clear, concise language
- Include code examples
- Link to related documentation
- Keep documentation up to date

## Issue Reporting

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for solutions
3. **Try the latest version** from main branch

### Issue Templates

Use the appropriate issue template:

- **[POLICY]**: For data policy and compliance issues
- **[SHAPE]**: For shape safety verification issues
- **[BUG]**: For general bugs
- **[FEATURE]**: For feature requests

### Issue Guidelines

- **Be specific**: Include steps to reproduce
- **Include context**: Environment, versions, etc.
- **Provide examples**: Code, data, error messages
- **Use labels**: Help categorize the issue

## Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Update version** in all relevant files
2. **Update CHANGELOG.md** with changes
3. **Run full test suite** and benchmarks
4. **Create release bundle** using `./bundle.sh bundle`
5. **Tag the release** with version number
6. **Publish packages** to PyPI and crates.io (when ready)

## Getting Help

### Resources

- **Documentation**: Check the docs/ directory
- **Issues**: Search existing issues for solutions
- **Discussions**: Use GitHub Discussions for questions

### Contact

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Report security issues privately

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## License

By contributing to Dataset Safety Specifications, you agree that your contributions will be licensed under the MIT License.
