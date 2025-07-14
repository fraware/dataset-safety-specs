#!/bin/bash
# Bundle script for Dataset Safety Specs
# Packages Spec.lean, guard.rs, guard.py, and lean-hash.txt into distributable bundles

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION="0.1.0"
BUNDLE_DIR="dist"
LEAN_SPEC_FILE="src/DatasetSafetySpecs/DatasetSafetySpecs.lean"
RUST_GUARD_FILE="extracted/rust/src/lib.rs"
PYTHON_GUARD_FILE="extracted/python/ds_guard/__init__.py"
HASH_FILE="lean-hash.txt"

echo -e "${BLUE}Dataset Safety Specs Bundle Script${NC}"
echo "=================================="

# Create bundle directory
mkdir -p "$BUNDLE_DIR"

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: File $1 not found${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Found $1${NC}"
}

# Function to generate lean hash
generate_lean_hash() {
    echo "Generating Lean specification hash..."
    if [ -f "$LEAN_SPEC_FILE" ]; then
        sha256sum "$LEAN_SPEC_FILE" | cut -d' ' -f1 > "$HASH_FILE"
        echo -e "${GREEN}✓ Generated $HASH_FILE${NC}"
    else
        echo -e "${YELLOW}Warning: $LEAN_SPEC_FILE not found, creating empty hash${NC}"
        echo "0000000000000000000000000000000000000000000000000000000000000000" > "$HASH_FILE"
    fi
}

# Function to extract guards
extract_guards() {
    echo "Extracting guards from Lean predicates..."
    if command -v lake >/dev/null 2>&1; then
        lake exe extract_guard
        echo -e "${GREEN}✓ Guards extracted${NC}"
    else
        echo -e "${YELLOW}Warning: lake not found, skipping guard extraction${NC}"
    fi
}

# Function to create Python wheel
create_python_wheel() {
    echo "Creating Python wheel..."
    if [ -d "extracted/python" ]; then
        cd extracted/python
        python setup.py sdist bdist_wheel
        cp dist/*.whl ../../"$BUNDLE_DIR"/
        cd ../..
        echo -e "${GREEN}✓ Python wheel created${NC}"
    else
        echo -e "${YELLOW}Warning: Python guards not found${NC}"
    fi
}

# Function to create Rust crate
create_rust_crate() {
    echo "Creating Rust crate..."
    if [ -d "extracted/rust" ]; then
        cd extracted/rust
        cargo build --release
        # Create a simple tarball of the crate
        tar -czf ../../"$BUNDLE_DIR"/ds-guard-rust.tar.gz src/ Cargo.toml
        cd ../..
        echo -e "${GREEN}✓ Rust crate created${NC}"
    else
        echo -e "${YELLOW}Warning: Rust guards not found${NC}"
    fi
}

# Function to create Lean bundle
create_lean_bundle() {
    echo "Creating Lean specification bundle..."
    
    # Check for required files
    check_file "$LEAN_SPEC_FILE" || return 1
    
    # Create Lean bundle
    mkdir -p "$BUNDLE_DIR/lean"
    cp "$LEAN_SPEC_FILE" "$BUNDLE_DIR/lean/"
    cp "$HASH_FILE" "$BUNDLE_DIR/lean/"
    
    # Add README for Lean bundle
    cat > "$BUNDLE_DIR/lean/README.md" << EOF
# Dataset Safety Specs - Lean Bundle

This bundle contains the formal Lean specifications for dataset safety verification.

## Files:
- \`DatasetSafetySpecs.lean\`: Main specification file
- \`lean-hash.txt\`: SHA256 hash of the specification

## Usage:
\`\`\`bash
# Import in Lean
import DatasetSafetySpecs
\`\`\`

Version: $VERSION
EOF
    
    # Create tarball
    tar -czf "$BUNDLE_DIR/dataset-safety-specs-lean-$VERSION.tar.gz" -C "$BUNDLE_DIR" lean/
    echo -e "${GREEN}✓ Lean bundle created${NC}"
}

# Function to create complete bundle
create_complete_bundle() {
    echo "Creating complete bundle..."
    
    # Create bundle manifest
    cat > "$BUNDLE_DIR/manifest.json" << EOF
{
  "version": "$VERSION",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "components": {
    "lean_spec": "$(basename "$LEAN_SPEC_FILE")",
    "rust_guard": "$(basename "$RUST_GUARD_FILE")",
    "python_guard": "$(basename "$PYTHON_GUARD_FILE")",
    "lean_hash": "$(basename "$HASH_FILE")"
  },
  "files": [
    "dataset-safety-specs-lean-$VERSION.tar.gz"
  ]
}
EOF
    
    # Create complete tarball
    tar -czf "$BUNDLE_DIR/dataset-safety-specs-$VERSION.tar.gz" -C "$BUNDLE_DIR" .
    echo -e "${GREEN}✓ Complete bundle created${NC}"
}

# Function to publish to PyPI (stub)
publish_to_pypi() {
    echo "Publishing to PyPI..."
    echo -e "${YELLOW}Note: PyPI publishing not yet implemented${NC}"
    echo "Would upload Python wheel to PyPI"
}

# Function to publish to crates.io (stub)
publish_to_crates() {
    echo "Publishing to crates.io..."
    echo -e "${YELLOW}Note: crates.io publishing not yet implemented${NC}"
    echo "Would upload Rust crate to crates.io"
}

# Main execution
main() {
    local action="${1:-bundle}"
    
    case "$action" in
        "bundle")
            echo "Creating complete bundle..."
            extract_guards
            generate_lean_hash
            create_lean_bundle
            create_python_wheel
            create_rust_crate
            create_complete_bundle
            echo -e "${GREEN}✓ Bundle complete!${NC}"
            ;;
        "lean")
            echo "Creating Lean bundle only..."
            generate_lean_hash
            create_lean_bundle
            ;;
        "python")
            echo "Creating Python wheel..."
            extract_guards
            create_python_wheel
            ;;
        "rust")
            echo "Creating Rust crate..."
            extract_guards
            create_rust_crate
            ;;
        "publish")
            echo "Publishing packages..."
            publish_to_pypi
            publish_to_crates
            ;;
        "clean")
            echo "Cleaning bundle directory..."
            rm -rf "$BUNDLE_DIR"
            echo -e "${GREEN}✓ Cleaned${NC}"
            ;;
        *)
            echo "Usage: $0 [bundle|lean|python|rust|publish|clean]"
            echo ""
            echo "Actions:"
            echo "  bundle  - Create complete bundle (default)"
            echo "  lean    - Create Lean specification bundle only"
            echo "  python  - Create Python wheel only"
            echo "  rust    - Create Rust crate only"
            echo "  publish - Publish to PyPI and crates.io (stub)"
            echo "  clean   - Clean bundle directory"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 