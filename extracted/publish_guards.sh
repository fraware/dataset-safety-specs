#!/bin/bash
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
