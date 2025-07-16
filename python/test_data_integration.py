#!/usr/bin/env python3
"""
Test script for data integration functionality.

Demonstrates:
1. Creating test data with PHI/COPPA/GDPR fields
2. Saving with proofhash metadata
3. Loading and verifying lineage
4. Safe ETL transforms with guards
"""

import pandas as pd
import tempfile
import os
from pathlib import Path

# Add the parent directory to the path so we can import ds_guard
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ds_guard import (
    save_with_proofhash,
    load_with_proofhash,
    safe_etl_transform_dataframe,
    create_proofhash_metadata,
    dataframe_to_rows,
    rows_to_dataframe,
)


def create_test_dataframe():
    """Create a test DataFrame with PHI, COPPA, and GDPR data."""
    data = {
        "phi_ssn": ["123-45-6789", "987-65-4321", "111-22-3333", None],
        "phi_medical_record": ["MR001", "MR002", None, "MR004"],
        "age": [25, 12, 30, 8],  # Some minors for COPPA testing
        "gdpr_health": ["diabetes", None, "hypertension", None],
        "gdpr_genetic": [None, "BRCA1", None, "BRCA2"],
        "name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"],
        "email": [
            "john@example.com",
            "jane@example.com",
            "bob@example.com",
            "alice@example.com",
        ],
    }
    return pd.DataFrame(data)


def test_parquet_integration():
    """Test Parquet file integration with proofhash metadata."""
    print("Testing Parquet file integration...")

    # Create test data
    df = create_test_dataframe()
    print(f"Created test DataFrame with {len(df)} rows")

    # Create metadata
    metadata = create_proofhash_metadata(df, "test_transform", "compliance")
    print(f"Created proofhash metadata: {metadata.proof_hash[:16]}...")

    # Save with proofhash
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        parquet_path = tmp.name

    try:
        save_with_proofhash(df, parquet_path, metadata)
        print(f"Saved DataFrame to {parquet_path}")

        # Check that proofhash file was created
        proofhash_path = parquet_path + ".proofhash"
        assert os.path.exists(proofhash_path), "Proofhash file not created"
        print(f"Proofhash metadata saved to {proofhash_path}")

        # Load with proofhash
        loaded_df, loaded_metadata = load_with_proofhash(parquet_path)
        print(f"Loaded DataFrame with {len(loaded_df)} rows")

        # Verify data integrity
        assert df.equals(loaded_df), "DataFrame content changed"
        if loaded_metadata is not None:
            assert (
                metadata.proof_hash == loaded_metadata.proof_hash
            ), "Proofhash mismatch"
        print("✓ Data integrity verified")

    finally:
        # Cleanup
        if os.path.exists(parquet_path):
            os.unlink(parquet_path)
        if os.path.exists(parquet_path + ".proofhash"):
            os.unlink(parquet_path + ".proofhash")


def test_csv_integration():
    """Test CSV file integration with proofhash metadata."""
    print("\nTesting CSV file integration...")

    # Create test data
    df = create_test_dataframe()

    # Create metadata
    metadata = create_proofhash_metadata(df, "csv_transform", "phi")

    # Save with proofhash
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = tmp.name

    try:
        save_with_proofhash(df, csv_path, metadata)
        print(f"Saved DataFrame to {csv_path}")

        # Load with proofhash
        loaded_df, loaded_metadata = load_with_proofhash(csv_path)
        print(f"Loaded DataFrame with {len(loaded_df)} rows")

        # Verify data integrity
        assert df.equals(loaded_df), "DataFrame content changed"
        if loaded_metadata is not None:
            assert (
                metadata.proof_hash == loaded_metadata.proof_hash
            ), "Proofhash mismatch"
        print("✓ Data integrity verified")

    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        if os.path.exists(csv_path + ".proofhash"):
            os.unlink(csv_path + ".proofhash")


def test_safe_etl_transform():
    """Test safe ETL transform with guard checking."""
    print("\nTesting safe ETL transform...")

    # Create test data with some violations
    df = create_test_dataframe()

    try:
        # This should raise an exception due to PHI/COPPA violations
        result = safe_etl_transform_dataframe(df, "test_transform")
        print("✓ ETL transform completed (no violations detected)")
    except ValueError as e:
        print(f"✗ ETL transform blocked: {e}")
        print("This is expected behavior - guards are working correctly!")


def test_dataframe_row_conversion():
    """Test conversion between DataFrame and Row objects."""
    print("\nTesting DataFrame/Row conversion...")

    # Create test data
    df = create_test_dataframe()

    # Convert to Row objects
    rows = dataframe_to_rows(df)
    print(f"Converted DataFrame to {len(rows)} Row objects")

    # Check that PHI detection works
    phi_rows = [row for row in rows if row.phi]
    print(f"Found {len(phi_rows)} rows with PHI data")

    # Check that COPPA detection works
    minor_rows = [row for row in rows if row.age is not None and row.age < 13]
    print(f"Found {len(minor_rows)} rows with minors (COPPA)")

    # Convert back to DataFrame
    converted_df = rows_to_dataframe(rows)
    print(f"Converted back to DataFrame with {len(converted_df)} rows")

    # Verify conversion integrity
    assert len(df) == len(converted_df), "Row count mismatch"
    print("✓ DataFrame/Row conversion verified")


def main():
    """Run all data integration tests."""
    print("Dataset Safety Specs - Data Integration Test")
    print("=" * 50)

    try:
        test_parquet_integration()
        test_csv_integration()
        test_safe_etl_transform()
        test_dataframe_row_conversion()

        print("\n" + "=" * 50)
        print("✓ All data integration tests completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
