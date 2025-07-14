"""
Data Integration Module

Handles Parquet/CSV file operations with proofhash metadata for lineage tracking.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from .__init__ import Row, lean_guarded


class ProofHashMetadata:
    """Metadata container for proof hashes and lineage information."""

    def __init__(self, proof_hash: str, guard_policy: str, transform_name: str):
        self.proof_hash = proof_hash
        self.guard_policy = guard_policy
        self.transform_name = transform_name
        self.timestamp = pd.Timestamp.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_hash": self.proof_hash,
            "guard_policy": self.guard_policy,
            "transform_name": self.transform_name,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofHashMetadata":
        metadata = cls(data["proof_hash"], data["guard_policy"], data["transform_name"])
        metadata.timestamp = data.get("timestamp", "")
        return metadata


def save_with_proofhash(
    df: pd.DataFrame, filepath: Union[str, Path], metadata: ProofHashMetadata, **kwargs
) -> None:
    """
    Save DataFrame with proofhash metadata.

    Args:
        df: DataFrame to save
        filepath: Output file path
        metadata: ProofHash metadata
        **kwargs: Additional arguments for pandas save methods
    """
    filepath = Path(filepath)

    # Save the main data file
    if filepath.suffix.lower() == ".parquet":
        df.to_parquet(filepath, **kwargs)
    elif filepath.suffix.lower() == ".csv":
        df.to_csv(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Save proofhash metadata
    metadata_path = filepath.with_suffix(filepath.suffix + ".proofhash")
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def load_with_proofhash(
    filepath: Union[str, Path],
) -> tuple[pd.DataFrame, Optional[ProofHashMetadata]]:
    """
    Load DataFrame with proofhash metadata.

    Args:
        filepath: Input file path

    Returns:
        Tuple of (DataFrame, ProofHashMetadata or None)
    """
    filepath = Path(filepath)

    # Load the main data file
    if filepath.suffix.lower() == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix.lower() == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Load proofhash metadata if it exists
    metadata_path = filepath.with_suffix(filepath.suffix + ".proofhash")
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata_data = json.load(f)
            metadata = ProofHashMetadata.from_dict(metadata_data)

    return df, metadata


def dataframe_to_rows(df: pd.DataFrame) -> List[Row]:
    """Convert DataFrame to list of Row objects for guard checking."""
    rows = []
    for _, row in df.iterrows():
        # Extract PHI fields (assuming columns with 'phi' in name)
        phi_fields = []
        for col in df.columns:
            if "phi" in col.lower() and pd.notna(row[col]):
                phi_fields.append(str(row[col]))

        # Extract age field (assuming 'age' column)
        age = None
        if "age" in df.columns and pd.notna(row["age"]):
            age = int(row["age"])

        # Extract GDPR special categories (assuming 'gdpr' columns)
        gdpr_fields = []
        for col in df.columns:
            if "gdpr" in col.lower() and pd.notna(row[col]):
                gdpr_fields.append(str(row[col]))

        # Extract custom fields
        custom_fields = []
        for col in df.columns:
            if col not in ["age"] and not any(
                x in col.lower() for x in ["phi", "gdpr"]
            ):
                if pd.notna(row[col]):
                    custom_fields.append((col, str(row[col])))

        rows.append(
            Row(
                phi=phi_fields,
                age=age,
                gdpr_special=gdpr_fields,
                custom_fields=custom_fields,
            )
        )

    return rows


def rows_to_dataframe(rows: List[Row]) -> pd.DataFrame:
    """Convert list of Row objects back to DataFrame."""
    data = []
    for row in rows:
        row_dict = {}

        # Add PHI fields
        for i, phi_field in enumerate(row.phi):
            row_dict[f"phi_{i}"] = phi_field

        # Add age
        if row.age is not None:
            row_dict["age"] = row.age

        # Add GDPR fields
        for i, gdpr_field in enumerate(row.gdpr_special):
            row_dict[f"gdpr_{i}"] = gdpr_field

        # Add custom fields
        for key, value in row.custom_fields:
            row_dict[key] = value

        data.append(row_dict)

    return pd.DataFrame(data)


@lean_guarded(policy="compliance")
def safe_etl_transform_dataframe(
    df: pd.DataFrame, transform_name: str, **kwargs
) -> pd.DataFrame:
    """
    Safe ETL transform for DataFrames with automatic guard checking.

    Args:
        df: Input DataFrame
        transform_name: Name of the transform for lineage tracking
        **kwargs: Additional transform parameters

    Returns:
        Transformed DataFrame
    """
    # Convert DataFrame to Row objects for guard checking
    rows = dataframe_to_rows(df)

    # Apply the transform (this will be checked by the decorator)
    # For now, just return the original DataFrame
    # In practice, this would apply actual transformations
    return df


def create_proofhash_metadata(
    df: pd.DataFrame, transform_name: str, guard_policy: str = "compliance"
) -> ProofHashMetadata:
    """Create proofhash metadata for a DataFrame and transform."""
    # Create a hash of the DataFrame content and transform parameters
    content_hash = hashlib.sha256(
        df.to_json(orient="records", sort_keys=True).encode()
    ).hexdigest()

    transform_hash = hashlib.sha256(
        json.dumps(
            {"transform_name": transform_name, "content_hash": content_hash},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    return ProofHashMetadata(transform_hash, guard_policy, transform_name)
