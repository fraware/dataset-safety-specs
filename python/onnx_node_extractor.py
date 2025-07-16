#!/usr/bin/env python3
"""
Public ONNX Node Name Extractor

Robust extraction of ONNX node names with fallback strategies and error recovery.
Handles complex naming conventions and non-standard node names.

## DeepSeek Touch-points:
- Public ONNX node name extraction
- Fallback naming strategies
- Node name validation
- Parsing error recovery
"""

import re
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json

try:
    import onnx
    from onnx import numpy_helper
    import numpy as np

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx library not available. Install with: pip install onnx")


class ONNXNodeNameExtractor:
    """Robust ONNX node name extractor with fallback strategies."""

    def __init__(self):
        self.logger = logging.getLogger("ONNXNodeNameExtractor")
        self.extraction_stats = {
            "total_nodes": 0,
            "successful_extractions": 0,
            "fallback_used": 0,
            "errors": 0,
        }

        # Common naming patterns
        self.naming_patterns = {
            "tensorflow": r"^[A-Za-z_][A-Za-z0-9_]*$",
            "pytorch": r"^[A-Za-z_][A-Za-z0-9_]*$",
            "generic": r"^[A-Za-z_][A-Za-z0-9_]*$",
            "complex": r"^[A-Za-z_][A-Za-z0-9_\-\.]*$",
        }

        # Fallback naming strategies
        self.fallback_strategies = [
            "op_type_index",
            "op_type_hash",
            "input_based",
            "output_based",
            "position_based",
        ]

    def extract_node_names(self, model_path: str) -> Dict[str, Any]:
        """Extract node names from ONNX model with robust error handling."""
        if not ONNX_AVAILABLE:
            return self._handle_missing_onnx()

        try:
            self.logger.info(f"Extracting node names from: {model_path}")

            # Load ONNX model
            model = onnx.load(model_path)
            graph = model.graph

            # Extract nodes
            nodes = []
            for i, node in enumerate(graph.node):
                node_info = self._extract_single_node(node, i)
                nodes.append(node_info)

            # Update stats
            self.extraction_stats["total_nodes"] = len(nodes)
            self.extraction_stats["successful_extractions"] = sum(
                1 for node in nodes if node["extraction_success"]
            )

            return {
                "success": True,
                "nodes": nodes,
                "stats": self.extraction_stats.copy(),
                "model_info": {
                    "ir_version": model.ir_version,
                    "producer_name": model.producer_name,
                    "producer_version": model.producer_version,
                    "domain": model.domain,
                    "model_version": model.model_version,
                    "doc_string": model.doc_string,
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to extract node names: {e}")
            self.extraction_stats["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "stats": self.extraction_stats.copy(),
            }

    def _extract_single_node(self, node: Any, index: int) -> Dict[str, Any]:
        """Extract name from a single ONNX node with fallback strategies."""
        original_name = node.name
        op_type = node.op_type

        # Try to extract valid name
        extracted_name = self._extract_valid_name(original_name, op_type, index)

        # Determine extraction method
        extraction_method = "original"
        if extracted_name != original_name:
            extraction_method = "fallback"
            self.extraction_stats["fallback_used"] += 1

        return {
            "index": index,
            "original_name": original_name,
            "extracted_name": extracted_name,
            "op_type": op_type,
            "extraction_method": extraction_method,
            "extraction_success": extracted_name is not None,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": self._extract_attributes(node.attribute),
        }

    def _extract_valid_name(self, original_name: str, op_type: str, index: int) -> str:
        """Extract a valid node name using multiple strategies."""

        # Strategy 1: Use original name if valid
        if self._is_valid_name(original_name):
            return original_name

        # Strategy 2: Try to clean the original name
        cleaned_name = self._clean_name(original_name)
        if self._is_valid_name(cleaned_name):
            return cleaned_name

        # Strategy 3: Use op_type with index
        op_type_name = f"{op_type}_{index}"
        if self._is_valid_name(op_type_name):
            return op_type_name

        # Strategy 4: Use op_type with hash
        name_hash = hashlib.md5(original_name.encode()).hexdigest()[:8]
        hash_name = f"{op_type}_{name_hash}"
        if self._is_valid_name(hash_name):
            return hash_name

        # Strategy 5: Use position-based naming
        position_name = f"node_{index}"
        return position_name

    def _is_valid_name(self, name: str) -> bool:
        """Check if a name is valid according to common patterns."""
        if not name or len(name) == 0:
            return False

        # Check against naming patterns
        for pattern_name, pattern in self.naming_patterns.items():
            if re.match(pattern, name):
                return True

        # Additional checks
        if name.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            return False

        if len(name) > 100:  # Too long
            return False

        return True

    def _clean_name(self, name: str) -> str:
        """Clean a potentially invalid name."""
        if not name:
            return ""

        # Remove or replace invalid characters
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)

        # Remove leading numbers
        cleaned = re.sub(r"^[0-9]+", "", cleaned)

        # Remove multiple underscores
        cleaned = re.sub(r"_+", "_", cleaned)

        # Remove leading/trailing underscores
        cleaned = cleaned.strip("_")

        # Ensure it starts with a letter or underscore
        if cleaned and not cleaned[0].isalpha() and cleaned[0] != "_":
            cleaned = f"node_{cleaned}"

        return cleaned

    def _extract_attributes(self, attributes: List[Any]) -> Dict[str, Any]:
        """Extract node attributes safely."""
        attr_dict = {}
        for attr in attributes:
            try:
                attr_name = attr.name
                attr_type = str(attr.type)

                # Extract value based on type
                if attr_type == "INT":
                    attr_value = attr.i
                elif attr_type == "FLOAT":
                    attr_value = attr.f
                elif attr_type == "STRING":
                    attr_value = attr.s.decode("utf-8")
                elif attr_type == "INTS":
                    attr_value = list(attr.ints)
                elif attr_type == "FLOATS":
                    attr_value = list(attr.floats)
                elif attr_type == "STRINGS":
                    attr_value = [s.decode("utf-8") for s in attr.strings]
                else:
                    attr_value = str(attr.type)

                attr_dict[attr_name] = {"type": attr_type, "value": attr_value}

            except Exception as e:
                self.logger.warning(f"Failed to extract attribute: {e}")
                continue

        return attr_dict

    def _handle_missing_onnx(self) -> Dict[str, Any]:
        """Handle case when ONNX library is not available."""
        return {
            "success": False,
            "error": "ONNX library not available",
            "stats": self.extraction_stats.copy(),
        }

    def validate_node_names(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate extracted node names."""
        validation_results = {
            "total_nodes": len(nodes),
            "valid_names": 0,
            "invalid_names": 0,
            "duplicate_names": 0,
            "issues": [],
        }

        seen_names = set()

        for node in nodes:
            name = node.get("extracted_name", "")

            # Check if name is valid
            if self._is_valid_name(name):
                validation_results["valid_names"] += 1
            else:
                validation_results["invalid_names"] += 1
                validation_results["issues"].append(
                    {
                        "node_index": node.get("index"),
                        "issue": "invalid_name",
                        "name": name,
                    }
                )

            # Check for duplicates
            if name in seen_names:
                validation_results["duplicate_names"] += 1
                validation_results["issues"].append(
                    {
                        "node_index": node.get("index"),
                        "issue": "duplicate_name",
                        "name": name,
                    }
                )
            else:
                seen_names.add(name)

        return validation_results

    def generate_naming_report(
        self, extraction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive naming report."""
        if not extraction_result.get("success"):
            return {
                "success": False,
                "error": extraction_result.get("error", "Unknown error"),
            }

        nodes = extraction_result.get("nodes", [])
        validation = self.validate_node_names(nodes)

        # Analyze naming patterns
        naming_analysis = {
            "original_names_used": sum(
                1 for node in nodes if node["extraction_method"] == "original"
            ),
            "fallback_names_used": sum(
                1 for node in nodes if node["extraction_method"] == "fallback"
            ),
            "op_types": {},
            "naming_patterns": {},
        }

        for node in nodes:
            op_type = node["op_type"]
            naming_analysis["op_types"][op_type] = (
                naming_analysis["op_types"].get(op_type, 0) + 1
            )

            method = node["extraction_method"]
            naming_analysis["naming_patterns"][method] = (
                naming_analysis["naming_patterns"].get(method, 0) + 1
            )

        return {
            "success": True,
            "extraction_stats": extraction_result.get("stats", {}),
            "validation_results": validation,
            "naming_analysis": naming_analysis,
            "model_info": extraction_result.get("model_info", {}),
        }

    def export_node_names(
        self, extraction_result: Dict[str, Any], output_path: str
    ) -> bool:
        """Export extracted node names to file."""
        try:
            if not extraction_result.get("success"):
                return False

            # Generate report
            report = self.generate_naming_report(extraction_result)

            # Export to JSON
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Failed to export node names: {e}")
            return False


def extract_onnx_node_names(
    model_path: str, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to extract ONNX node names."""
    extractor = ONNXNodeNameExtractor()

    # Extract node names
    result = extractor.extract_node_names(model_path)

    # Export if output path provided
    if output_path and result.get("success"):
        extractor.export_node_names(result, output_path)

    return result


# Example usage and testing
if __name__ == "__main__":
    # Example: Extract node names from ONNX model
    result = extract_onnx_node_names(
        model_path="test_model.onnx", output_path="node_names_report.json"
    )

    if result.get("success"):
        print("✓ ONNX node names extracted successfully!")
        print(f"  Total nodes: {result['stats']['total_nodes']}")
        print(f"  Successful extractions: {result['stats']['successful_extractions']}")
        print(f"  Fallback used: {result['stats']['fallback_used']}")
    else:
        print(f"✗ Failed to extract node names: {result.get('error')}")
