#!/usr/bin/env python3
"""
Real ONNX Parser

Real implementation using protobuf to parse ONNX files and extract computation graphs.
This replaces the stub implementations in Shape.lean and ShapeSafeVerify.lean.

Usage:
    python real_onnx_parser.py model.onnx
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

try:
    import onnx
    from onnx import numpy_helper
    import numpy as np

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Error: onnx library not available. Install with: pip install onnx numpy")


class RealONNXParser:
    """Real ONNX model parser using protobuf."""

    def __init__(self):
        self.logger = logging.getLogger("RealONNXParser")
        self.parsing_stats = {
            "parse_time": 0.0,
            "nodes_extracted": 0,
            "inputs_extracted": 0,
            "outputs_extracted": 0,
            "shape_inference_time": 0.0,
            "errors": [],
            "warnings": [],
        }

        # Shape inference cache
        self.shape_cache = {}

    def parse_onnx_model(self, model_path: str) -> Dict[str, Any]:
        """Parse ONNX model and extract computation graph."""
        if not ONNX_AVAILABLE:
            return self._handle_missing_onnx()

        start_time = time.time()

        try:
            self.logger.info(f"Parsing ONNX model: {model_path}")

            # Load ONNX model using protobuf
            model = onnx.load(model_path)
            graph = model.graph

            # Extract nodes
            nodes = self._extract_nodes(graph.node)

            # Extract inputs with shapes
            inputs = self._extract_inputs(graph.input)

            # Extract outputs with shapes
            outputs = self._extract_outputs(graph.output)

            # Extract initializers (weights)
            initializers = self._extract_initializers(graph.initializer)

            # Perform shape inference
            shape_inference_start = time.time()
            inferred_shapes = self._infer_shapes(nodes, inputs, initializers)
            self.parsing_stats["shape_inference_time"] = (
                time.time() - shape_inference_start
            )

            # Build computation graph
            computation_graph = {
                "nodes": nodes,
                "inputs": inputs,
                "outputs": outputs,
                "initializers": initializers,
                "inferred_shapes": inferred_shapes,
                "model_info": {
                    "ir_version": model.ir_version,
                    "producer_name": model.producer_name,
                    "producer_version": model.producer_version,
                    "domain": model.domain,
                    "model_version": model.model_version,
                    "doc_string": model.doc_string,
                },
            }

            # Update stats
            self.parsing_stats["parse_time"] = time.time() - start_time
            self.parsing_stats["nodes_extracted"] = len(nodes)
            self.parsing_stats["inputs_extracted"] = len(inputs)
            self.parsing_stats["outputs_extracted"] = len(outputs)

            self.logger.info(
                f"✓ Successfully parsed {len(nodes)} nodes, {len(inputs)} inputs, {len(outputs)} outputs"
            )
            self.logger.info(f"  Parse time: {self.parsing_stats['parse_time']:.3f}s")
            self.logger.info(
                f"  Shape inference time: {self.parsing_stats['shape_inference_time']:.3f}s"
            )

            return {
                "success": True,
                "graph": computation_graph,
                "stats": self.parsing_stats.copy(),
            }

        except Exception as e:
            error_msg = f"Failed to parse ONNX file: {str(e)}"
            self.logger.error(error_msg)
            self.parsing_stats["errors"].append(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "stats": self.parsing_stats.copy(),
            }

    def _extract_nodes(self, graph_nodes) -> List[Dict[str, Any]]:
        """Extract nodes from ONNX graph."""
        nodes = []

        for i, node in enumerate(graph_nodes):
            node_info = {
                "index": i,
                "name": node.name or f"node_{i}",
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": self._extract_attributes(node.attribute),
                "domain": node.domain or "",
            }
            nodes.append(node_info)

        return nodes

    def _extract_inputs(self, graph_inputs) -> List[Dict[str, Any]]:
        """Extract input tensors with shapes."""
        inputs = []

        for input_info in graph_inputs:
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    shape.append(dim.dim_value)

            input_data = {
                "name": input_info.name,
                "shape": shape,
                "type": str(input_info.type.tensor_type.elem_type),
            }
            inputs.append(input_data)

        return inputs

    def _extract_outputs(self, graph_outputs) -> List[Dict[str, Any]]:
        """Extract output tensors with shapes."""
        outputs = []

        for output_info in graph_outputs:
            shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    shape.append(dim.dim_value)

            output_data = {
                "name": output_info.name,
                "shape": shape,
                "type": str(output_info.type.tensor_type.elem_type),
            }
            outputs.append(output_data)

        return outputs

    def _extract_initializers(self, graph_initializers) -> List[Dict[str, Any]]:
        """Extract initializer tensors (weights)."""
        initializers = []

        for init in graph_initializers:
            init_data = {
                "name": init.name,
                "dims": list(init.dims),
                "data_type": str(init.data_type),
                "raw_data_size": len(init.raw_data) if init.raw_data else 0,
            }
            initializers.append(init_data)

        return initializers

    def _extract_attributes(self, node_attributes) -> Dict[str, Any]:
        """Extract node attributes safely."""
        attr_dict = {}

        for attr in node_attributes:
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

    def _infer_shapes(
        self,
        nodes: List[Dict[str, Any]],
        inputs: List[Dict[str, Any]],
        initializers: List[Dict[str, Any]],
    ) -> Dict[str, List[int]]:
        """Infer shapes for all nodes in the graph."""
        # Create shape mapping
        shape_map = {}

        # Add input shapes
        for input_info in inputs:
            shape_map[input_info["name"]] = input_info["shape"]

        # Add initializer shapes
        for init in initializers:
            shape_map[init["name"]] = init["dims"]

        # Infer shapes for each node
        for node in nodes:
            try:
                output_shapes = self._infer_node_shapes(node, shape_map)
                for i, output_name in enumerate(node["outputs"]):
                    if i < len(output_shapes):
                        shape_map[output_name] = output_shapes[i]
            except Exception as e:
                self.logger.warning(
                    f"Failed to infer shapes for node {node['name']}: {e}"
                )
                # Add placeholder shape
                for output_name in node["outputs"]:
                    shape_map[output_name] = [-1]  # Unknown shape

        return shape_map

    def _infer_node_shapes(
        self, node: Dict[str, Any], shape_map: Dict[str, List[int]]
    ) -> List[List[int]]:
        """Infer output shapes for a single node."""
        op_type = node["op_type"]
        inputs = node["inputs"]
        outputs = node["outputs"]
        attributes = node["attributes"]

        # Get input shapes
        input_shapes = []
        for input_name in inputs:
            if input_name in shape_map:
                input_shapes.append(shape_map[input_name])
            else:
                input_shapes.append([-1])  # Unknown shape

        # Infer output shapes based on operation type
        if op_type == "Conv":
            return self._infer_conv_shapes(input_shapes, attributes)
        elif op_type == "MatMul":
            return self._infer_matmul_shapes(input_shapes)
        elif op_type == "Gemm":
            return self._infer_gemm_shapes(input_shapes, attributes)
        elif op_type == "Add":
            return self._infer_add_shapes(input_shapes)
        elif op_type == "Relu":
            return [input_shapes[0]] if input_shapes else [[]]
        elif op_type == "MaxPool":
            return self._infer_maxpool_shapes(input_shapes, attributes)
        elif op_type == "Flatten":
            return self._infer_flatten_shapes(input_shapes, attributes)
        elif op_type == "LayerNormalization":
            return [input_shapes[0]] if input_shapes else [[]]
        elif op_type == "Softmax":
            return [input_shapes[0]] if input_shapes else [[]]
        else:
            # Default: assume same shape as first input
            return [input_shapes[0]] if input_shapes else [[]]

    def _infer_conv_shapes(
        self, input_shapes: List[List[int]], attributes: Dict[str, Any]
    ) -> List[List[int]]:
        """Infer convolution output shapes."""
        if len(input_shapes) < 2:
            return [[]]

        input_shape = input_shapes[0]
        kernel_shape = input_shapes[1]

        if len(input_shape) != 4 or len(kernel_shape) != 4:
            return [[]]

        # Extract attributes
        strides = attributes.get("strides", {"value": [1, 1]})["value"]
        pads = attributes.get("pads", {"value": [0, 0, 0, 0]})["value"]
        dilations = attributes.get("dilations", {"value": [1, 1]})["value"]

        batch, in_channels, height, width = input_shape
        out_channels, _, kernel_h, kernel_w = kernel_shape

        # Calculate output dimensions
        out_height = (height + pads[0] + pads[2] - kernel_h) // strides[0] + 1
        out_width = (width + pads[1] + pads[3] - kernel_w) // strides[1] + 1

        return [[batch, out_channels, out_height, out_width]]

    def _infer_matmul_shapes(self, input_shapes: List[List[int]]) -> List[List[int]]:
        """Infer matrix multiplication output shapes."""
        if len(input_shapes) < 2:
            return [[]]

        a_shape = input_shapes[0]
        b_shape = input_shapes[1]

        if len(a_shape) != 2 or len(b_shape) != 2:
            return [[]]

        m, n1 = a_shape
        n2, p = b_shape

        if n1 != n2:
            return [[]]

        return [[m, p]]

    def _infer_gemm_shapes(
        self, input_shapes: List[List[int]], attributes: Dict[str, Any]
    ) -> List[List[int]]:
        """Infer GEMM output shapes."""
        if len(input_shapes) < 2:
            return [[]]

        a_shape = input_shapes[0]
        b_shape = input_shapes[1]

        if len(a_shape) != 2 or len(b_shape) != 2:
            return [[]]

        m, n1 = a_shape
        n2, p = b_shape

        if n1 != n2:
            return [[]]

        return [[m, p]]

    def _infer_add_shapes(self, input_shapes: List[List[int]]) -> List[List[int]]:
        """Infer addition output shapes."""
        if not input_shapes:
            return [[]]

        # For element-wise addition, return shape of first input
        return [input_shapes[0]]

    def _infer_maxpool_shapes(
        self, input_shapes: List[List[int]], attributes: Dict[str, Any]
    ) -> List[List[int]]:
        """Infer max pooling output shapes."""
        if not input_shapes:
            return [[]]

        input_shape = input_shapes[0]
        if len(input_shape) != 4:
            return [[]]

        kernel_shape = attributes.get("kernel_shape", {"value": [2, 2]})["value"]
        strides = attributes.get("strides", {"value": [2, 2]})["value"]

        batch, channels, height, width = input_shape
        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides

        out_height = height // stride_h
        out_width = width // stride_w

        return [[batch, channels, out_height, out_width]]

    def _infer_flatten_shapes(
        self, input_shapes: List[List[int]], attributes: Dict[str, Any]
    ) -> List[List[int]]:
        """Infer flatten output shapes."""
        if not input_shapes:
            return [[]]

        input_shape = input_shapes[0]
        axis = attributes.get("axis", {"value": 1})["value"]

        if axis < 0:
            axis = len(input_shape) + axis

        # Calculate flattened dimension
        flattened = 1
        for i in range(axis, len(input_shape)):
            flattened *= input_shape[i]

        output_shape = input_shape[:axis] + [flattened]
        return [output_shape]

    def _handle_missing_onnx(self) -> Dict[str, Any]:
        """Handle case when ONNX library is not available."""
        return {
            "success": False,
            "error": "ONNX library not available",
            "stats": self.parsing_stats.copy(),
        }

    def export_graph_json(self, graph: Dict[str, Any], output_path: str) -> bool:
        """Export computation graph to JSON."""
        try:
            with open(output_path, "w") as f:
                json.dump(graph, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export graph: {e}")
            return False


def parse_onnx_model(
    model_path: str, output_json: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to parse ONNX model."""
    parser = RealONNXParser()
    result = parser.parse_onnx_model(model_path)

    if result["success"] and output_json:
        parser.export_graph_json(result["graph"], output_json)

    return result


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python real_onnx_parser.py <model.onnx> [output.json]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else None

    result = parse_onnx_model(model_path, output_json)

    if result["success"]:
        print("✓ ONNX model parsed successfully!")
        stats = result["stats"]
        print(f"  Nodes: {stats['nodes_extracted']}")
        print(f"  Inputs: {stats['inputs_extracted']}")
        print(f"  Outputs: {stats['outputs_extracted']}")
        print(f"  Parse time: {stats['parse_time']:.3f}s")
        if output_json:
            print(f"  Exported to: {output_json}")
    else:
        print(f"✗ Failed to parse ONNX model: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
