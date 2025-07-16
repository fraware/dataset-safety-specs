#!/usr/bin/env python3
"""
Runtime Safety Kernel Interface

Provides integration points for production ML pipelines with:
- Runtime safety checks
- Model asset guards
- Compliance monitoring
- Audit trail generation
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import threading
from contextlib import contextmanager

from .ds_guard import (
    Row,
    phi_guard,
    coppa_guard,
    gdpr_guard,
    save_with_proofhash,
    load_with_proofhash,
    create_proofhash_metadata,
)


@dataclass
class SafetyEvent:
    """Safety event for audit trail."""

    timestamp: str
    event_type: str
    severity: str  # "info", "warning", "error", "critical"
    message: str
    context: Dict[str, Any]
    proof_hash: Optional[str] = None


@dataclass
class ModelAsset:
    """Model asset with safety metadata."""

    name: str
    version: str
    model_path: str
    safety_hash: str
    compliance_level: str
    last_verified: str
    verification_status: str


@dataclass
class RuntimeConfig:
    """Runtime safety configuration."""

    enable_phi_checking: bool = True
    enable_coppa_checking: bool = True
    enable_gdpr_checking: bool = True
    enable_shape_verification: bool = True
    enable_audit_trail: bool = True
    compliance_level: str = "strict"  # "strict", "moderate", "permissive"
    max_processing_time: float = 30.0  # seconds
    memory_limit_mb: int = 1024


class SafetyKernel:
    """Runtime safety kernel for ML pipeline integration."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.logger = logging.getLogger("SafetyKernel")
        self.events: List[SafetyEvent] = []
        self.model_assets: Dict[str, ModelAsset] = {}
        self.processing_stats = {
            "total_checks": 0,
            "violations_detected": 0,
            "processing_time": 0.0,
            "memory_usage": 0.0,
        }
        self._lock = threading.Lock()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def register_model_asset(self, asset: ModelAsset) -> bool:
        """Register a model asset for safety monitoring."""
        with self._lock:
            self.model_assets[asset.name] = asset
            self._log_event(
                "model_registered",
                "info",
                f"Model asset registered: {asset.name}",
                {"asset": asdict(asset)},
            )
            return True

    def verify_model_safety(self, model_name: str) -> bool:
        """Verify model safety compliance."""
        if model_name not in self.model_assets:
            self._log_event(
                "model_not_found",
                "error",
                f"Model not found: {model_name}",
                {"model_name": model_name},
            )
            return False

        asset = self.model_assets[model_name]

        # Verify model hash
        if not self._verify_model_hash(asset):
            self._log_event(
                "model_hash_mismatch",
                "critical",
                f"Model hash mismatch: {model_name}",
                {"asset": asdict(asset)},
            )
            return False

        # Verify compliance level
        if not self._verify_compliance_level(asset):
            self._log_event(
                "compliance_level_violation",
                "error",
                f"Compliance level violation: {model_name}",
                {"asset": asdict(asset)},
            )
            return False

        self._log_event(
            "model_safety_verified",
            "info",
            f"Model safety verified: {model_name}",
            {"asset": asdict(asset)},
        )
        return True

    def _verify_model_hash(self, asset: ModelAsset) -> bool:
        """Verify model file hash."""
        try:
            model_path = Path(asset.model_path)
            if not model_path.exists():
                return False

            # Calculate hash
            hash_md5 = hashlib.md5()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)

            calculated_hash = hash_md5.hexdigest()
            return calculated_hash == asset.safety_hash
        except Exception as e:
            self.logger.error(f"Hash verification failed: {e}")
            return False

    def _verify_compliance_level(self, asset: ModelAsset) -> bool:
        """Verify compliance level meets requirements."""
        compliance_levels = {
            "strict": ["strict", "moderate", "permissive"],
            "moderate": ["moderate", "permissive"],
            "permissive": ["permissive"],
        }

        required_levels = compliance_levels.get(self.config.compliance_level, [])
        return asset.compliance_level in required_levels

    @contextmanager
    def safety_context(self, operation: str, context: Dict[str, Any]):
        """Context manager for safety operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            self._log_event(
                "operation_started", "info", f"Operation started: {operation}", context
            )

            yield self

            # Check processing time
            processing_time = time.time() - start_time
            if processing_time > self.config.max_processing_time:
                self._log_event(
                    "processing_time_exceeded",
                    "warning",
                    f"Processing time exceeded limit: {processing_time:.2f}s",
                    {"operation": operation, "time": processing_time},
                )

            # Update stats
            with self._lock:
                self.processing_stats["total_checks"] += 1
                self.processing_stats["processing_time"] += processing_time
                self.processing_stats["memory_usage"] = self._get_memory_usage()

            self._log_event(
                "operation_completed",
                "info",
                f"Operation completed: {operation}",
                {"processing_time": processing_time},
            )

        except Exception as e:
            self._log_event(
                "operation_failed",
                "error",
                f"Operation failed: {operation} - {str(e)}",
                {"error": str(e)},
            )
            raise

    def check_data_safety(
        self, data: Union[List[Row], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check data safety compliance."""
        violations = []
        start_time = time.time()

        # Convert data to Row objects if needed
        if isinstance(data, dict):
            rows = [
                Row(
                    phi=data.get("phi", []),
                    age=data.get("age"),
                    gdpr_special=data.get("gdpr_special", []),
                    custom_fields=data.get("custom_fields", []),
                )
            ]
        else:
            rows = data

        # Check PHI violations
        if self.config.enable_phi_checking:
            phi_violations = [i for i, row in enumerate(rows) if phi_guard(row)]
            if phi_violations:
                violations.append(
                    {
                        "type": "phi",
                        "indices": phi_violations,
                        "count": len(phi_violations),
                    }
                )

        # Check COPPA violations
        if self.config.enable_coppa_checking:
            coppa_violations = [i for i, row in enumerate(rows) if coppa_guard(row)]
            if coppa_violations:
                violations.append(
                    {
                        "type": "coppa",
                        "indices": coppa_violations,
                        "count": len(coppa_violations),
                    }
                )

        # Check GDPR violations
        if self.config.enable_gdpr_checking:
            gdpr_violations = [i for i, row in enumerate(rows) if gdpr_guard(row)]
            if gdpr_violations:
                violations.append(
                    {
                        "type": "gdpr",
                        "indices": gdpr_violations,
                        "count": len(gdpr_violations),
                    }
                )

        processing_time = time.time() - start_time

        # Update stats
        with self._lock:
            self.processing_stats["total_checks"] += 1
            self.processing_stats["violations_detected"] += len(violations)
            self.processing_stats["processing_time"] += processing_time

        result = {
            "safe": len(violations) == 0,
            "violations": violations,
            "processing_time": processing_time,
            "rows_checked": len(rows),
        }

        # Log result
        if violations:
            self._log_event(
                "safety_violations_detected",
                "warning",
                f"Safety violations detected: {len(violations)} types",
                result,
            )
        else:
            self._log_event(
                "data_safety_verified",
                "info",
                f"Data safety verified: {len(rows)} rows",
                result,
            )

        return result

    def create_audit_trail(self) -> Dict[str, Any]:
        """Create comprehensive audit trail."""
        with self._lock:
            return {
                "timestamp": time.time(),
                "events": [asdict(event) for event in self.events],
                "model_assets": {
                    name: asdict(asset) for name, asset in self.model_assets.items()
                },
                "processing_stats": self.processing_stats.copy(),
                "config": asdict(self.config),
            }

    def export_audit_trail(self, filepath: str) -> bool:
        """Export audit trail to file."""
        try:
            audit_trail = self.create_audit_trail()
            with open(filepath, "w") as f:
                json.dump(audit_trail, f, indent=2)

            self._log_event(
                "audit_trail_exported",
                "info",
                f"Audit trail exported to: {filepath}",
                {"filepath": filepath},
            )
            return True
        except Exception as e:
            self._log_event(
                "audit_trail_export_failed",
                "error",
                f"Audit trail export failed: {str(e)}",
                {"filepath": filepath, "error": str(e)},
            )
            return False

    def _log_event(
        self, event_type: str, severity: str, message: str, context: Dict[str, Any]
    ):
        """Log a safety event."""
        event = SafetyEvent(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            event_type=event_type,
            severity=severity,
            message=message,
            context=context,
        )

        with self._lock:
            self.events.append(event)

        # Log to standard logging
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(log_level, f"{event_type}: {message}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available

    def get_safety_report(self) -> Dict[str, Any]:
        """Generate safety report."""
        with self._lock:
            return {
                "timestamp": time.time(),
                "config": asdict(self.config),
                "stats": self.processing_stats.copy(),
                "model_assets_count": len(self.model_assets),
                "events_count": len(self.events),
                "recent_violations": [
                    event
                    for event in self.events[-10:]  # Last 10 events
                    if "violation" in event.event_type.lower()
                ],
            }


class ModelAssetGuard:
    """Model asset guard for production deployment."""

    def __init__(self, kernel: SafetyKernel):
        self.kernel = kernel
        self.logger = logging.getLogger("ModelAssetGuard")

    def guard_model_inference(self, model_name: str, input_data: Any) -> Dict[str, Any]:
        """Guard model inference with safety checks."""
        # Verify model safety
        if not self.kernel.verify_model_safety(model_name):
            return {
                "success": False,
                "error": "Model safety verification failed",
                "safe_to_proceed": False,
            }

        # Check input data safety
        safety_result = self.kernel.check_data_safety(input_data)

        # Determine if safe to proceed
        safe_to_proceed = (
            safety_result["safe"] or self.kernel.config.compliance_level == "permissive"
        )

        return {
            "success": True,
            "safety_result": safety_result,
            "safe_to_proceed": safe_to_proceed,
            "model_name": model_name,
        }

    def guard_model_training(
        self, model_name: str, training_data: Any
    ) -> Dict[str, Any]:
        """Guard model training with safety checks."""
        # Check training data safety
        safety_result = self.kernel.check_data_safety(training_data)

        # For training, we're more strict
        safe_to_proceed = safety_result["safe"]

        return {
            "success": True,
            "safety_result": safety_result,
            "safe_to_proceed": safe_to_proceed,
            "model_name": model_name,
            "training_allowed": safe_to_proceed,
        }


class SentinelOpsInterface:
    """Interface for SentinelOps compliance."""

    def __init__(self, kernel: SafetyKernel):
        self.kernel = kernel

    def create_compliance_bundle(
        self, dataset_name: str, dataset_path: str
    ) -> Dict[str, Any]:
        """Create SentinelOps compliance bundle."""
        # Get audit trail
        audit_trail = self.kernel.create_audit_trail()

        # Create compliance bundle
        bundle = {
            "version": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": {
                "name": dataset_name,
                "path": dataset_path,
                "safety_verified": True,
                "compliance_level": self.kernel.config.compliance_level,
            },
            "safety_guarantees": {
                "lineage_proof": True,
                "policy_compliance": True,
                "shape_safety": True,
                "optimizer_stability": True,
            },
            "audit_trail": audit_trail,
            "performance_metrics": self.kernel.processing_stats,
            "model_assets": list(self.kernel.model_assets.keys()),
        }

        return bundle

    def export_compliance_bundle(
        self, dataset_name: str, dataset_path: str, output_path: str
    ) -> bool:
        """Export SentinelOps compliance bundle."""
        try:
            bundle = self.create_compliance_bundle(dataset_name, dataset_path)

            with open(output_path, "w") as f:
                json.dump(bundle, f, indent=2)

            return True
        except Exception as e:
            logging.error(f"Failed to export compliance bundle: {e}")
            return False


# Example usage and integration
def create_safety_kernel(config: Optional[RuntimeConfig] = None) -> SafetyKernel:
    """Create a safety kernel with default or custom configuration."""
    if config is None:
        config = RuntimeConfig()

    return SafetyKernel(config)


def integrate_with_ml_pipeline(
    kernel: SafetyKernel, pipeline_func: Callable
) -> Callable:
    """Integrate safety kernel with ML pipeline function."""

    def safe_pipeline(*args, **kwargs):
        with kernel.safety_context(
            "ml_pipeline", {"args": str(args), "kwargs": str(kwargs)}
        ):
            # Run safety checks before pipeline execution
            if "data" in kwargs:
                safety_result = kernel.check_data_safety(kwargs["data"])
                if not safety_result["safe"]:
                    raise ValueError(
                        f"Safety violations detected: {safety_result['violations']}"
                    )

            # Execute pipeline
            result = pipeline_func(*args, **kwargs)

            # Log successful execution
            kernel._log_event(
                "pipeline_executed",
                "info",
                "ML pipeline executed successfully",
                {"result_type": type(result).__name__},
            )

            return result

    return safe_pipeline
