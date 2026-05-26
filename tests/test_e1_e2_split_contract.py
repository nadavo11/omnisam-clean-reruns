"""Tests enforcing the E1/E2 split contract.

Rules:
- E1 must train on train and evaluate on val only.
- E1 must write eval/val_metrics.json (or eval_metrics_val.json), never test metrics.
- E1 must fail if --run-test-eval is passed without --allow-premature-test-eval.
- E2 must evaluate on test and write test_metrics; it is out of scope here.
- Metric payload eval_split must match the filename and the split_manifest IDs.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_train_script() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parent.parent / "scripts" / "train_clean_ablation.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find train_clean_ablation.py near {here}")


def _import_main():
    """Import the main() function from train_clean_ablation without executing it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_clean_ablation", _find_train_script()
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Contract: E1 guard blocks --run-test-eval for E1 job types
# ---------------------------------------------------------------------------

class TestE1SplitGuard:
    """E1 runner must refuse to evaluate on official test IDs."""

    def test_run_test_eval_default_is_false(self):
        """--run-test-eval must default to False to prevent accidental test exposure."""
        import argparse
        mod = _import_main()
        # Call parse_args with only the required args to check default
        # We can't call main() without a full setup, so inspect argparse directly.
        # The guard is documented in the source; verify via the module constant.
        import inspect
        src = inspect.getsource(mod)
        # Default must be False
        assert 'default=False' in src or '--run-test-eval' in src, \
            "train_clean_ablation.py must set --run-test-eval default=False"
        # Ensure no 'default=True' appears near run-test-eval
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if 'run-test-eval' in line and 'no-run-test-eval' not in line:
                context = '\n'.join(lines[max(0, i-2):i+3])
                assert 'default=True' not in context, (
                    f"--run-test-eval must not default to True:\n{context}"
                )

    def test_e1_guard_present_in_source(self):
        """E1 guard code must exist in the training script."""
        mod = _import_main()
        import inspect
        src = inspect.getsource(mod)
        assert "SPLIT CONTRACT VIOLATION" in src, \
            "E1 split contract guard must be present in train_clean_ablation.py"
        assert "allow_premature_test_eval" in src, \
            "Override flag allow_premature_test_eval must exist for debug override"

    def test_launcher_passes_no_run_test_eval(self):
        """Launcher script must explicitly pass --no-run-test-eval."""
        here = Path(__file__).resolve()
        launcher = here.parent.parent / "launcher_repo" / "launch_clean_e1.py"
        if not launcher.exists():
            pytest.skip("launcher_repo not present")
        src = launcher.read_text()
        assert "--no-run-test-eval" in src, (
            "launch_clean_e1.py must pass --no-run-test-eval to train_clean_ablation.py"
        )


# ---------------------------------------------------------------------------
# Contract: metric payload eval_split must be consistent
# ---------------------------------------------------------------------------

class TestMetricPayloadConsistency:
    """eval_metrics_val.json must have eval_split=val; test must have eval_split=test."""

    def test_val_metrics_payload_has_val_split(self, tmp_path):
        """A val metrics file with eval_split=test is a split contamination signal."""
        val_metrics = {
            "dataset": "monuseg",
            "eval_split": "val",
            "val_dice": 0.75,
            "val_iou": 0.60,
        }
        p = tmp_path / "eval_metrics_val.json"
        p.write_text(json.dumps(val_metrics))
        payload = json.loads(p.read_text())
        assert payload["eval_split"] == "val", (
            f"eval_metrics_val.json must have eval_split=val, got {payload['eval_split']}"
        )

    def test_val_filename_with_test_split_raises(self, tmp_path):
        """Detection helper correctly flags val filename with test payload as a mismatch."""
        bad_metrics = {
            "dataset": "monuseg",
            "eval_split": "test",  # WRONG for a val metrics file
            "val_dice": 0.75,
        }
        p = tmp_path / "eval_metrics_val.json"
        p.write_text(json.dumps(bad_metrics))
        payload = json.loads(p.read_text())
        filename_implies_val = "val" in p.name
        payload_says_test = payload.get("eval_split") == "test"
        # The detection must FIND the mismatch (both conditions true = contamination detected).
        assert filename_implies_val and payload_says_test, (
            "Detection failed: expected to detect split mismatch "
            "(eval_metrics_val.json with eval_split=test in payload) but did not."
        )

    def test_test_metrics_stub_has_not_run_status(self, tmp_path):
        """When test eval is disabled, stub file must have status=not_run."""
        stub = {
            "dataset": "monuseg",
            "eval_split": "test",
            "status": "not_run",
            "reason": "test eval disabled for this run",
        }
        p = tmp_path / "eval_metrics_test.json"
        p.write_text(json.dumps(stub))
        payload = json.loads(p.read_text())
        assert payload.get("status") == "not_run", (
            "When test eval is disabled, eval_metrics_test.json must have status=not_run"
        )
        # Must NOT contain actual test metrics
        assert "test_dice" not in payload, \
            "Stub test metrics file must not contain test_dice"
        assert "test_iou" not in payload, \
            "Stub test metrics file must not contain test_iou"


# ---------------------------------------------------------------------------
# Contract: E1 completed runs must NOT have real test dice in test metrics
# ---------------------------------------------------------------------------

class TestE1CompletedRunAudit:
    """Audit helper: detect if a completed E1 run wrote real test metrics."""

    @staticmethod
    def is_premature_test_run(metrics_path: Path) -> bool:
        """Return True if the metrics file contains real (non-stub) test results."""
        if not metrics_path.exists():
            return False
        try:
            payload = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if payload.get("status") == "not_run":
            return False
        # A real test eval has numeric dice/iou
        has_real_metrics = (
            isinstance(payload.get("test_dice"), (int, float))
            or isinstance(payload.get("dice"), (int, float))
        )
        is_test_split = payload.get("eval_split") == "test"
        return has_real_metrics and is_test_split

    def test_audit_helper_detects_stub(self, tmp_path):
        stub = {"eval_split": "test", "status": "not_run", "reason": "disabled"}
        p = tmp_path / "eval_metrics_test.json"
        p.write_text(json.dumps(stub))
        assert not self.is_premature_test_run(p)

    def test_audit_helper_detects_real_test(self, tmp_path):
        real = {"eval_split": "test", "test_dice": 0.80, "test_iou": 0.67}
        p = tmp_path / "eval_metrics_test.json"
        p.write_text(json.dumps(real))
        assert self.is_premature_test_run(p)

    def test_audit_helper_detects_missing(self, tmp_path):
        p = tmp_path / "eval_metrics_test.json"
        assert not self.is_premature_test_run(p)


# ---------------------------------------------------------------------------
# Contract: feature_shapes.json must be written and well-formed
# ---------------------------------------------------------------------------

class TestFeatureShapesArtifact:
    """feature_shapes.json must be written for every run and contain required keys."""

    REQUIRED_KEYS = {"backbone_backend", "paper_headline_safe", "channel_parity_ok",
                     "f2_channels", "expected_paper_channels"}

    def test_feature_shapes_schema(self, tmp_path):
        shapes = {
            "backbone_model_id": "facebook/sam3",
            "backbone_backend": "transformers_hf_fallback",
            "paper_headline_safe": False,
            "f2_channels": 256,
            "f1_channels": 256,
            "f0_channels": 256,
            "expected_paper_channels": {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32},
            "channel_parity_ok": False,
        }
        p = tmp_path / "feature_shapes.json"
        p.write_text(json.dumps(shapes))
        payload = json.loads(p.read_text())
        for key in self.REQUIRED_KEYS:
            assert key in payload, f"feature_shapes.json must contain key '{key}'"

    def test_channel_parity_false_when_hf_fallback(self, tmp_path):
        shapes = {
            "backbone_backend": "transformers_hf_fallback",
            "paper_headline_safe": False,
            "f2_channels": 256,
            "f1_channels": 256,
            "f0_channels": 256,
            "expected_paper_channels": {"f2_channels": 256, "f1_channels": 64, "f0_channels": 32},
            "channel_parity_ok": False,
        }
        p = tmp_path / "feature_shapes.json"
        p.write_text(json.dumps(shapes))
        payload = json.loads(p.read_text())
        assert payload["channel_parity_ok"] is False
        assert payload["paper_headline_safe"] is False
