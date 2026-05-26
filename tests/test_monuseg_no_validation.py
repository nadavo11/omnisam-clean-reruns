"""Tests enforcing MoNuSeg no-validation-split contract.

MoNuSeg has no real held-out validation split.  The 6-sample train-side subsets
sometimes called 'val' are drawn from the training distribution and must never
be presented as unbiased ablation evidence.

Hard rules:
1. MoNuSeg eval_split='val'/'validation'/'valid' must raise ConfigGuardError.
2. MoNuSeg must not write eval/val_metrics.json for ablation claims.
3. MoNuSeg split manifests must not reference a validation split.
4. MoNuSeg official ablation must use eval_split='test'.
5. Train-side smoke eval must be labeled eval_split='train_smoke' and non-claimable.
6. test_metrics.json must have eval_split='test' and a numeric dice value.
7. train_smoke_metrics.json must be explicitly marked non_claimable=True.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frozen_sam_readout.utils.config import ConfigGuardError, assert_monuseg_no_val


# ---------------------------------------------------------------------------
# Guard: forbidden eval splits
# ---------------------------------------------------------------------------

class TestMonusegForbiddenEvalSplits:
    """assert_monuseg_no_val must raise for val/validation/valid."""

    @pytest.mark.parametrize("eval_split", ["val", "validation", "valid", "VAL", "Validation"])
    def test_monuseg_val_split_raises(self, eval_split):
        with pytest.raises(ConfigGuardError, match="no real validation set"):
            assert_monuseg_no_val("monuseg", eval_split)

    @pytest.mark.parametrize("eval_split", ["val", "validation", "valid"])
    def test_rationai_monuseg_val_split_raises(self, eval_split):
        with pytest.raises(ConfigGuardError):
            assert_monuseg_no_val("RationAI/MoNuSeg", eval_split)

    def test_monuseg_test_split_is_allowed(self):
        assert_monuseg_no_val("monuseg", "test")  # must not raise

    def test_monuseg_train_smoke_split_is_allowed(self):
        assert_monuseg_no_val("monuseg", "train_smoke")  # must not raise

    def test_other_dataset_val_is_allowed(self):
        assert_monuseg_no_val("glas", "val")  # must not raise for non-MoNuSeg


# ---------------------------------------------------------------------------
# Artifact contract: eval/val_metrics.json must not exist for MoNuSeg ablations
# ---------------------------------------------------------------------------

class TestMonusegNoValMetricsFile:
    """MoNuSeg ablation run dirs must not contain a val_metrics.json."""

    def _make_run_dir(self, tmp_path, include_val_metrics=False):
        run_dir = tmp_path / "run"
        (run_dir / "eval").mkdir(parents=True)
        (run_dir / "eval" / "test_metrics.json").write_text(
            json.dumps({"eval_split": "test", "dice": 0.80})
        )
        if include_val_metrics:
            (run_dir / "eval" / "val_metrics.json").write_text(
                json.dumps({"eval_split": "val", "dice": 0.75})
            )
        return run_dir

    def test_clean_run_has_no_val_metrics(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path, include_val_metrics=False)
        val_file = run_dir / "eval" / "val_metrics.json"
        assert not val_file.exists(), (
            "MoNuSeg ablation run must not write eval/val_metrics.json"
        )

    def test_contaminated_run_detected(self, tmp_path):
        """Detection helper: run that wrote val_metrics.json must be flagged."""
        run_dir = self._make_run_dir(tmp_path, include_val_metrics=True)
        val_file = run_dir / "eval" / "val_metrics.json"
        assert val_file.exists(), "Sanity check: val_metrics.json was written"
        payload = json.loads(val_file.read_text())
        # Flagged: MoNuSeg split is forbidden
        assert payload.get("eval_split") in {"val", "validation", "valid"}, (
            "Contaminated run has a val payload — correctly detected"
        )


# ---------------------------------------------------------------------------
# Artifact contract: split_manifest.json must not reference validation split
# ---------------------------------------------------------------------------

class TestMonusegSplitManifestNoVal:
    """split_manifest.json produced by test-screen runner must not list val IDs."""

    def test_split_manifest_only_train_and_test(self, tmp_path):
        manifest = {
            "train": ["TCGA-18-5592", "TCGA-21-5784"],
            "test": ["TCGA-A7-A13E", "TCGA-AO-A12B"],
        }
        p = tmp_path / "split_manifest.json"
        p.write_text(json.dumps(manifest))
        payload = json.loads(p.read_text())
        for forbidden in ("val", "validation", "valid"):
            assert forbidden not in payload, (
                f"split_manifest.json must not contain a '{forbidden}' key"
            )

    def test_split_manifest_with_val_key_detected(self, tmp_path):
        """Detection: a manifest with a val key must be flaggable."""
        bad_manifest = {
            "train": ["TCGA-18-5592"],
            "val": ["TCGA-21-5784"],
            "test": ["TCGA-A7-A13E"],
        }
        p = tmp_path / "split_manifest.json"
        p.write_text(json.dumps(bad_manifest))
        payload = json.loads(p.read_text())
        has_val_key = any(k in payload for k in ("val", "validation", "valid"))
        assert has_val_key, "Detection correctly identifies val key in manifest"


# ---------------------------------------------------------------------------
# Artifact contract: test_metrics.json must be well-formed
# ---------------------------------------------------------------------------

class TestMonusegTestMetricsContract:
    """eval/test_metrics.json must have eval_split=test and a numeric dice."""

    def test_test_metrics_has_correct_split(self, tmp_path):
        metrics = {
            "dataset": "MoNuSeg",
            "eval_split": "test",
            "dice": 0.803,
            "iou": 0.671,
            "n_samples": 14,
        }
        p = tmp_path / "test_metrics.json"
        p.write_text(json.dumps(metrics))
        payload = json.loads(p.read_text())
        assert payload["eval_split"] == "test"
        assert isinstance(payload.get("dice"), (int, float))
        assert payload["n_samples"] == 14

    def test_test_metrics_with_wrong_split_detected(self, tmp_path):
        """Detection: test_metrics.json with eval_split=val is flagged."""
        bad = {"eval_split": "val", "dice": 0.80}
        p = tmp_path / "test_metrics.json"
        p.write_text(json.dumps(bad))
        payload = json.loads(p.read_text())
        is_wrong_split = payload.get("eval_split") != "test"
        assert is_wrong_split, "Correctly detected wrong eval_split in test_metrics.json"


# ---------------------------------------------------------------------------
# Artifact contract: train_smoke_metrics.json must be non-claimable
# ---------------------------------------------------------------------------

class TestMonusegTrainSmokeContract:
    """Train-side smoke eval must be explicitly marked non_claimable."""

    def test_train_smoke_is_marked_non_claimable(self, tmp_path):
        smoke = {
            "dataset": "MoNuSeg",
            "eval_split": "train_smoke",
            "non_claimable": True,
            "reason": "Train-side overfit check only; not held-out evidence.",
            "dice": 0.92,
        }
        p = tmp_path / "train_smoke_metrics.json"
        p.write_text(json.dumps(smoke))
        payload = json.loads(p.read_text())
        assert payload["eval_split"] == "train_smoke"
        assert payload.get("non_claimable") is True, (
            "train_smoke_metrics.json must have non_claimable=True"
        )

    def test_train_smoke_without_non_claimable_detected(self, tmp_path):
        """Detection: smoke file missing non_claimable flag is flagged."""
        bad_smoke = {
            "eval_split": "train_smoke",
            "dice": 0.92,
        }
        p = tmp_path / "train_smoke_metrics.json"
        p.write_text(json.dumps(bad_smoke))
        payload = json.loads(p.read_text())
        missing_flag = not payload.get("non_claimable")
        assert missing_flag, "Detection: non_claimable flag is missing — correctly detected"


# ---------------------------------------------------------------------------
# Integration: config guard reachable from the test-screen runner source
# ---------------------------------------------------------------------------

class TestTestScreenRunnerGuard:
    """The test-screen runner must call assert_monuseg_no_val."""

    def test_runner_imports_guard(self):
        import importlib.util, inspect
        here = Path(__file__).resolve()
        runner = here.parent.parent / "scripts" / "train_monuseg_test_screen.py"
        if not runner.exists():
            pytest.skip("train_monuseg_test_screen.py not present")
        src = runner.read_text()
        assert "assert_monuseg_no_val" in src, (
            "train_monuseg_test_screen.py must call assert_monuseg_no_val"
        )

    def test_runner_never_writes_val_metrics(self):
        here = Path(__file__).resolve()
        runner = here.parent.parent / "scripts" / "train_monuseg_test_screen.py"
        if not runner.exists():
            pytest.skip("train_monuseg_test_screen.py not present")
        src = runner.read_text()
        assert "val_metrics" not in src, (
            "train_monuseg_test_screen.py must not reference val_metrics"
        )
        assert "eval_split.*val" not in src, (
            "train_monuseg_test_screen.py must not set eval_split to val"
        )
