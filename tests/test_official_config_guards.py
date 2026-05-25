"""Guards in utils.config must fail loudly on bad configs."""

import pytest

from frozen_sam_readout.utils.config import ConfigGuardError, validate_official_guards


_GOOD = {
    "sam": {"model_id": "sam-vit-h", "frozen": True, "use_prompt_encoder": False, "use_mask_decoder": False},
    "evaluation": {"threshold": 0.5},
    "guards": {"fail_if_sam_trainable": True, "fail_if_prompt_encoder_used": True, "fail_if_mask_decoder_used": True},
}


def test_official_good_config_passes():
    validate_official_guards(_GOOD)


def test_sam_trainable_fails():
    bad = {**_GOOD, "sam": {**_GOOD["sam"], "frozen": False}}
    with pytest.raises(ConfigGuardError):
        validate_official_guards(bad)


def test_prompt_encoder_fails():
    bad = {**_GOOD, "sam": {**_GOOD["sam"], "use_prompt_encoder": True}}
    with pytest.raises(ConfigGuardError):
        validate_official_guards(bad)


def test_mask_decoder_fails():
    bad = {**_GOOD, "sam": {**_GOOD["sam"], "use_mask_decoder": True}}
    with pytest.raises(ConfigGuardError):
        validate_official_guards(bad)


def test_wrong_threshold_fails():
    bad = {**_GOOD, "evaluation": {"threshold": 0.7}}
    with pytest.raises(ConfigGuardError):
        validate_official_guards(bad)


def test_allow_nonofficial_skips_guards():
    bad = {**_GOOD, "sam": {**_GOOD["sam"], "frozen": False}}
    validate_official_guards(bad, allow_nonofficial=True)
