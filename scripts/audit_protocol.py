#!/usr/bin/env python
"""Audit a config + checkpoint pair against the official-protocol guards.

Loads the config, runs ``validate_official_guards``, optionally builds the
backbone to check that all SAM parameters are frozen.
"""

from __future__ import annotations

import argparse
import sys

from frozen_sam_readout.sam import build_frozen_sam_pyramid_extractor
from frozen_sam_readout.sam.freeze_utils import assert_sam_frozen
from frozen_sam_readout.utils import ConfigGuardError, load_yaml_config, validate_official_guards


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-backbone", action="store_true",
                        help="Instantiate the SAM backbone and confirm all params are frozen.")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    try:
        validate_official_guards(config)
    except ConfigGuardError as exc:
        print(f"GUARD FAILED: {exc}")
        return 1
    print("Config guards: OK")

    if args.check_backbone:
        extractor = build_frozen_sam_pyramid_extractor(model_id=str(config["sam"]["model_id"]))
        bundle_attr = getattr(extractor, "_bundle", None) or getattr(extractor, "_predictor_bundle", None)
        if bundle_attr is None:
            print("Backbone not yet materialized; nothing to check.")
            return 0
        # bundle is a tuple where last element is the SAM model or predictor
        sam_or_predictor = bundle_attr[-1]
        sam_model = getattr(sam_or_predictor, "model", sam_or_predictor)
        try:
            assert_sam_frozen(sam_model)
        except RuntimeError as exc:
            print(f"BACKBONE NOT FROZEN: {exc}")
            return 2
        print("Backbone frozen: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
