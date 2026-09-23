"""Infer entry for Ming sliders.

Sampling numbers below are the upstream layer-decomposition defaults.
Adapter execution uses the shared stamp and is not implemented in this scaffold.
"""

from __future__ import annotations

import argparse

from ming.surfaces import (
    BASE_CFG,
    BASE_MODEL_ID,
    BASE_RESOLUTION,
    BASE_STEPS,
    HUB_MODEL_ID,
    HUB_PUBLISHED,
    stamp,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Ming particle-slider infer stub. Imports the shared stamp and does not sample."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Import winning_formulation() and print the product surface.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    loaded = stamp()
    if not args.dry_run:
        raise SystemExit(
            "Ming inference is not wired. Product weights are not published at "
            f"{HUB_MODEL_ID}. Base weights stay at {BASE_MODEL_ID}. "
            "Use --dry-run to import the stamp."
        )
    low, high = loaded.spec["recommended_range"]
    print(f"architecture_id={loaded.architecture_id}")
    print(f"formulation_id={loaded.formulation_id}")
    print(f"formulation_provisional={loaded.formulation_provisional}")
    print(f"recommended_range={low},{high}")
    print(f"base={BASE_MODEL_ID}")
    print(f"base_steps={BASE_STEPS}")
    print(f"base_cfg={BASE_CFG}")
    print(f"base_resolution={BASE_RESOLUTION}")
    print(f"hub={HUB_MODEL_ID}")
    print(f"hub_published={HUB_PUBLISHED}")
    return 0
