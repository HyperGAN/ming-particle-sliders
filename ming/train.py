"""Train entry for Ming sliders.

This module only checks shared formulation metadata and records the intended
Hub repo. There is no native Ming update game in this scaffold.
"""

from __future__ import annotations

import argparse

from ming.surfaces import BASE_MODEL_ID, HUB_MODEL_ID, HUB_PUBLISHED, stamp


def build_parser():
    parser = argparse.ArgumentParser(
        description="Ming particle-slider train stub. Imports the shared stamp and does not train."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Import winning_formulation(), require the unmodified stamp, and exit.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    loaded = stamp()
    if not args.dry_run:
        raise SystemExit(
            "Ming training is not wired. Product weights are not published at "
            f"{HUB_MODEL_ID}, and this repo does not vendor the gmix game. "
            "Use --dry-run to import the stamp."
        )
    print(f"architecture_id={loaded.architecture_id}")
    print(f"formulation_id={loaded.formulation_id}")
    print(f"formulation_provisional={loaded.formulation_provisional}")
    print(f"base={BASE_MODEL_ID}")
    print(f"hub={HUB_MODEL_ID}")
    print(f"hub_published={HUB_PUBLISHED}")
    print("execution=scaffold-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
