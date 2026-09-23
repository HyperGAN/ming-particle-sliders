#!/usr/bin/env python3
"""Train a Ming Design-Layer particle slider from this repo.

    python scripts/train_ming.py --dry-run

Imports ``particle_sliders.winning_formulation``. Does not vendor the game.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ming.train import main  # noqa: E402
from particle_sliders import winning_formulation  # noqa: E402

if __name__ == "__main__":
    winning_formulation()
    raise SystemExit(main())
