#!/usr/bin/env python3
"""Sample a Ming design-layer slider from this repo.

    python scripts/infer_ming.py --help
    python scripts/infer_ming.py --dummy --load_lora models/layer-ming-design_lora

Same code as training. ``--load_lora`` skips the train loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ming.infer import main  # noqa: E402


if __name__ == "__main__":
    main()
