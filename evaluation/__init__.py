"""Offline and live evaluation of the retrieval engine (Recall@k, Precision@k, MRR@k).

Run it from the repository root::

    python -m evaluation.run --fake-models          # sample dataset, no model server
    python -m evaluation.run                         # models configured in .env

The evaluation reuses the backend code (``backend/app``): the backend directory is
added to the import path here so that no installation step is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if BACKEND_DIR.is_dir() and str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
