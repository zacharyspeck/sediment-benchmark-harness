"""Put the repo root on sys.path so `harness` and `graders` import as siblings."""

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
