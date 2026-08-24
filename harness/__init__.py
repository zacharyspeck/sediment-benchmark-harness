"""benchmark-v1 harness package.

Pure-Python evaluation harness. No model calls live in this package; arms that
would make network calls are declared in config and are hard-disabled unless
explicitly enabled (see harness/arms.py).
"""

__version__ = "1.0.0"
