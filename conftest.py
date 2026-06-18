import os
import sys

# Make the `ppgr` package importable in tests without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
