"""VoxFusion HTTP API server package.

Puts the repo's ``src/`` on sys.path so ``voxfusion`` resolves even when the
package is not pip-installed in the active environment (mirrors cli_start.py).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
