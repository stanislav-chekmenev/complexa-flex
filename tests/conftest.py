"""Worktree-local test bootstrap.

Ensures the worktree's `src/` is imported, not the package installed at the main
checkout. The worktree may be ahead of the installed editable install, so we
prepend it to `sys.path`.
"""

import sys
from pathlib import Path

_WORKTREE_ROOT = Path(__file__).resolve().parents[1]
_SRC = _WORKTREE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
