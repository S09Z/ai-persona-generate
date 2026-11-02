"""
idol_generate/__main__.py
Enables: uv run -m idol_generate  and  idol-generate  CLI entry point.
Delegates to the project-root generate.py main() function.
"""

import runpy
import sys
from pathlib import Path

# Ensure project root is on path so generate.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).parent.parent.parent / "generate.py"),
        run_name="__main__",
    )
