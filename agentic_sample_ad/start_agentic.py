from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover
    package_parent = Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

from agentic_sample_ad.main_agent.start_agentic import main


if __name__ == "__main__":
    main()
