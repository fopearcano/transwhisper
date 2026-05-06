from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


def main() -> int:
    from voice_lan_stt.gui import main as package_main

    return package_main()


if __name__ == "__main__":
    raise SystemExit(main())
