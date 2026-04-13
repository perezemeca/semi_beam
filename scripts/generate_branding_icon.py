from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from semi_beam.services.branding import ensure_calculeitor_icon


def main():
    icon_path = ensure_calculeitor_icon(os.path.join(ROOT, "assets", "branding", "calculeitor.ico"))
    print(icon_path)


if __name__ == "__main__":
    main()
