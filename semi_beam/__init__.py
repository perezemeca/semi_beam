from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

# Bootstrap para permitir `python -m semi_beam` desde la raiz del repo
# sin instalar el paquete (usa src/semi_beam).
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
_src_pkg = Path(__file__).resolve().parents[1] / "src" / "semi_beam"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))  # type: ignore[attr-defined]
