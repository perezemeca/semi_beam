from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


STUDY_FILE_VERSION = 1


def save_study_file(path: str | Path, payload: Dict[str, Any]) -> None:
    data = dict(payload or {})
    data.setdefault("version", STUDY_FILE_VERSION)
    out_path = Path(path)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_study_file(path: str | Path) -> Dict[str, Any]:
    in_path = Path(path)
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("El archivo de estudio no contiene un objeto JSON válido.")
    version = raw.get("version", STUDY_FILE_VERSION)
    if int(version) > STUDY_FILE_VERSION:
        raise ValueError(
            f"Versión de estudio no soportada: {version}. "
            f"Versión máxima compatible: {STUDY_FILE_VERSION}."
        )
    raw["version"] = int(version)
    return raw
