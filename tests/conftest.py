from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest


_TMP_ROOT = Path(__file__).resolve().parents[1] / ".pytest_tmp"
_TMP_ROOT.mkdir(exist_ok=True)

os.environ.setdefault("TMP", str(_TMP_ROOT))
os.environ.setdefault("TEMP", str(_TMP_ROOT))
tempfile.tempdir = str(_TMP_ROOT)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "tmp"


def _make_temp_dir(prefix: str = "tmp") -> Path:
    path = _TMP_ROOT / f"{_safe_name(prefix)}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class _WorkspaceTemporaryDirectory:
    def __init__(self, suffix: str | None = None, prefix: str | None = None, dir: str | os.PathLike[str] | None = None):
        base = Path(dir) if dir is not None else _TMP_ROOT
        base.mkdir(parents=True, exist_ok=True)
        name = f"{_safe_name(prefix or 'tmp')}_{uuid.uuid4().hex}{suffix or ''}"
        self.name = str(base / name)
        Path(self.name).mkdir(parents=True, exist_ok=False)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory


@pytest.fixture
def tmp_path(request):
    path = _make_temp_dir(request.node.name)
    yield path
    shutil.rmtree(path, ignore_errors=True)
