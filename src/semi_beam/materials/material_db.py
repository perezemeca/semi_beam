from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MPA_TO_KGCM2 = 10.197162129779  # 1 MPa = 10.197162... kg/cm²


@dataclass(frozen=True)
class Material:
    """
    Material para verificación elástica por tensiones admisibles.

    Campos esperados (mínimos):
      - id
      - thk_min_mm
      - thk_max_mm
      - sigma_adm_kgcm2

    Los demás ayudan a trazabilidad y futuras verificaciones.
    """
    id: str
    family: str = ""
    brand: str = ""
    grade: str = ""
    thk_min_mm: float = 0.0
    thk_max_mm: float = 1e12
    sigma_adm_kgcm2: float = 0.0
    sigma_basis: str = ""
    rm_kgcm2: Optional[float] = None
    rm_basis: str = ""
    notes: str = ""

    def covers_thickness(self, thk_mm: float) -> bool:
        return (thk_mm >= float(self.thk_min_mm) - 1e-9) and (thk_mm <= float(self.thk_max_mm) + 1e-9)


class MaterialDB:
    def __init__(self, materials: List[Material]):
        self.materials: List[Material] = list(materials)
        self.by_id: Dict[str, Material] = {m.id.strip(): m for m in self.materials if m.id.strip()}

    def ids(self) -> List[str]:
        return [m.id for m in self.materials]

    def get(self, mat_id: str) -> Optional[Material]:
        return self.by_id.get((mat_id or "").strip())

    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip()

    @classmethod
    def from_txt(cls, path: str | Path) -> "MaterialDB":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No existe el archivo de materiales: {p}")

        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        rows: List[List[str]] = []
        for ln in lines:
            t = ln.strip()
            if not t:
                continue
            if t.startswith("#") or t.startswith("//"):
                continue
            rows.append([c.strip() for c in t.split(";")])

        if not rows:
            raise ValueError("Archivo de materiales vacío o sin filas válidas.")

        # Detectar header
        header = [h.strip() for h in rows[0]]
        data_rows = rows[1:] if any(x.lower() in {"id", "material", "grade", "sigma_adm_kgcm2", "fy_mpa"} for x in header) else rows

        # Mapear índices de columnas
        def idx(name: str) -> Optional[int]:
            name_l = name.lower()
            for i, h in enumerate(header):
                if h.lower() == name_l:
                    return i
            return None

        # columnas posibles (nuevo formato recomendado)
        i_id = idx("id") or idx("material")
        i_family = idx("family")
        i_brand = idx("brand")
        i_grade = idx("grade")
        i_thk_min = idx("thk_min_mm")
        i_thk_max = idx("thk_max_mm")
        i_sigma_kgcm2 = idx("sigma_adm_kgcm2")
        i_sigma_basis = idx("sigma_basis")
        i_rm_kgcm2 = idx("rm_kgcm2")
        i_rm_basis = idx("rm_basis")
        i_notes = idx("notes")

        # compatibilidad: archivos antiguos con MPa
        i_sigma_mpa = idx("sigma_adm_mpa")
        i_fy_mpa = idx("fy_mpa")
        i_fy_basis = idx("fy_basis")
        i_rm_mpa = idx("rm_mpa")
        i_rm_mpa_basis = idx("rm_basis")

        def get_cell(row: List[str], i: Optional[int]) -> str:
            if i is None:
                return ""
            return row[i] if i < len(row) else ""

        def try_float(s: str) -> Optional[float]:
            t = (s or "").strip().replace(",", ".")
            if t == "":
                return None
            try:
                return float(t)
            except Exception:
                return None

        mats: List[Material] = []
        for r in data_rows:
            mid = cls._norm(get_cell(r, i_id)) if i_id is not None else cls._norm(r[0] if r else "")
            if not mid:
                continue

            thk_min = try_float(get_cell(r, i_thk_min)) or 0.0
            thk_max = try_float(get_cell(r, i_thk_max)) or 1e12

            sigma_kgcm2 = try_float(get_cell(r, i_sigma_kgcm2))
            sigma_basis = cls._norm(get_cell(r, i_sigma_basis))

            # fallback si no viene sigma en kg/cm²
            if sigma_kgcm2 is None:
                sigma_mpa = try_float(get_cell(r, i_sigma_mpa))
                if sigma_mpa is not None:
                    sigma_kgcm2 = sigma_mpa * MPA_TO_KGCM2
                else:
                    fy_mpa = try_float(get_cell(r, i_fy_mpa))
                    if fy_mpa is not None:
                        sigma_kgcm2 = fy_mpa * MPA_TO_KGCM2
                        if not sigma_basis:
                            sigma_basis = cls._norm(get_cell(r, i_fy_basis))

            if sigma_kgcm2 is None:
                # si no hay σ_adm, el material no sirve para verificar
                continue

            rm_kgcm2 = try_float(get_cell(r, i_rm_kgcm2))
            if rm_kgcm2 is None:
                rm_mpa = try_float(get_cell(r, i_rm_mpa))
                if rm_mpa is not None:
                    rm_kgcm2 = rm_mpa * MPA_TO_KGCM2

            mats.append(Material(
                id=mid,
                family=cls._norm(get_cell(r, i_family)),
                brand=cls._norm(get_cell(r, i_brand)),
                grade=cls._norm(get_cell(r, i_grade)),
                thk_min_mm=float(thk_min),
                thk_max_mm=float(thk_max),
                sigma_adm_kgcm2=float(sigma_kgcm2),
                sigma_basis=sigma_basis,
                rm_kgcm2=float(rm_kgcm2) if rm_kgcm2 is not None else None,
                rm_basis=cls._norm(get_cell(r, i_rm_basis)) or cls._norm(get_cell(r, i_rm_mpa_basis)),
                notes=cls._norm(get_cell(r, i_notes)),
            ))

        if not mats:
            raise ValueError("No se pudieron cargar materiales: faltan columnas o valores de σ_adm/fy.")

        # ordenar por id para UI estable
        mats.sort(key=lambda m: m.id.upper())
        return cls(mats)


def default_materials_path() -> Path:
    """
    Ruta por defecto esperada dentro del repo.
    Colocá el TXT en:
      src/semi_beam/data/materials_kgcm2.txt
    """
    here = Path(__file__).resolve()
    return here.parents[1] / "data" / "materials_kgcm2.txt"
