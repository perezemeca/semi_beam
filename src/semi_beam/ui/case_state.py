from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List


@dataclass
class CaseState:
    """
    Estado por TAB (Acoplado / Semirremolque / Bitren).
    Se guarda TODO lo necesario para que no se pisen datos.
    """
    # Motor inputs (valores numéricos y selección config)
    motor: Dict[str, Any] = field(default_factory=dict)

    # Snapshots de tablas (listas de filas: cada fila es lista[str])
    tbl_points: List[List[str]] = field(default_factory=list)
    tbl_dists: List[List[str]] = field(default_factory=list)
    tbl_moments: List[List[str]] = field(default_factory=list)

    # Estado del verificador de sección (lo que el panel necesite)
    section_state: Optional[Dict[str, Any]] = None

    # Cache del “modo” actual: entradas vs solución
    mode: str = "inputs"  # "inputs" | "solved"

    # Cache de la última corrida (para replot rápido)
    cached: Optional[Tuple[Any, Any, Any, Any, str]] = None  # (beam_plot, points, dists, moms, note_text)

    # Extra: texto de notas ya armado (seleccionable)
    note_text: str = ""
