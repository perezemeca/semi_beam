from __future__ import annotations

import re
from typing import Optional, Dict

LABEL_SIGN_CONVENTION: Dict[str, str] = {
    "Rp1": "up_positive",
    "Rp2": "down_positive",
    "Rt":  "up_positive",
    "Rd":  "up_positive",
    "P":   "down_positive",
    "q":   "down_positive",
}

# Para "P" (incluye P1, P2, ...) fuera de [0,L] -> momento en borde
OUTSIDE_AS_EDGE_MOMENT_LABELS = set()


P_INDEX_RE = re.compile(r"^P(\d+)$", re.IGNORECASE)


def is_indexed_P(label: str) -> bool:
    return bool(P_INDEX_RE.match(label.strip()))


def p_index(label: str) -> Optional[int]:
    m = P_INDEX_RE.match(label.strip())
    if not m:
        return None
    return int(m.group(1))


def label_kind(label: str) -> str:
    """
    Normaliza el label para aplicar convenciones:
      - P1, P2, ... => 'P'
    """
    l = (label or "").strip()
    if l.upper() == "P" or is_indexed_P(l):
        return "P"
    return l


def next_free_p_index(used: set[int]) -> int:
    k = 1
    while k in used:
        k += 1
    return k


def to_internal_Fy(label: str, value_user: float) -> float:
    """
    Convierte el valor ingresado por el usuario a fuerza interna Fy (+up).
    """
    kind = label_kind(label)
    conv = LABEL_SIGN_CONVENTION.get(kind, "down_positive")
    if conv == "up_positive":
        return float(value_user)           # + => up
    return -float(value_user)              # + => down => Fy negativo


def to_internal_w_up(label: str, q_user: float) -> float:
    """
    Distribuida uniforme ingresada por usuario. Para q: positivo hacia abajo.
    Interno w_up > 0 => hacia arriba.
    """
    kind = label_kind(label)
    conv = LABEL_SIGN_CONVENTION.get(kind, "down_positive")
    if conv == "up_positive":
        return float(q_user)
    return -float(q_user)

