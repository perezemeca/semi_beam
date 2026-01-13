from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import (
    PointForce, DistUniform, PointMoment,
    NormalizedPointForce, NormalizedDistUniform, NormalizedPointMoment
)
from semi_beam.domain.supports import FixedSupport, TandemSupport, DirectionalSupport
from semi_beam.domain.unknowns import UnknownUniformLoad


@dataclass
class FBDData:
    beam: Beam
    point_forces: List[NormalizedPointForce]
    dist_loads: List[NormalizedDistUniform]
    moments: List[NormalizedPointMoment]
    notes: List[str]


@dataclass(frozen=True)
class BeamCase:
    """
    Caso completo para el motor:
      - cargas conocidas (puntuales, distribuidas, momentos)
      - soportes (algunos con posición fija, tándem con posición a calcular)
      - carga faltante (q uniforme) a calcular
    """
    beam: Beam

    # Cargas conocidas (no incluir aquí la q faltante)
    point_forces: List[PointForce]
    dist_loads: List[DistUniform]
    moments: List[PointMoment]

    # Soportes / reacciones
    kingpin: FixedSupport                 # perno rey (Rp1)
    tandem: TandemSupport                 # tándem (Rt) con x desconocida

    directional: Optional[DirectionalSupport] = None  # Rd (x depende del tándem)
    hitch: Optional[FixedSupport] = None              # plato de enganche (Rp2) - bitren

    # Incógnita: q uniforme
    unknown_uniform: UnknownUniformLoad = UnknownUniformLoad()
