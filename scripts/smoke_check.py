from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import PointForce
from semi_beam.domain.supports import FixedSupport, TandemSupport, DirectionalSupport
from semi_beam.domain.unknowns import UnknownUniformLoad
from semi_beam.domain.cases import BeamCase
from semi_beam.engine.equilibrium import solve_equilibrium
from semi_beam.engine.normalize import normalize_inputs
from semi_beam.engine.diagrams import build_V_M
from semi_beam.view.renderer_fbd import render_fbd
from semi_beam.view.renderer_vm import render_shear, render_moment
from semi_beam.view.style import RenderStyle
from semi_beam.services.memoria_calculo_docx import (
    export_memoria_docx,
    ensure_memoria_template,
    default_template_path,
    MemoriaHeader,
    MemoriaCaso,
    MemoriaResultados,
    MemoriaSeccion,
)


def _fmt(v: float, dec: int = 2) -> str:
    s = f"{float(v):.{dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def main():
    tmpdir = tempfile.mkdtemp(prefix="semi_beam_smoke_")
    print(f"[smoke] output dir: {tmpdir}")

    beam = Beam(L_mm=10365.0)
    case = BeamCase(
        beam=beam,
        point_forces=[
            PointForce(label="P1", x_mm=2500.0, value_user=3200.0),
            PointForce(label="P2", x_mm=6100.0, value_user=1800.0),
        ],
        dist_loads=[],
        moments=[],
        kingpin=FixedSupport(label="Rp1", x_mm=2200.0, reaction_user=9000.0),
        tandem=TandemSupport(label="Rt", reaction_user=15800.0),
        directional=DirectionalSupport(label="Rd", reaction_user=9200.0, offset_mm=3075.0),
        hitch=None,
        unknown_uniform=UnknownUniformLoad(label="q", span_start_mm=0.0, span_len_mm=10365.0),
    )
    res = solve_equilibrium(case)

    points = res.solved_point_forces
    dists = res.solved_dist_loads
    moms = res.solved_moments
    x_start, x_end = -500.0, beam.L_mm + 500.0
    diag = build_V_M(beam_L_mm=beam.L_mm, point_forces=points, dist_loads=dists, moments=moms, x_start=x_start, x_end=x_end)

    fig = plt.Figure(figsize=(12, 8))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 1.0], hspace=0.58)
    ax_fbd = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0], sharex=ax_fbd)
    ax_m = fig.add_subplot(gs[2, 0], sharex=ax_fbd)
    render_fbd(ax_fbd, normalize_inputs(beam, points, dists, moms), RenderStyle(), y_zoom=1.0, xlim=(x_start, x_end))
    render_shear(ax_v, diag, y_zoom=1.0, xlim=(x_start, x_end))
    render_moment(ax_m, diag, y_zoom=1.0, xlim=(x_start, x_end))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    p_fbd = os.path.join(tmpdir, "FBD.jpg")
    p_v = os.path.join(tmpdir, "V.jpg")
    p_m = os.path.join(tmpdir, "M.jpg")
    fig.savefig(p_fbd, dpi=300, bbox_inches=ax_fbd.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))
    fig.savefig(p_v, dpi=300, bbox_inches=ax_v.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))
    fig.savefig(p_m, dpi=300, bbox_inches=ax_m.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))

    # DOCX (usa template versionado o lo crea si falta)
    template_path = ensure_memoria_template(default_template_path())
    out_docx = os.path.join(tmpdir, "Memoria - Smoke.docx")

    caso = MemoriaCaso(
        unidad="Smoke",
        L_carrozable_mm=beam.L_mm,
        L_viga_total_mm=beam.L_mm,
        descripcion_config="Smoke 1+2 ejes",
        apoyos=[
            ("Rp1", "x=2200 mm; R=9000 kg"),
            ("Rd", f"x={_fmt(res.x_d_mm, 0)} mm; R=9200 kg"),
            ("Rt", f"x={_fmt(res.x_t_mm, 0)} mm; R=15800 kg"),
        ],
        cargas=[("P1", "x=2500 mm; P=3200 kg"), ("P2", "x=6100 mm; P=1800 kg")],
    )
    resultados = MemoriaResultados(
        q_user_kgmm=float(res.q_user_kg_per_mm),
        x_t_mm=float(res.x_t_mm),
        x_d_mm=float(res.x_d_mm) if res.x_d_mm is not None else None,
        residual_Fy=float(res.residual_Fy),
        residual_M0=float(res.residual_M0),
        extremos_V=[],
        extremos_M=[],
    )
    seccion = MemoriaSeccion(
        materiales=[("Planchuela sup", "F36"), ("Planchuela inf", "F36"), ("Alma", "F24")],
        fs_min=2.9,
        n_vigas=2,
        parametros=[],
        tabla=[],
    )
    header = MemoriaHeader(
        titulo="Memoria de Cálculo — Smoke",
        cliente_proyecto="Smoke",
        autor="smoke_check",
        fecha=datetime.now(),
        revision="A",
    )
    imgs = {
        "fbd": p_fbd,
        "v": p_v,
        "m": p_m,
        "sec_a": p_fbd,
        "sec_b": p_v,
        "sec_c": p_m,
        "sec_d": p_fbd,
        "sec_e": p_v,
        "stab_long": p_m,
        "stab_lat": p_fbd,
        "secciones": p_v,
    }
    extras = {
        "dist_perno_mm": 2200.0,
        "peso_eje1_kg": 9200.0,
        "peso_eje2_kg": 15800.0,
        "dist_eje1_mm": float(res.x_d_mm) if res.x_d_mm is not None else 0.0,
        "dist_eje2_mm": float(res.x_t_mm),
        "mmax_kgcm": 0.0,
        "mmax_x_mm": 0.0,
        "alas": "F36/F36",
        "alma": "F24",
        "fy_kgcm2": 2400.0,
        "fs_min_real": 2.9,
        "flex_rows": [],
    }
    export_memoria_docx(
        out_docx,
        template_path=template_path,
        header=header,
        caso=caso,
        resultados=resultados,
        seccion=seccion,
        imagenes=imgs,
        extras=extras,
    )

    print(f"[smoke] FBD: {p_fbd}")
    print(f"[smoke] V:   {p_v}")
    print(f"[smoke] M:   {p_m}")
    print(f"[smoke] DOCX:{out_docx}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
