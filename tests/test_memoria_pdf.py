# path: tests/test_memoria_pdf.py
import os
import tempfile
from datetime import datetime

from semi_beam.services.memoria_calculo_pdf import (
    export_memoria_pdf, MemoriaHeader, MemoriaCaso, MemoriaResultados, MemoriaSeccion
)


def test_export_memoria_pdf_creates_file():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "memoria.pdf")

        header = MemoriaHeader(titulo="Test Memoria", fecha=datetime.now())
        caso = MemoriaCaso(
            unidad="Test",
            L_carrozable_mm=1000.0,
            L_viga_total_mm=1200.0,
            descripcion_config="Dummy",
            apoyos=[("Rp1", "x=0; R=0"), ("Rt", "x=500; R=0")],
            cargas=[("P1", "x=200; P=100")],
        )
        resultados = MemoriaResultados(
            q_user_kgmm=0.1,
            x_t_mm=500.0,
            x_d_mm=None,
            residual_Fy=0.0,
            residual_M0=0.0,
            extremos_V=[],
            extremos_M=[],
        )
        seccion = MemoriaSeccion(
            materiales=[("Sup", "Mat / σadm=1000")],
            fs_min=2.0,
            n_vigas=2,
            parametros=[("t_top", "1/4")],
            tabla=[["Sec","x"],["S1","0"]],
        )

        export_memoria_pdf(out, header=header, caso=caso, resultados=resultados, seccion=seccion, imagenes={})
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
