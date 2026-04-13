# path: src/semi_beam/ui/memoria_header_dialog.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass
class MemoriaHeaderValues:
    cliente: str = ""
    proyecto: str = ""
    autor: str = ""
    revision: str = ""
    extra_linea: str = ""


class MemoriaHeaderDialog(QDialog):
    """Diálogo para capturar encabezado de Memoria de Cálculo desde UI.

    - No depende del motor ni del exportador.
    - Devuelve strings (vacíos si el usuario no completa).
    """

    def __init__(self, parent: Optional[QWidget] = None, defaults: Optional[Dict[str, str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Encabezado — Memoria de cálculo")
        self.setModal(True)

        defaults = defaults or {}

        self._cliente = QLineEdit(defaults.get("cliente", ""))
        self._proyecto = QLineEdit(defaults.get("proyecto", ""))
        self._autor = QLineEdit(defaults.get("autor", ""))
        self._revision = QLineEdit(defaults.get("revision", "A"))
        self._extra = QLineEdit(defaults.get("extra_linea", ""))

        # UX: placeholders
        self._cliente.setPlaceholderText("Ej: Lambert Remolques")
        self._proyecto.setPlaceholderText("Ej: Semirremolque X / Obra / Interno")
        self._autor.setPlaceholderText("Ej: Matias")
        self._revision.setPlaceholderText("Ej: A")
        self._extra.setPlaceholderText("Opcional: observaciones / norma / versión")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        form.addRow("Cliente:", self._cliente)
        form.addRow("Proyecto:", self._proyecto)
        form.addRow("Autor:", self._autor)
        form.addRow("Revisión:", self._revision)
        form.addRow("Extra:", self._extra)

        hint = QLabel("Estos datos se insertan en el encabezado del PDF. Campos opcionales.")
        hint.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout()
        lay.addWidget(hint)
        lay.addLayout(form)
        lay.addWidget(btns)

        self.setLayout(lay)
        self.resize(520, 220)

    def values(self) -> MemoriaHeaderValues:
        return MemoriaHeaderValues(
            cliente=self._cliente.text().strip(),
            proyecto=self._proyecto.text().strip(),
            autor=self._autor.text().strip(),
            revision=self._revision.text().strip(),
            extra_linea=self._extra.text().strip(),
        )

    def values_dict(self) -> Dict[str, str]:
        v = self.values()
        return {
            "cliente": v.cliente,
            "proyecto": v.proyecto,
            "autor": v.autor,
            "revision": v.revision,
            "extra_linea": v.extra_linea,
        }
