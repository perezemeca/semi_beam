from __future__ import annotations

from PySide6.QtCore import Qt
import re

from PySide6.QtGui import QDoubleValidator, QValidator
from PySide6.QtWidgets import QLineEdit, QStyledItemDelegate, QDoubleSpinBox


TABLE_TEXT_COLOR = "#111111"
TABLE_INPUT_BG = "#FFFFFF"
TABLE_READONLY_BG = "#EEF2F6"
TABLE_OK_BG = "#DFF3E4"
TABLE_ERROR_BG = "#FFDCDC"
TABLE_SELECTION_BG = "#2F6DB2"
TABLE_SELECTION_TEXT = "#FFFFFF"
TABLE_GRID_COLOR = "#C7CED6"
TABLE_HEADER_BG = "#E8EDF3"


def combo_cell_style(background: str = TABLE_INPUT_BG) -> str:
    return (
        "QComboBox {"
        f"background-color: {background};"
        f"color: {TABLE_TEXT_COLOR};"
        f"selection-background-color: {TABLE_SELECTION_BG};"
        f"selection-color: {TABLE_SELECTION_TEXT};"
        "padding: 0 6px;"
        "}"
        "QComboBox QAbstractItemView {"
        f"background-color: {TABLE_INPUT_BG};"
        f"color: {TABLE_TEXT_COLOR};"
        f"selection-background-color: {TABLE_SELECTION_BG};"
        f"selection-color: {TABLE_SELECTION_TEXT};"
        "}"
    )


def apply_table_readability_style(tbl) -> None:
    tbl.setStyleSheet(
        f"""
        QTableWidget {{
            background-color: {TABLE_INPUT_BG};
            color: {TABLE_TEXT_COLOR};
            gridline-color: {TABLE_GRID_COLOR};
            selection-background-color: {TABLE_SELECTION_BG};
            selection-color: {TABLE_SELECTION_TEXT};
            alternate-background-color: #F8FAFC;
        }}
        QHeaderView::section {{
            background-color: {TABLE_HEADER_BG};
            color: {TABLE_TEXT_COLOR};
            padding: 4px 6px;
            border: 1px solid {TABLE_GRID_COLOR};
            font-weight: 600;
        }}
        QTableWidget::item:selected {{
            background-color: {TABLE_SELECTION_BG};
            color: {TABLE_SELECTION_TEXT};
        }}
        QLineEdit, QAbstractSpinBox {{
            background-color: {TABLE_INPUT_BG};
            color: {TABLE_TEXT_COLOR};
            selection-background-color: {TABLE_SELECTION_BG};
            selection-color: {TABLE_SELECTION_TEXT};
            border: 1px solid #7F8C9A;
            padding: 0 4px;
        }}
        QComboBox {{
            background-color: {TABLE_INPUT_BG};
            color: {TABLE_TEXT_COLOR};
            selection-background-color: {TABLE_SELECTION_BG};
            selection-color: {TABLE_SELECTION_TEXT};
            padding: 0 6px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {TABLE_INPUT_BG};
            color: {TABLE_TEXT_COLOR};
            selection-background-color: {TABLE_SELECTION_BG};
            selection-color: {TABLE_SELECTION_TEXT};
        }}
        """
    )


class NullableFloatDelegate(QStyledItemDelegate):
    """
    Editor numérico que:
    - Acepta SOLO números (con punto o coma)
    - Permite celda vacía
    - Evita texto inválido
    """
    def __init__(self, parent=None, *, decimals: int = 2, minv: float = -1e18, maxv: float = 1e18):
        super().__init__(parent)
        self.decimals = int(decimals)
        self.minv = float(minv)
        self.maxv = float(maxv)

    def createEditor(self, parent, option, index):
        ed = QLineEdit(parent)
        ed.setAlignment(Qt.AlignCenter)
        ed.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: {TABLE_INPUT_BG};
                color: {TABLE_TEXT_COLOR};
                selection-background-color: {TABLE_SELECTION_BG};
                selection-color: {TABLE_SELECTION_TEXT};
            }}
            """
        )

        val = QDoubleValidator(self.minv, self.maxv, self.decimals, ed)
        val.setNotation(QDoubleValidator.StandardNotation)  # sin científica
        ed.setValidator(val)
        return ed

    def setEditorData(self, editor, index):
        s = index.data()
        editor.setText("" if s is None else str(s))

    def setModelData(self, editor, model, index):
        t = (editor.text() or "").strip().replace(",", ".")
        if t == "":
            model.setData(index, "")
            return
        try:
            v = float(t)
        except Exception:
            model.setData(index, "")
            return

        if self.decimals <= 0:
            model.setData(index, str(int(round(v))))
        else:
            s = f"{v:.{self.decimals}f}".rstrip("0").rstrip(".")
            model.setData(index, s)


class SpinBoxDelegate(QStyledItemDelegate):
    """
    Editor con QDoubleSpinBox para tablas (solo números).
    Nota: un SpinBox no puede quedar "vacío" real, por eso usamos:
      - mínimo = 0
      - specialValueText = ""  -> 0 se muestra en blanco
    """
    def __init__(
        self,
        parent=None,
        *,
        minv: float = 0.0,
        maxv: float = 1e12,
        decimals: int = 1,
        step: float = 10.0,
        blank_is_min: bool = True,
    ):
        super().__init__(parent)
        self.minv = float(minv)
        self.maxv = float(maxv)
        self.decimals = int(decimals)
        self.step = float(step)
        self.blank_is_min = bool(blank_is_min)

    def createEditor(self, parent, option, index):
        sp = QDoubleSpinBox(parent)
        sp.setRange(self.minv, self.maxv)
        sp.setDecimals(self.decimals)
        sp.setSingleStep(self.step)
        sp.setKeyboardTracking(False)
        sp.setAlignment(Qt.AlignCenter)
        sp.setStyleSheet(
            f"""
            QDoubleSpinBox {{
                background-color: {TABLE_INPUT_BG};
                color: {TABLE_TEXT_COLOR};
                selection-background-color: {TABLE_SELECTION_BG};
                selection-color: {TABLE_SELECTION_TEXT};
            }}
            """
        )
        if self.blank_is_min:
            sp.setSpecialValueText("")  # si vale min -> se ve vacío
        return sp

    def setEditorData(self, editor, index):
        txt = index.data()
        t = "" if txt is None else str(txt).strip().replace(",", ".")
        if t == "":
            editor.setValue(self.minv)
            return
        try:
            editor.setValue(float(t))
        except Exception:
            editor.setValue(self.minv)

    def setModelData(self, editor, model, index):
        v = float(editor.value())
        if self.blank_is_min and abs(v - self.minv) < 1e-12:
            model.setData(index, "")
            return
        s = f"{v:.{self.decimals}f}".rstrip("0").rstrip(".")
        model.setData(index, s)


class FlexibleDoubleSpinBox(QDoubleSpinBox):
    """
    SpinBox que acepta coma o punto decimal y permite visual en blanco
    usando specialValueText="" con mínimo como centinela.
    """
    _PAT = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)$")

    def validate(self, text: str, pos: int):
        t = (text or "").strip()
        if t == "":
            return (QValidator.Intermediate, text, pos)
        if self._PAT.match(t):
            return (QValidator.Acceptable, text, pos)
        return (QValidator.Invalid, text, pos)

    def valueFromText(self, text: str) -> float:
        t = (text or "").strip().replace(",", ".")
        if t == "":
            return float(self.minimum())
        try:
            return float(t)
        except Exception:
            return float(self.minimum())
