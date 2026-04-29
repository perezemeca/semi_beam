from __future__ import annotations

from PySide6.QtCore import Qt

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QLineEdit, QStyledItemDelegate, QDoubleSpinBox

from semi_beam.ui.number_parsing import (
    normalize_decimal_text,
    normalize_line_edit_text,
    parse_user_float,
    try_parse_user_float,
)


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
        ed.textEdited.connect(lambda _text, line_edit=ed: normalize_line_edit_text(line_edit, allow_negative=self.minv < 0.0))
        return ed

    def setEditorData(self, editor, index):
        s = index.data()
        editor.setText("" if s is None else normalize_decimal_text(str(s), allow_negative=self.minv < 0.0))

    def setModelData(self, editor, model, index):
        t = normalize_decimal_text(editor.text() or "", allow_negative=self.minv < 0.0)
        if t in {"", "-", ".", "-."}:
            model.setData(index, "")
            return
        v = try_parse_user_float(t, allow_negative=self.minv < 0.0)
        if v is None or v < self.minv or v > self.maxv:
            model.setData(index, "")
            return

        if self.decimals <= 0:
            model.setData(index, str(int(round(v))))
        else:
            s = f"{v:.{self.decimals}f}".rstrip("0").rstrip(".")
            model.setData(index, s)


class SpinBoxDelegate(QStyledItemDelegate):
    """
    Editor numérico de tabla con texto normalizado.
    Permite estados parciales y confirma el valor con rango min/max al cerrar.
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
        ed.textEdited.connect(lambda _text, line_edit=ed: normalize_line_edit_text(line_edit, allow_negative=self.minv < 0.0))
        return ed

    def setEditorData(self, editor, index):
        txt = index.data()
        editor.setText("" if txt is None else normalize_decimal_text(str(txt), allow_negative=self.minv < 0.0))

    def setModelData(self, editor, model, index):
        t = normalize_decimal_text(editor.text() or "", allow_negative=self.minv < 0.0)
        if t in {"", "-", ".", "-."}:
            model.setData(index, "")
            return
        v = try_parse_user_float(t, allow_negative=self.minv < 0.0)
        if v is None or v < self.minv or v > self.maxv:
            model.setData(index, "")
            return
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
    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            self.lineEdit().textEdited.connect(self._normalize_editor_text)
        except Exception:
            pass

    def _allows_negative(self) -> bool:
        return float(self.minimum()) < 0.0

    def _normalize_editor_text(self, _text: str = "") -> None:
        normalize_line_edit_text(self.lineEdit(), allow_negative=self._allows_negative())

    def validate(self, text: str, pos: int):
        normalized = normalize_decimal_text(text or "", allow_negative=self._allows_negative())
        if normalized in {"", "-", ".", "-."}:
            return (QValidator.Intermediate, text, pos)
        value = try_parse_user_float(normalized, allow_negative=self._allows_negative())
        if value is None:
            return (QValidator.Intermediate, text, pos)
        if float(self.minimum()) <= value <= float(self.maximum()):
            return (QValidator.Acceptable, text, pos)
        return (QValidator.Intermediate, text, pos)

    def valueFromText(self, text: str) -> float:
        value = try_parse_user_float(text or "", allow_negative=self._allows_negative())
        if value is None:
            return float(self.minimum())
        return float(value)

    def textFromValue(self, value: float) -> str:
        text = super().textFromValue(value)
        return normalize_decimal_text(text, allow_negative=self._allows_negative())
