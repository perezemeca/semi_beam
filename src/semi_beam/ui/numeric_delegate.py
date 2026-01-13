from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QLineEdit, QStyledItemDelegate, QDoubleSpinBox


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
