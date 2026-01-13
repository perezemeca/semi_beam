# semi_beam

Aplicación de escritorio (Windows) para análisis de **vigas isostáticas** aplicadas a chasis de **semirremolques / acoplados / bitren**.

Incluye:
- Carga de **fuerzas puntuales**, **cargas distribuidas** y **momentos**.
- Cálculo de equilibrio: **reacción faltante como carga distribuida uniforme** y **posición de tándem**.
- Gráficos: **FBD**, **V(x)** y **M(x)**.
- Verificación de sección a flexión (doble T idealizada + materiales tabulados) con preview y tabla.
- Exportación de gráficos a **JPG (1200 dpi)**.

---

## Requisitos

- Windows 10/11  
- Python **3.10+** (recomendado 3.11)
- Dependencias (principales): PySide6, numpy, matplotlib, reportlab

> Nota: Si necesitás compatibilidad con Windows 7, suele implicar restricciones de versión (por ejemplo, PySide6 no es la mejor opción para Win7). En ese caso, conviene fijar versiones y validar toolchain explícitamente.

---

## Instalación (Windows)

Desde la raíz del proyecto:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Ejecución

### Opción A — Script de ejecución
```bat
.venv\Scripts\python.exe scripts\run_app.py
```

### Opción B — Como módulo
Si el paquete está instalado en editable:
```bat
pip install -e .
python -m semi_beam
```

---

## Convenciones de cálculo (resumen)

- Eje X: izquierda → derecha, `0 … L` (puede haber cargas fuera del tramo).
- Fuerzas: `+Fy` hacia arriba.
- Distribuidas: `q(x)` positiva hacia abajo.
- Diagramas V(x), M(x): sagging positivo.
- Unidades: kg, mm (y M reportado en kg·cm cuando corresponde).

---

## Empaquetado (opcional) con PyInstaller

Si más adelante querés generar ejecutable:

```bat
pip install pyinstaller
pyinstaller --noconfirm --clean --windowed ^
  --name "semi_beam" ^
  scripts\run_app.py
```

Notas:
- Si usás assets (íconos, archivos .txt de materiales), probablemente necesites `--add-data`.
- Recomendación: cuando estabilices el build, generar y **versionar** un `.spec` para builds reproducibles.

---

## Licencia

- Si el repositorio es **privado/uso interno**, podés omitir `LICENSE`.
- Si lo vas a compartir o publicar, incluí una licencia explícita (por ejemplo, MIT).

Ver archivo `LICENSE`.
