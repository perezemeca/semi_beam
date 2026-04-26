Calculeitor (semi_beam)

Aplicación de escritorio para cálculo estructural preliminar de vigas de carrocerías en:
- Acoplado
- Semirremolque
- Bitren (primera especie)

Incluye:
- Esquema de cuerpo libre y diagramas V(x), M(x) y deformada.
- Resolución de equilibrio para q faltante y posición de tándem.
- Pestaña "Cálculo y verificación" para reacciones de semirremolque y búsqueda de configuración.
- Verificación de sección a flexión con materiales y tabla de secciones.
- Guardado y carga de estudios `.sbeam`.
- Exportación de imágenes JPG.
- Exportación de memoria de cálculo en DOCX.

Nota operativa:
- El exportador principal genera DOCX. Si hace falta PDF, se puede convertir desde Word.

Requisitos
- Windows 11
- Python 3.11 o superior
- Dependencias de runtime en `requirements.txt`
- Herramientas opcionales de pruebas/armado en `requirements-dev.txt`

Dependencias
- Ejecución: PySide6, matplotlib, numpy, reportlab y python-docx.
- Pruebas: pytest.
- Armado del ejecutable: pyinstaller.

Estructura clave
- `scripts/run_app.py` (arranque de la app)
- `scripts/smoke_check.py` (prueba mínima sin abrir UI)
- `scripts/prepare_release_assets.py` (genera icono/template si faltan)
- `src/semi_beam/...` (paquete principal)
- `assets/branding/calculeitor.ico` (icono)
- `assets/templates/memoria_base.docx` (template de memoria)
- `calculeitor.spec` (ejecutable PyInstaller de un solo archivo)

Comandos (PowerShell)

1) Instalar dependencias
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para correr pruebas o armar el ejecutable:
```powershell
pip install -r requirements-dev.txt
```

2) Correr la app
```powershell
python scripts/run_app.py
```

Opcional:
```powershell
python -m semi_beam
```

3) Ejecutar smoke_check
```powershell
python scripts/smoke_check.py
```
Genera en carpeta temporal:
- `FBD.jpg`
- `V.jpg`
- `M.jpg`
- `Memoria - Smoke.docx`

4) Correr pruebas
```powershell
pytest -q
```

5) Generar EXE de un solo archivo con PyInstaller
```powershell
python scripts/prepare_release_assets.py
pyinstaller --clean calculeitor.spec
```
Salida esperada:
- `dist\Calculeitor.exe`

Notas de build
- El `.spec` empaqueta:
  - `src/semi_beam/data/materials_kgcm2.txt`
  - `assets/templates/memoria_base.docx`
  - `assets/branding/calculeitor.ico`
- Nombre de producto/EXE: `Calculeitor`.
- Los artefactos generados (`dist/`, `build/`, `.tmp/`, logs, memorias exportadas y estudios `.sbeam`) quedan fuera de Git por `.gitignore`.

Licencia
- Uso interno / privado.
