calculeitor (semi_beam) - ETAPA 1

Aplicacion de escritorio para calculo de viga isostatica en:
- Acoplado
- Semirremolque
- Bitren (primera especie)

Incluye:
- FBD + diagramas V(x), M(x)
- Resolucion de equilibrio (q faltante + posicion de tandem)
- Pestana "Semirremolque - Reacciones" con calculo en tiempo real de reacciones y busqueda de configuracion
- Verificacion de seccion a flexion
- Export de imagenes JPG
- Export de memoria en PDF
- Export de memoria en DOCX (base template)

Nota operativa:
- Word convierte DOCX->PDF de forma manual.

Requisitos
- Windows 11
- Python 3.14.2

Estructura clave
- `scripts/run_app.py` (arranque app)
- `scripts/smoke_check.py` (prueba minima sin abrir UI)
- `scripts/prepare_release_assets.py` (genera icono/template si faltan)
- `src/semi_beam/...` (paquete principal)
- `assets/branding/calculeitor.ico` (icono)
- `assets/templates/memoria_base.docx` (template memoria)
- `calculeitor.spec` (build PyInstaller one-file)

Comandos (PowerShell)

1) Instalar dependencias
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Correr la app
```powershell
.venv\Scripts\python.exe scripts\run_app.py
```
Opcional (si `src` esta en PYTHONPATH o editable):
```powershell
.venv\Scripts\python.exe -m semi_beam
```

3) Ejecutar smoke_check
```powershell
.venv\Scripts\python.exe scripts\smoke_check.py
```
Genera en carpeta temporal:
- `FBD.jpg`
- `V.jpg`
- `M.jpg`
- `Memoria - Smoke.docx`

4) Generar EXE one-file con PyInstaller
```powershell
.venv\Scripts\python.exe scripts\prepare_release_assets.py
.venv\Scripts\pyinstaller.exe --clean calculeitor.spec
```
Salida esperada:
- `dist\calculeitor.exe`

5) Ejecutar tests
```powershell
.venv\Scripts\python.exe -m pytest tests
```

Notas de build
- El `.spec` empaqueta:
  - `src/semi_beam/data/materials_kgcm2.txt`
  - `assets/templates/memoria_base.docx`
  - `assets/branding/calculeitor.ico`
- Nombre de producto/EXE: `calculeitor`.

Licencia
- Uso interno / privado.
