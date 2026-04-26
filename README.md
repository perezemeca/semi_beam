# calculeitor (semi_beam)

Aplicacion de escritorio para analisis estructural preliminar de vigas isostaticas aplicadas a chasis de semirremolques, acoplados y bitrenes.

El proyecto permite modelar cargas, apoyos y geometria longitudinal de una viga, obtener reacciones, diagramas de corte y momento, deformada, verificar secciones a flexion y exportar una memoria de calculo en DOCX.

## Stack

- Python
- PySide6
- Matplotlib
- NumPy
- ReportLab
- python-docx

## Requisitos

- Windows 11
- Python 3.11 o superior
- Dependencias de runtime en `requirements.txt`
- Herramientas de prueba y armado en `requirements-dev.txt`

## Ejecucion en Windows

Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ejecutar la aplicacion:

```powershell
python scripts/run_app.py
```

Entrada alternativa:

```powershell
python -m semi_beam
```

## Funcionalidad principal

- Esquema de cuerpo libre para la viga.
- Diagramas `V(x)` y `M(x)`.
- Calculo de deformada.
- Resolucion de equilibrio con carga distribuida faltante y posicion de tandem.
- Pestaña de calculo y verificacion de reacciones.
- Verificador de seccion a flexion.
- Guardado y carga de estudios `.sbeam`.
- Exportacion de imagenes JPG.
- Exportacion de memoria de calculo en DOCX.

## Verificador de seccion

El verificador de seccion calcula una seccion resistente compuesta a partir de dos vigas doble T simetricas y componentes opcionales.

La seccion compuesta puede incluir:

- Dos vigas doble T.
- Bastidor lateral con perfiles C.
- Piso superior.
- Chapon inferior SAE 1010.

El calculo contempla materiales por componente y verificacion por componente. El resultado final se toma a partir del componente gobernante, es decir, el componente con menor factor de seguridad valido.

## Geometria de componentes opcionales

### Bastidor lateral

- Perfil C lateral.
- Espesor: `3/16"` (`4.7625 mm`).
- Alas: `45 mm`.
- Altura configurable: `130 mm` a `170 mm`.
- Ubicacion simetrica respecto del eje central: `X = -1250 mm` y `X = +1250 mm`.
- Apertura de cada perfil hacia el centro de la seccion.
- Alineacion vertical: cara superior del bastidor igual a la cara superior de la planchuela superior.

### Piso superior

- Ancho: `2430 mm`.
- Centrado en `X = 0`.
- Ubicado sobre la planchuela superior.
- Espesores disponibles: `2 mm`, `3 mm`, `4 mm`, `1/8"` y `3/16"`.
- Puede activarse y marcarse como estructural para que participe en la seccion resistente.

### Chapon inferior

- Ancho: `1050 mm`.
- Centrado en `X = 0`.
- Ubicado bajo la planchuela inferior.
- Material: SAE 1010.
- Espesores disponibles: `1/4"`, `5/16"` y `3/8"`.
- Tramo longitudinal considerado: desde `x = 0` hasta `posicion_perno_mm + 1000 mm`.
- Requiere contexto longitudinal de largo de viga y posicion de perno para determinar si aplica en cada estacion.

## Estados no verificables

El verificador distingue fallas resistentes numericas de secciones no verificables por falta de datos.

- `ERR MAT`: falta material admisible para un componente activo incluido en la seccion resistente. La fila no se considera verificada hasta corregir la base de materiales o la asignacion de material.
- `ERR CHAPON`: el chapon esta activo, pero falta contexto longitudinal para evaluar su inclusion. Se debe definir largo de viga y posicion de perno.

Estos estados tambien se exportan a la memoria DOCX sin convertirlos en `FS = 0`.

## Preview de seccion

El verificador incluye un preview de seccion con Matplotlib embebido en PySide6.

- Zoom con rueda del mouse.
- Pan/desplazamiento de la vista.
- Vista ajustable para volver al encuadre completo.
- Dibujo de la seccion compuesta con vigas doble T y componentes opcionales cuando corresponda.

## Memoria DOCX

La exportacion DOCX documenta los resultados principales del estudio y del verificador de seccion.

Para el verificador de seccion incluye:

- Tabla general de verificacion.
- Resultado mostrado en tabla, con los mismos valores visibles en la UI.
- Estado de componentes opcionales.
- Componente gobernante.
- Tabla por componente.
- Imagen de la seccion compuesta.
- Trazabilidad de `ERR MAT` y `ERR CHAPON` sin tratarlos como `FS = 0`.

## Tests

Instalar dependencias de desarrollo:

```powershell
python -m pip install -r requirements-dev.txt
```

Ejecutar la suite:

```powershell
python -m pytest -q
```

Estado esperado actual:

```text
29 passed
```

## Empaquetado

La entrada del ejecutable es:

```text
scripts/run_app.py
```

El archivo de PyInstaller es:

```text
calculeitor.spec
```

Comando de armado:

```powershell
python scripts/prepare_release_assets.py
pyinstaller --clean calculeitor.spec
```

Assets incluidos por el `.spec`:

- `src/semi_beam/data/materials_kgcm2.txt`
- `assets/branding/calculeitor.ico`
- `assets/templates/memoria_base.docx`

Salida esperada:

```text
dist\calculeitor.exe
```

## Estructura clave

- `scripts/run_app.py`: arranque de la aplicacion.
- `scripts/smoke_check.py`: prueba minima sin abrir la UI.
- `scripts/prepare_release_assets.py`: genera icono y template si faltan.
- `src/semi_beam/`: paquete principal.
- `src/semi_beam/data/materials_kgcm2.txt`: base de materiales.
- `src/semi_beam/services/memoria_calculo_docx.py`: exportador DOCX.
- `calculeitor.spec`: configuracion de PyInstaller.

## Nota sobre README.txt

`README.md` es la documentacion principal para GitHub. `README.txt` se conserva sin cambios por compatibilidad e historial del proyecto.

## Licencia

Uso interno / privado.
