# Calculeitor (semi_beam)

Calculeitor / semi_beam es una aplicacion de escritorio para calculo estructural preliminar de vigas aplicadas a carrocerias y chasis de Acoplado, Semirremolque y Bitren. Tambien incluye una pestana independiente de `Calculo y verificacion` para analisis de reacciones y configuraciones.

La aplicacion permite cargar geometria longitudinal, apoyos, cargas y secciones; generar esquema de cuerpo libre (FBD), diagramas de corte `V(x)`, momento `M(x)`, deformada, verificacion de seccion a flexion y memoria de calculo DOCX.

## Stack

- Python
- PySide6
- Matplotlib
- NumPy
- ReportLab
- python-docx

## Pestañas principales

- `Acoplado`: resuelve el caso longitudinal del acoplado y permite verificar secciones.
- `Semirremolque`: resuelve configuraciones de semirremolque, incluyendo configuraciones con direccional y `1+1+1`.
- `Bitren`: resuelve Bitren de primera especie, incluyendo el segundo perno rey `Rp2`.
- `Calculo y verificacion`: pestana independiente para calculo/verificacion de reacciones y busqueda de configuracion. Conserva su flujo propio y no usa el nuevo selector de modo de carga.

El modo `Cargas reales` aplica actualmente solo a `Acoplado`, `Semirremolque` y `Bitren`. No fue agregado a `Calculo y verificacion`.

## Modos de carga

Las pestanas `Acoplado`, `Semirremolque` y `Bitren` tienen un selector `Modo de carga` con dos opciones.

### Carga distribuida equivalente

Es el modo historico y el valor por defecto.

- Usa la configuracion de unidad/ejes y reacciones conocidas.
- Calcula una carga distribuida equivalente `q` para cerrar equilibrio.
- Mantiene el comportamiento previo de la aplicacion.
- Los estudios `.sbeam` antiguos cargan por defecto en este modo.
- El levante de eje, cuando esta activo, se incorpora al equilibrio final y recalcula reacciones.

### Cargas reales

En este modo el usuario ingresa directamente las cargas reales en la tabla `Cargas`.

- No existen subcasos ni plantillas tipo contenedor, maquinaria o carga concentrada.
- La app no interpreta el objeto transportado.
- Solo interpreta:
  - `Tipo`
  - `Magnitud`
  - `Posicion / centro`
  - `Longitud`
- Tipos soportados:
  - `Puntual`
  - `Distribuida`
  - `Momento`
- No se genera `q` equivalente automatica.
- Las reacciones, FBD, `V(x)`, `M(x)` y verificacion se calculan desde las cargas reales ingresadas.
- En Bitren se conserva `Rp2` y su influencia en el resultado.

Limitacion actual: en `Cargas reales`, la posicion del tandem se obtiene desde la geometria/configuracion vigente. No hay todavia un optimizador automatico nuevo de posicion admisible de tandem para este modo.

## Tabla unica de cargas

En `Acoplado`, `Semirremolque` y `Bitren`, las cargas se ingresan en una tabla unica dentro del bloque colapsable `Cargas`.

Columnas:

- `Tipo`
- `Magnitud`
- `Posicion / centro [mm]`
- `Longitud [mm]`

Uso por tipo:

- `Puntual`: usa magnitud en kg y posicion.
- `Distribuida`: usa magnitud total en kg, centro y longitud. Internamente se convierte a intensidad uniforme usando `magnitud / longitud`.
- `Momento`: usa magnitud en kg*mm y posicion.
- `Longitud` aplica solo a cargas distribuidas.

La tabla se guarda y restaura en estudios `.sbeam`. Tambien se conserva compatibilidad con estructuras legacy de cargas separadas cuando existen.

## Reglas de posicion respecto de L

En varios flujos `L` representa el largo carrozable, no necesariamente todo el dominio estructural donde puede actuar una carga o reaccion.

- Una posicion `x > L` puede ser valida si pertenece fisicamente al sistema.
- Una posicion `x < 0` no debe aceptarse.
- No debe usarse rigidamente `0 <= x <= L` como unica regla cuando `x > L` es fisicamente valido.

Esto aplica especialmente a Bitren y a `Calculo y verificacion`, donde una reaccion, carga o centro de ejes puede quedar mas alla del largo carrozable.

## Bitren y Rp2

Bitren incluye el segundo perno rey `Rp2`.

- `Rp2` se define mediante `x_Rp2 relativo a L [mm]` y `Rp2 [Kg]`.
- En el equilibrio base, `Rp2` participa como apoyo/reaccion del modelo.
- En levante de eje, el recalculo final de reacciones conserva `Rp2`; no debe desaparecer del FBD ni de `V(x)` / `M(x)`.
- En modo `Cargas reales`, `Rp2` tambien se conserva.

## Semirremolque 1+1+1

La configuracion `1+1+1` representa tres ejes individuales.

- Cada eje individual: `10500 kg`.
- Total bruto: `31500 kg`.
- Se resta peso de direccional: `1300 kg`.
- Se resta peso de dos ejes: `2200 kg`.
- Reaccion total resultante: `28000 kg`.
- Separacion entre ejes: `2450 mm`.
- Se integra con la logica existente de Semirremolque y con el levante de direccional.

## Levante de eje

El levante de eje simula una carga puntual descendente automatica.

- Configuraciones con direccional: agrega `1300 kg`.
- Configuraciones de 2 o 3 ejes: levante del primer eje agrega `1200 kg`.
- El levante entra en el equilibrio final.
- No recalcula la posicion geometrica del tandem `x_t`.
- Si recalcula reacciones para cerrar `V(x)` y `M(x)`.
- En Bitren, el recalculo considera `Rp2`.

## Offset direccional

El offset del direccional se ingresa en la UI desde la posicion del segundo eje.

- Rango normativo permitido: `2400 mm` a `4000 mm`.
- La app puede transformar internamente ese valor si el calculo usa otra referencia.
- La UI debe hablar en terminos de la referencia del usuario: distancia desde el segundo eje.

## Calculo y verificacion

`Calculo y verificacion` es una pestana independiente.

- Conserva su flujo propio.
- No recibe por ahora el modo `Cargas reales`.
- Su tabla de cargas y sus controles no fueron reemplazados por el selector de modo de las pestanas de unidad.
- Debe aceptar posiciones mayores que `L` cuando correspondan fisicamente al sistema.
- Debe seguir bloqueando posiciones negativas.

## Panel lateral y UI

- El bloque `Cargas` esta dentro de un `CollapsibleBox`.
- Las pestanas superiores permanecen visibles/fijas.
- `Bastidor lateral estructural` ya no es seleccionable: el bastidor lateral se considera siempre estructural cuando esta activo.
- El bloque `Deformada` ya no aparece como opcion visible; la deformada esta siempre activa en el flujo actual.
- En Bitren no aparecen opciones referentes al direccional.
- `Acerca de Calculeitor` aparece como boton unico en la barra superior, alineado a la derecha, sin menu `Ayuda`.

## Verificador de seccion

El verificador calcula una seccion resistente compuesta a partir de dos vigas doble T simetricas y componentes opcionales.

Componentes:

- Dos vigas doble T.
- Bastidor lateral con perfiles C.
- Piso superior.
- Chapon inferior SAE 1010.
- Doble alma por seccion/fila.
- Refuerzo de bastidor opcional.

El resultado final se toma desde el componente gobernante, es decir, el componente con menor factor de seguridad valido.

### Bastidor lateral

- Perfil C lateral.
- Espesor: `3/16"` (`4.7625 mm`).
- Alas: `45 mm`.
- Altura configurable: `130 mm` a `170 mm`.
- Ubicacion simetrica respecto del eje central: `X = -1250 mm` y `X = +1250 mm`.
- Apertura hacia el centro de la seccion.
- Cuando esta activo, se considera estructural.

### Refuerzo de bastidor

- Refuerzo vertical opcional paralelo al alma del bastidor lateral.
- Ubicacion fija: `40 mm` hacia el centro desde el eje del bastidor lateral (`x = -1210 mm` y `x = 1210 mm` con semiancho `1250 mm`).
- Altura igual a la altura configurada del bastidor lateral.
- Espesores disponibles: `3/16"`, `1/4"`, `5/16"` y `3/8"`.
- Aporta al calculo de centroide, inercia, modulo resistente y FS.

### Piso superior

- Puede incluirse o no.
- Cuando se incluye, aporta como componente estructural.
- Ancho editable, valor inicial `2430 mm`.
- Centrado en `X = 0`.
- Ubicado sobre la planchuela superior.
- Espesores disponibles: `2 mm`, `3 mm`, `4 mm`, `1/8"` y `3/16"`.

### Chapon inferior

- Ancho: `1050 mm`.
- Centrado en `X = 0`.
- Ubicado bajo la planchuela inferior.
- Material: SAE 1010.
- Espesores disponibles: `1/4"`, `5/16"` y `3/8"`.
- El usuario ingresa el largo del chapon en mm.
- Tramo longitudinal considerado: desde `x = 0` hasta `largo_chapon_mm`.

### Doble alma

- Se activa por seccion/fila.
- Reemplaza el alma central por dos almas simetricas respecto del centro geometrico de la planchuela.
- El usuario define la distancia desde el centro geometrico de la planchuela hasta la cara interna de cada alma.
- La configuracion se guarda/carga en `.sbeam` y se exporta a DOCX.

## Estados no verificables

El verificador distingue fallas resistentes numericas de secciones no verificables por falta de datos.

- `ERR MAT`: falta material admisible para un componente activo incluido en la seccion resistente.
- `ERR CHAPON`: el chapon esta activo, pero el largo ingresado no es valido.
- `ERR DOBLE ALMA`: la doble alma esta activa, pero la geometria no es verificable.

Estos estados tambien se exportan a la memoria DOCX sin convertirlos en `FS = 0`.

## Diagramas e inspeccion

Luego de resolver un caso, se puede mover el mouse sobre `V(x)`, `M(x)` o deformada para consultar el valor local.

- El marcador indica el punto inspeccionado.
- La caja fija del eje activo muestra posicion `X` y magnitud local.
- El FBD no tiene inspeccion hover.
- La inspeccion es solo visual.
- El hover se oculta antes de exportar JPG o DOCX.

### Estrategia de actualizacion del canvas

- Cambios de cargas, apoyos, reacciones o geometria longitudinal reconstruyen FBD, V, M y deformada.
- Cambios de material o tension admisible actualizan el verificador de seccion sin invalidar el canvas principal.
- En Acoplado, Semirremolque y Bitren, los cambios geometricos de la seccion conservan FBD/V/M y actualizan el preview, el verificador y la deformada cuando el estado resuelto y los diagramas existentes siguen siendo validos.
- En esas tres pestanas, si faltan cache, diagrama o artists V/M validos, la actualizacion parcial usa como fallback el replot completo.
- En `Calculo y verificacion`, los cambios geometricos de la seccion conservan su flujo actual de replot completo; no utilizan la ruta parcial anterior.
- Un cambio de tab o el final de un resize siempre ejecuta un replot completo. Ese replot tiene prioridad sobre cualquier actualizacion parcial pendiente.

La separacion responde a las dependencias fisicas: cargas y apoyos definen FBD/V/M, `I(x)` afecta la deformada y el material admisible afecta la verificacion resistente.

## Persistencia .sbeam

Los estudios `.sbeam` guardan/restauran:

- Entradas de unidad.
- Modo de carga activo.
- Tabla de cargas.
- Estado de levante.
- Configuracion de verificador de seccion.
- Datos necesarios para restaurar estudios legacy.

Los estudios viejos sin `load_mode` cargan con `Carga distribuida equivalente`.

## Memoria DOCX

La exportacion DOCX documenta resultados principales del estudio y del verificador.

- En modo `Carga distribuida equivalente`, mantiene la logica historica y reporta la `q` calculada.
- En modo `Cargas reales`, la `q` equivalente figura como no aplicada/no utilizada cuando corresponde.
- Incluye tablas de verificacion, componentes opcionales, componente gobernante, seccion compuesta y estados no verificables.

## Ejecucion local

Crear y activar entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ejecutar desde source:

```powershell
.\.venv\Scripts\python.exe -m semi_beam
```

Entrada alternativa:

```powershell
.\.venv\Scripts\python.exe scripts\run_app.py
```

## Validaciones

Instalar dependencias de desarrollo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Validacion principal:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

Smoke check:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_check.py
```

Chequeo de diff:

```powershell
git diff --check
```

Si RTK esta disponible:

```powershell
$env:Path = "$PWD\.venv\Scripts;$env:Path"
rtk pytest -p no:cacheprovider
```

En algunos entornos `rtk pytest -p no:cacheprovider` puede responder `Pytest: No tests collected`. Si ocurre, reportarlo como limitacion del entorno y usar `python -m pytest` como validacion principal.

## Empaquetado

El archivo de PyInstaller es:

```text
calculeitor.spec
```

Comando de armado:

```powershell
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean .\calculeitor.spec
```

Salida esperada:

```text
dist\Calculeitor.exe
```

Assets incluidos por el `.spec`:

- `src/semi_beam/data/materials_kgcm2.txt`
- `assets/branding/calculeitor.ico`
- `assets/templates/memoria_base.docx`

## Estructura clave

- `scripts/run_app.py`: arranque de la aplicacion.
- `scripts/smoke_check.py`: prueba minima sin abrir la UI.
- `scripts/prepare_release_assets.py`: genera icono y template si faltan.
- `src/semi_beam/`: paquete principal.
- `src/semi_beam/ui/main_window.py`: UI principal, tabs de unidad, modos de carga, FBD/V/M y DOCX.
- `src/semi_beam/ui/reactions_tab.py`: pestana independiente `Calculo y verificacion`.
- `src/semi_beam/ui/section_check_panel.py`: verificador de seccion.
- `src/semi_beam/engine/equilibrium.py`: equilibrio con `q` equivalente.
- `src/semi_beam/engine/reactions.py`: calculo de reacciones con apoyos conocidos.
- `src/semi_beam/engine/diagrams.py`: diagramas `V(x)` y `M(x)`.
- `src/semi_beam/services/study_storage.py`: lectura/escritura `.sbeam`.
- `src/semi_beam/services/memoria_calculo_docx.py`: exportador DOCX.
- `calculeitor.spec`: configuracion de PyInstaller.

## No implementado

No documentar como disponible:

- Optimizador automatico nuevo de posicion admisible de tandem en `Cargas reales`.
- Subcasos o plantillas de contenedor, maquinaria o carga concentrada.
- Modo `Cargas reales` en `Calculo y verificacion`.

## Nota sobre README.txt

`README.md` es la documentacion principal para GitHub. `README.txt` se conserva como guia breve por compatibilidad e historial del proyecto.

## Licencia

Uso interno / privado.
