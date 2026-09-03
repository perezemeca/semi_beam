Calculeitor (semi_beam)

Aplicacion de escritorio para calculo estructural preliminar de vigas en:
- Acoplado
- Semirremolque
- Bitren
- Calculo y verificacion

Incluye:
- FBD, diagramas V(x), M(x) y deformada.
- Verificacion de seccion a flexion.
- Guardado/carga de estudios .sbeam.
- Exportacion JPG y memoria DOCX.

Pestanas
- Acoplado, Semirremolque y Bitren usan el flujo principal de unidad.
- Calculo y verificacion es una pestana independiente y conserva su flujo propio.

Modos de carga en Acoplado/Semirremolque/Bitren
1. Carga distribuida equivalente
   - Modo historico/default.
   - Usa configuracion de ejes y reacciones conocidas.
   - Calcula q equivalente para cerrar equilibrio.
   - Estudios .sbeam antiguos cargan en este modo.

2. Cargas reales
   - El usuario ingresa las cargas en la tabla Cargas.
   - No hay plantillas ni subcasos de contenedor, maquinaria o carga concentrada.
   - Tipos: Puntual, Distribuida, Momento.
   - No genera q equivalente automatica.
   - Calcula reacciones, FBD, V(x), M(x) y verificacion con las cargas reales.
   - En Bitren conserva Rp2.

Tabla Cargas
- Columnas: Tipo, Magnitud, Posicion / centro [mm], Longitud [mm].
- Puntual: magnitud y posicion.
- Distribuida: magnitud total, centro y longitud.
- Momento: magnitud y posicion.
- Longitud aplica a distribuidas.

Reglas importantes
- En algunos casos L es largo carrozable, no el limite estructural total.
- x > L puede ser valido si pertenece fisicamente al sistema.
- x < 0 no debe aceptarse.
- Calculo y verificacion tambien acepta x > L cuando corresponda, pero no tiene el modo Cargas reales.

Semirremolque 1+1+1
- Tres ejes individuales.
- 10500 kg por eje.
- Total bruto 31500 kg.
- Se resta direccional 1300 kg y dos ejes 2200 kg.
- Reaccion total resultante 28000 kg.
- Separacion entre ejes 2450 mm.

Levante de eje
- Direccional: carga automatica de 1300 kg.
- 2 o 3 ejes: primer eje con carga automatica de 1200 kg.
- No modifica x_t geometrico.
- Recalcula reacciones para cerrar V/M.
- En Bitren considera Rp2.

Offset direccional
- Se ingresa desde el segundo eje.
- Rango permitido: 2400 mm a 4000 mm.

UI
- Cargas esta en un CollapsibleBox.
- Las pestanas superiores quedan visibles/fijas.
- Bastidor lateral estructural no es seleccionable: el bastidor activo es estructural.
- Deformada esta siempre activa y no aparece como bloque visible de opcion.
- Bitren no muestra opciones de direccional.
- Acerca de Calculeitor es un boton unico en la barra superior, alineado a la derecha.

Comandos PowerShell

Ejecutar app:
```powershell
.\.venv\Scripts\python.exe -m semi_beam
```

Tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

Smoke check:
```powershell
.\.venv\Scripts\python.exe scripts\smoke_check.py
```

Build:
```powershell
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean .\calculeitor.spec
```

RTK:
```powershell
$env:Path = "$PWD\.venv\Scripts;$env:Path"
rtk pytest -p no:cacheprovider
```

Si RTK responde Pytest: No tests collected, usar python -m pytest como validacion principal.

Ver README.md para documentacion completa.
