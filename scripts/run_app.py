# path: scripts/run_app.py
import os
import sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from semi_beam.services.logging_setup import setup_logging  # NUEVO
logger = setup_logging()  # NUEVO

def _excepthook(exctype, value, tb):
    msg = "".join(traceback.format_exception(exctype, value, tb))
    logger.error("Excepción no capturada:\n%s", msg)
    # Mantener también el comportamiento por defecto (útil si hay consola)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = _excepthook  # NUEVO

from semi_beam.ui.main_window import main

if __name__ == "__main__":
    main()
