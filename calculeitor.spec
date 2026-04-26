# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH).resolve()

datas = [
    (str(ROOT / "src" / "semi_beam" / "data" / "materials_kgcm2.txt"), "semi_beam/data"),
    (str(ROOT / "assets" / "templates" / "memoria_base.docx"), "assets/templates"),
    (str(ROOT / "assets" / "branding" / "calculeitor.ico"), "assets/branding"),
]

a = Analysis(
    [str(ROOT / "scripts" / "run_app.py")],
    pathex=[
        str(ROOT / "src"),
        str(ROOT),
    ],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Calculeitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "branding" / "calculeitor.ico"),
)
