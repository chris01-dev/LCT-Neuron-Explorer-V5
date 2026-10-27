# PyInstaller spec: a single LCTNeuronExplorer.exe that carries the whole app (code, static UI, shipped caches).
a = Analysis(
    ["bootstrap.py"],
    datas=[
        ("explorer", "explorer"),
        ("sweep", "sweep"),
        ("pyproject.toml", "."),
        ("uv.lock", "."),
        ("desktop.py", "."),
        (".python-version", "."),
        ("README.md", "."),
    ],
    excludes=["tkinter"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="LCTNeuronExplorer",
    icon="LCTNeuronExplorer.ico",
    console=True,
    onefile=True,
)
