"""Runner simple para ejecutar los 4 casos y guardar GIFs.
Ejecuta cada caso como un script independiente. Ajusta el intérprete si es necesario.
"""
import subprocess
import sys
import os

root = os.path.dirname(__file__)
paths = ["caso1_v2.py","caso2.py","caso3.py","caso4.py"]
for p in paths:
    full = os.path.join(root, p)
    print(f"Ejecutando {p}...")
    res = subprocess.run([sys.executable, full])
    print(f"{p} exit={res.returncode}")
