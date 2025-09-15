# simulacion_covid

Repositorio: `simulacion_covid` — simulaciones de agentes (covid-style) con visualización y física modular.

Estructura recomendada (moderada):

- `simulacion_covid/`
  - `caso1_v2.py`, `caso2.py`, `caso3.py`, `caso4.py` — scripts de ejemplo
  - `sim_visual.py` — utilidades de visualización (captura de frames)
  - `sim_physics.py` — física común (colisiones, correcciones)
  - `requirements.txt` — dependencias
  - `run_all.py` — helper para ejecutar los casos y generar GIFs

Requisitos
```
pip install -r requirements.txt
```

Ejecutar un caso
```
python caso1_v2.py
```

Ejecutar todos
```
python run_all.py
```

Notas
- `sim_physics.py` asume igual masa y restitución 1.0; es minimal.
- Uso recomendado: crear virtualenv, instalar `requirements.txt` y ejecutar `run_all.py`.
- Los artefactos generados (`*.gif`) están ignorados por `.gitignore`.
