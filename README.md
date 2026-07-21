# MRI Alignment — Deformable Alignment of 2D MR Images

Registro deformable supervisado de imágenes 2D de RM (cine-MRI) usando
deep learning (VoxelMorph difeomórfico, con TransMorph planeado como
segunda arquitectura para comparación), sobre datos sintéticos con
ground-truth DVF (TrackRad / Lugez et al.).

## Setup

Requiere [`uv`](https://docs.astral.sh/uv/):

```bash
# instalar uv (una sola vez)
curl -LsSf https://astral.sh/uv/install.sh | sh        # Mac/Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# clonar y sincronizar dependencias
git clone <repo-url>
cd mri-alignment
uv sync
```

`uv sync` instala automáticamente la versión correcta de PyTorch según la
plataforma (CPU/MPS en Mac, CUDA en Windows/Linux — ver `pyproject.toml`).

## Datos

Coloca el dataset descomprimido en `data/TrackRad/` (o define la variable
de entorno `MRI_DATA_DIR` apuntando a otra ubicación):

```bash
export MRI_DATA_DIR="/ruta/a/TrackRad"
```

Estructura esperada por paciente:
```
TrackRad/
└── A_001/
    ├── SynthesizedCine/           # img_000.mha, img_001.mha, ...
    ├── DVFReverse/                # dvfReverse001.mha, ...
    └── SynthesizedSegmentations/  # seg_000.mha, ... (opcional, para Dice/TRE)
```

## Uso

```bash
# entrenar
uv run python scripts/train_model.py

# evaluar el mejor checkpoint sobre el test set
uv run python scripts/evaluate_model.py --checkpoint best_voxelmorph.pt

# exploración / notebooks
uv run jupyter lab
```

## Estructura del proyecto

```
├── config.py             # rutas e hiperparámetros centralizados
├── src/
│   ├── compat.py          # compatibilidad con voxelmorph en Python 3.11+
│   ├── preprocessing.py   # lectura .mha, normalización, máscara, patches
│   ├── dataset.py         # split de pacientes, MRICineDataset
│   ├── models.py          # construcción de VoxelMorph
│   ├── losses.py          # Charbonnier-EPE + smoothness
│   ├── train.py           # loop de entrenamiento + benchmark
│   ├── evaluate.py        # reconstrucción de patches + métricas
│   └── visualize.py       # visualización de patches y resultados
├── scripts/
│   ├── train_model.py     # entry point de entrenamiento
│   └── evaluate_model.py  # entry point de evaluación
├── notebooks/              # exploración / prototipado rápido
├── checkpoints/            # modelos entrenados (no versionados en git)
└── data/                   # dataset (no versionado en git)
```

## Hallazgos importantes (ver reporte semanal para detalle completo)

- **Dirección del DVF**: el ground-truth satisface
  `moving(p) = fixed(p + DVF(p))`. El modelo debe llamarse como
  `model(img_fixed, img_moving, registration=True)` — el orden importa;
  invertirlo produce un campo en el sistema de referencia opuesto,
  entrenable (EPE bajo) pero que no alinea imágenes correctamente.
- **`registration=True`** es obligatorio para obtener el flujo integrado
  (diffeomórfico) a resolución completa; el default devuelve un campo sin
  integrar a media resolución.
- La pérdida de similitud (LNCC) se descartó por inestabilidad numérica en
  regiones de fondo con varianza casi cero.
