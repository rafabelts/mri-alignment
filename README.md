# MRI Alignment — Deformable Alignment of 2D MR Images

Deformable image registration of 2D cine-MRI frames for tumor motion tracking in
MR-guided radiotherapy. Given a **fixed** frame (reference) and a **moving** frame
(later time point) from the same cine sequence, the goal is to predict the dense
displacement vector field (DVF) that warps one onto the other, so that anatomy
(and the tumor) can be tracked frame to frame.

Two learning-based registration models and a classical baseline are trained/run
and compared against each other:

- **VoxelMorph** (Balakrishnan et al.), diffeomorphic variant, via the
  [`voxelmorph`](https://github.com/voxelmorph/voxelmorph) package.
- **TransMorph-diff** (`src/transmorph.py`) — an in-house 2D, lightweight
  reinterpretation of TransMorph-diff (Chen et al., 2021), built on the
  probabilistic registration framework of Dalca et al. (2018). It is a CNN
  encoder/decoder with a small Transformer self-attention bottleneck, and a
  probabilistic head that predicts a mean + log-variance velocity field
  (sampled during training, mean-only at inference). It reuses VoxelMorph's
  diffeomorphic integration (`VecInt`) and warping (`SpatialTransformer`) so
  both models share the same coordinate conventions and training/eval code
  paths (`model(source, target, registration=True) -> (moved, pos_flow)`).
- **Classical registration** (`src/classical_registration.py`) — a SimpleITK
  B-Spline baseline (mutual-information-driven, multi-resolution), with no
  learned parameters.

VoxelMorph and TransMorph-diff are trained with direct supervision against
ground-truth DVFs (Charbonnier EPE loss + smoothness regularization;
TransMorph-diff additionally adds a KL term for its probabilistic head). All
three methods are evaluated with the same metrics so results are directly
comparable: EPE (mm), % negative Jacobian (folding), SSIM, and
tumor-segmentation-based Dice / TRE (mm) / Hausdorff distance (mm) - all
physical-unit metrics go through the case's real spacing/origin/direction
(`EvaluationMetric._physical_points` / `_pixel_vector_to_physical` in
`src/evaluate.py`), not just a bare spacing multiply.

VoxelMorph and TransMorph are trained and evaluated with **nested
cross-validation** (outer/inner `StratifiedKFold` over patients, hyperparameter
search in the inner loop, pooled outer-test metrics as the headline result).

## Status

- [x] Data preprocessing (load `.mha`, normalize, mask, pad, patch)
- [x] Nested cross-validation (outer/inner `StratifiedKFold`, patient-level, stratified by cohort, no leakage)
- [x] VoxelMorph training/eval pipeline
- [x] TransMorph-diff (custom 2D probabilistic model)
- [x] Classical (non-DL) B-Spline registration baseline
- [x] Metrics: EPE (mm), % negative Jacobian, SSIM, Dice, TRE (mm), Hausdorff (mm)
- [x] Per-architecture best-model selection (majority-vote hyperparameters + best seed)
- [x] Qualitative comparison figures (VoxelMorph vs TransMorph vs Classical vs GT)
- [x] Cross-method quantitative results table + combined comparison plot

## Project structure

```
mri-alignment/
├── config.py                     # Centralized paths + hyperparameters (see "Configuration")
├── src/
│   ├── compat.py                 # Python 3.11+ compatibility shim, must be imported before voxelmorph/neurite
│   ├── preprocessing.py          # .mha loading, z-score normalization, anatomy mask, padding, patch extraction
│   ├── dataset.py                # Patient split (split_patients, cv_splits) + PyTorch Dataset (patches from full images)
│   ├── models.py                 # Model factory (build_model) for voxelmorph / transmorph
│   ├── transmorph.py             # TransMorph-diff: custom 2D probabilistic registration model
│   ├── classical_registration.py # Classical SimpleITK B-Spline baseline (file-path and array-based entry points)
│   ├── losses.py                 # Charbonnier EPE + smoothness loss
│   ├── train.py                  # Training loop (early stopping, LR scheduling) + inference benchmark
│   ├── evaluate.py               # Patch reconstruction + EPE/Jacobian/SSIM/Dice/TRE/Hausdorff metrics (mm-aware)
│   ├── visualize.py              # Qualitative plots: patches, reconstructed fixed/moving/warped/DVF
│   └── io_utils.py                # External-image inference: read/write .mha, pad/crop, denormalize
├── scripts/
│   ├── nested_cv.py                    # Nested CV + hyperparameter search + best-model selection, per architecture
│   ├── evaluate_checkpoints.py         # Re-run metrics on already-trained checkpoints (e.g. after a metric change)
│   ├── evaluate_classical_registration.py # Evaluate the classical baseline with the same metrics as the DL models
│   ├── build_results_table.py          # Combine all three methods into one CSV + comparison plot + significance test
│   └── generate_comparison_figure.py   # Side-by-side VoxelMorph vs TransMorph vs Classical qualitative comparison figure
├── notebooks/
│   └── exploracion.ipynb         # Exploratory analysis / scratch notebook
├── checkpoints/                  # Saved model weights (git-ignored, kept via .gitkeep)
└── pyproject.toml                # Dependencies (managed with uv)
```

## Data

The project expects the **TrackRad** dataset layout, one folder per patient
sequence, grouped by cohort (`A_*`, `B_*`, `C_*`, ...):

```
data/TrackRad/
└── A_001/
    ├── SynthesizedCine/
    │   ├── img_000.mha        # fixed frame (reference)
    │   └── img_XXX.mha        # moving frames
    ├── DVFReverse/
    │   └── dvfReverseXXX.mha  # ground-truth DVF for each moving frame
    └── SynthesizedSegmentations/
        ├── seg_000.mha        # tumor segmentation for the fixed frame
        └── seg_XXX.mha        # tumor segmentation for each moving frame
```

Patients are split at the patient level, stratified by cohort letter, so all
frames from the same patient always stay together. Two split strategies exist
in `src/dataset.py`:
- `cv_splits()` — nested outer/inner `StratifiedKFold`, used by `nested_cv.py`.
- `split_patients()` — a single fixed train/val/test split, kept for any
  standalone use outside the nested CV pipeline.

By default the dataset lives at `./data/TrackRad`, checkpoints at
`./checkpoints`, and outputs (figures, csv, exports) at `./outputs`. All three
can be overridden with environment variables (see `config.py`):

```
MRI_DATA_DIR=/path/to/TrackRad
MRI_CHECKPOINT_DIR=/path/to/checkpoints
MRI_OUTPUTS_DIR=/path/to/outputs
```

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```bash
# Run the full nested CV + hyperparameter search + best-model selection for one architecture
uv run python scripts/nested_cv.py --model voxelmorph
uv run python scripts/nested_cv.py --model transmorph

# Re-aggregate + re-plot an in-progress or finished run without training anything
uv run python scripts/nested_cv.py --model voxelmorph --plot-only

# Re-run metrics on checkpoints already trained (e.g. after changing EvaluationMetric)
uv run python scripts/evaluate_checkpoints.py --model voxelmorph

# Evaluate the classical B-Spline baseline on the full dataset with the same metrics
uv run python scripts/evaluate_classical_registration.py

# Build one combined results table + comparison plot + significance test across all three methods
uv run python scripts/build_results_table.py

# Generate a qualitative VoxelMorph vs TransMorph vs Classical comparison figure for specific cases
uv run python scripts/generate_comparison_figure.py --cases A_024:095 B_021:017
```

## Configuration

`config.py` centralizes every path and hyperparameter used across the project
(data/checkpoint/output directories, image target size, train/val/test split
ratios, VoxelMorph architecture parameters, training hyperparameters — batch
size, learning rate, epochs, early-stopping patience, LR scheduler — and loss
weights). Edit it directly, or override the three data/output paths via the
`MRI_DATA_DIR` / `MRI_CHECKPOINT_DIR` / `MRI_OUTPUTS_DIR` environment
variables described above. The nested CV pipeline's own design constants
(`OUTER_K`, `INNER_K`, the hyperparameter grid, epoch budgets, seed count) live
at the top of `scripts/nested_cv.py`, not in `config.py`.
