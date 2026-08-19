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
- **CNNTransformerSVF2D** (`src/proposal/cnn_transformer_svf_2d.py`) — an in-house lightweight 2D
  CNN encoder/decoder with skip connections and global Transformer attention
  at the bottleneck, inspired by TransMorph (Chen et al., 2021) but not a port
  of its Swin architecture. A deterministic head predicts a two-channel
  stationary velocity field. It reuses VoxelMorph's resolution transforms
  (`ResizeTransform`), integration (`VecInt`) and warping (`SpatialTransformer`) so
  both models share the same coordinate conventions and training/eval code
  paths (`model(source, target, registration=True) -> (moved, pos_flow)`).
- **Classical registration** (`src/classical_registration.py`) — a SimpleITK
  B-Spline baseline (mutual-information-driven, multi-resolution), with no
  learned parameters.

VoxelMorph and CNNTransformerSVF2D are trained with direct supervision against
ground-truth DVFs using Charbonnier EPE loss plus smoothness regularization. All
three methods are evaluated with the same metrics so results are directly
comparable: EPE (mm), % negative Jacobian (folding), SSIM, and
region-of-interest-segmentation-based Dice / TRE (mm) / HD95
(mm) (the segmented region is the tumor, an organ-at-risk, or both,
depending on the case) - all
physical-unit metrics go through the case's real spacing/origin/direction
(`EvaluationMetric._physical_points` / `_pixel_vector_to_physical` in
`src/evaluate.py`), not just a bare spacing multiply.

VoxelMorph and CNNTransformerSVF2D are trained and evaluated with **nested
cross-validation** (outer/inner `StratifiedKFold` over patients, hyperparameter
search in the inner loop; seeds are averaged within each outer fold before
the five fold means are summarized as the headline result).

## Status

- [x] Data preprocessing (load `.mha`, normalize, mask, pad, patch)
- [x] Nested cross-validation (outer/inner `StratifiedKFold`, patient-level, stratified by cohort, no leakage)
- [x] VoxelMorph training/eval pipeline
- [x] CNNTransformerSVF2D (custom deterministic CNN--Transformer SVF model)
- [x] Classical (non-DL) B-Spline registration baseline
- [x] Metrics: EPE (mm), % negative Jacobian, SSIM, Dice, TRE (mm), HD95 (mm)
- [x] Per-architecture best-model selection (median/mode-aggregated hyperparameters + best seed)
- [x] Qualitative comparison figures (VoxelMorph vs CNNTransformerSVF2D vs Classical vs GT)
- [x] Cross-method quantitative results table + combined comparison plot

## Project structure

```
mri-alignment/
├── config.py                     # Centralized paths + hyperparameters (see "Configuration")
├── src/
│   ├── compat.py                 # Python 3.11+ compatibility shim, must be imported before voxelmorph/neurite
│   ├── preprocessing.py          # .mha loading, z-score normalization, anatomy mask, padding, patch extraction
│   ├── dataset.py                # Patient split (split_patients, cv_splits) + PyTorch Dataset (patches from full images)
│   ├── models.py                 # Model factory for VoxelMorph / CNNTransformerSVF2D
│   ├── proposal/
│   │   └── cnn_transformer_svf_2d.py # Deterministic 2D CNN--Transformer SVF model
│   ├── classical_registration.py # Classical SimpleITK B-Spline baseline (file-path and array-based entry points)
│   ├── losses.py                 # Charbonnier EPE + smoothness loss
│   ├── train.py                  # Training loop (early stopping, LR scheduling) + inference benchmark
│   ├── evaluate.py               # Patch reconstruction + EPE/Jacobian/SSIM/Dice/TRE/HD95 metrics (mm-aware)
│   ├── visualize.py              # Qualitative plots: patches, reconstructed fixed/moving/warped/DVF
│   ├── io_utils.py               # External-image inference: read/write .mha, pad/crop, denormalize
│   └── utils.py                  # get_device() / set_seed() shared across the pipeline
├── scripts/
│   ├── nested_cv.py                    # Nested CV + hyperparameter search + best-model selection, per architecture
│   ├── evaluate_checkpoints.py         # Re-run metrics on already-trained checkpoints (e.g. after a metric change)
│   ├── evaluate_classical_registration.py # Evaluate the classical baseline with the same metrics as the DL models
│   ├── build_results_table.py          # Combine all three methods into one CSV + comparison plot + significance test
│   └── generate_comparison_figure.py   # Side-by-side VoxelMorph vs CNNTransformerSVF2D vs Classical figure
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
        ├── seg_000.mha        # region-of-interest (tumor/OAR) segmentation, fixed frame
        └── seg_XXX.mha        # region-of-interest (tumor/OAR) segmentation, each moving frame
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
uv run python scripts/nested_cv.py --model cnn_transformer_svf_2d

# Same, but split the search stage's Optuna trials across multiple GPUs
# (one trial per device at a time; final refit and best-model stages stay
# single-threaded on the first device regardless of how many are listed)
uv run python scripts/nested_cv.py --model voxelmorph --devices cuda:0,cuda:1

# Re-aggregate + re-plot an in-progress or finished run without training anything
uv run python scripts/nested_cv.py --model voxelmorph --plot-only

# Re-run metrics on checkpoints already trained (e.g. after changing EvaluationMetric)
uv run python scripts/evaluate_checkpoints.py --model voxelmorph

# Evaluate the classical B-Spline baseline on the full dataset with the same metrics
uv run python scripts/evaluate_classical_registration.py

# Build one combined results table + comparison plot + significance test across all three methods
uv run python scripts/build_results_table.py

# Generate a qualitative VoxelMorph vs CNNTransformerSVF2D vs Classical figure for specific cases
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
