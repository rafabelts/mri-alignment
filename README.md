# MRI Alignment — Deformable Alignment of 2D MR Images

Deformable image registration of 2D cine-MRI frames for tumor motion tracking in
MR-guided radiotherapy. Given a **fixed** frame (reference) and a **moving** frame
(later time point) from the same cine sequence, the goal is to predict the dense
displacement vector field (DVF) that warps one onto the other, so that anatomy
(and the tumor) can be tracked frame to frame.

Two learning-based registration models are trained and compared against each
other and against a classical (non-learning) baseline:

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
- **Classical registration** (`src/classical_registration.py`) — intended as a
  non-deep-learning baseline. **Not implemented yet** — currently just a stub.

Both DL models are trained with direct supervision against ground-truth DVFs
(Charbonnier EPE loss + smoothness regularization; TransMorph-diff additionally
adds a KL term for its probabilistic head), and are evaluated with the same
metrics so results are directly comparable: EPE, % negative Jacobian (folding),
SSIM, and tumor-segmentation-based Dice / TRE / Hausdorff distance.

## Status

- [x] Data preprocessing (load `.mha`, normalize, mask, pad, patch)
- [x] Patient-level train/val/test split (stratified by cohort, no patient leakage)
- [x] VoxelMorph training/eval pipeline
- [x] TransMorph-diff (custom 2D probabilistic model)
- [x] Metrics: EPE, % negative Jacobian, SSIM, Dice, TRE, Hausdorff
- [x] Multi-seed training + re-evaluation tooling
- [x] External-image inference (outside the training dataset)
- [x] Qualitative comparison figures (VoxelMorph vs TransMorph vs GT)
- [ ] Classical (non-DL) registration baseline — stubbed, not implemented

## Project structure

```
mri-alignment/
├── config.py                     # Centralized paths + hyperparameters (see "Configuration")
├── src/
│   ├── compat.py                 # Python 3.11+ compatibility shim, must be imported before voxelmorph/neurite
│   ├── preprocessing.py          # .mha loading, z-score normalization, anatomy mask, padding, patch extraction
│   ├── dataset.py                # Patient split (train/val/test) + PyTorch Dataset (patches from full images)
│   ├── models.py                 # Model factory (build_model) for voxelmorph / transmorph
│   ├── transmorph.py             # TransMorph-diff: custom 2D probabilistic registration model
│   ├── classical_registration.py # Classical (non-DL) baseline — WIP, not implemented
│   ├── losses.py                 # Charbonnier EPE + smoothness loss
│   ├── train.py                  # Training loop (early stopping, LR scheduling) + inference benchmark
│   ├── evaluate.py               # Patch reconstruction + EPE/Jacobian/SSIM/Dice/TRE/Hausdorff metrics
│   ├── visualize.py              # Qualitative plots: patches, reconstructed fixed/moving/warped/DVF
│   └── io_utils.py                # External-image inference: read/write .mha, pad/crop, denormalize
├── scripts/
│   ├── train_multi_seed.py           # Train a model across N random seeds, report averaged metrics
│   ├── evaluate_model.py             # Evaluate one checkpoint on the test split
│   ├── reevaluate_checkpoints.py     # Re-run metrics on already-trained checkpoints (e.g. after adding a metric)
│   ├── infer_new_images.py           # Run a trained model on a new fixed/moving pair outside the dataset
│   └── generate_comparison_figure.py # Side-by-side VoxelMorph vs TransMorph qualitative comparison figure
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

Patients are split into train/val/test (`src/dataset.py::split_patients`),
stratified by cohort letter, so all frames from the same patient always stay
in the same split.

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
# Train VoxelMorph or TransMorph across several random seeds
uv run python scripts/train_multi_seed.py --model voxelmorph --master-seed 0 --n-runs 5

# Evaluate a single checkpoint on the test split (metrics + inference benchmark)
uv run python scripts/evaluate_model.py --model voxelmorph --checkpoint best_voxelmorph.pt

# Re-run metrics on checkpoints already trained (e.g. after adding a new metric)
uv run python scripts/reevaluate_checkpoints.py --model voxelmorph --seeds 8506 6369 5111 2697 3078

# Run a trained model on a new fixed/moving pair outside the training dataset
uv run python scripts/infer_new_images.py --fixed path/to/fixed.mha --moving path/to/moving.mha --model voxelmorph

# Generate a qualitative VoxelMorph vs TransMorph comparison figure for specific cases
uv run python scripts/generate_comparison_figure.py --cases A_024:095 B_021:017
```

## Configuration

`config.py` centralizes every path and hyperparameter used across the project
(data/checkpoint/output directories, image target size, train/val/test split
ratios, VoxelMorph architecture parameters, training hyperparameters — batch
size, learning rate, epochs, early-stopping patience, LR scheduler — and loss
weights). Edit it directly, or override the three data/output paths via the
`MRI_DATA_DIR` / `MRI_CHECKPOINT_DIR` / `MRI_OUTPUTS_DIR` environment
variables described above.
