# Solar Filament Segmentation 2026

This repository contains the W1 V1 final inference pipeline and trained weight shards for the Solar Filament Segmentation Challenge 2026. The primary final is Kaggle submission **58360434** (public score 0.40). It includes no competition images, annotations, ground-truth masks, precomputed predictions, or submitted CSV files.

## Environment and weight reconstruction

Use Python 3.11 in a Conda environment. From the repository root:

```bash
conda env create -f environment.yml
conda activate solar_solution_rf
python merge_weights.py --root . --destination models
```

`merge_weights.py` verifies every shard against `weights_manifest.json`, reconstructs each model under `models/`, and verifies the reconstructed byte count and SHA-256. All weights and shards are unchanged from the documented RF and RF+R34 alternatives. The merged `models/` directory is generated locally and is not committed.

Install Jupyter in the Conda environment if needed:

```bash
conda install -c conda-forge jupyterlab
```

## Run the W1 V1 notebook

Set the image and output directories, start Jupyter from the repository root, and run every cell in [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb):

```bash
export SOLAR_IMAGES_DIR=/path/to/test_images
export SOLAR_OUTPUT_DIR="$PWD/run_w1"
jupyter lab notebooks/pipeline.ipynb
```

The notebook verifies and merges the shards, then executes `dense`, `support`, `score`, and `refiner` with the W1 policy. It writes the final `rf_r34_tta.csv`, an intermediate TTA-based `cache/rf.csv`, logs, and completion receipts under `SOLAR_OUTPUT_DIR`. The intermediate RF CSV is not the historical RF submission. For dependencies installed in separate Conda environments, set `SOLAR_DENSE_PYTHON`, `SOLAR_SUPPORT_PYTHON`, `SOLAR_SCORE_PYTHON`, and `SOLAR_REFINER_PYTHON` to their Python executables. The notebook does not submit to Kaggle.

## Reproduce W1 V1 with the CLI

Run from the repository root with the same Conda environment. Use the same new cache directory for all four stages:

```bash
export SOLAR_IMAGES_DIR=/path/to/test_images
mkdir -p run_w1
python scripts/w1_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w1/cache --output run_w1/rf_r34_tta.csv --stage dense --final rf_r34_tta --dense-long2x-tta --device cuda
python scripts/w1_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w1/cache --output run_w1/rf_r34_tta.csv --stage support --final rf_r34_tta --dense-long2x-tta --device cuda
python scripts/w1_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w1/cache --output run_w1/rf_r34_tta.csv --stage score --final rf_r34_tta --dense-long2x-tta --device cuda
python scripts/w1_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w1/cache --output run_w1/rf_r34_tta.csv --stage refiner --final rf_r34_tta --dense-long2x-tta --device cuda
```

For a single Conda environment, the complete command is:

```bash
python scripts/w1_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w1_all/cache --output run_w1_all/rf_r34_tta.csv --stage all --final rf_r34_tta --dense-long2x-tta --device cuda
```

W1 V1 adds four views (identity, horizontal flip, vertical flip, and both flips) **only to the long2x dense arm**. Predictions are unflipped, accumulated sequentially in NumPy float32, averaged over the four views, and quantized per fold; the five quantized folds are then averaged. The s05 and DINOv3 GONG dense arms, RF-DETR support, RF scorer/selection, and R34 refiner remain unchanged. R34 retains its existing five-fold, four-flip averaging and 0.40 projection threshold. The W1 numerical TTA function is preserved from the cold-verified export. [configs/rf_r34_tta.json](configs/rf_r34_tta.json) documents the frozen recipe; the wrapper applies it through the existing [configs/rf.json](configs/rf.json) implementation.

Use a fresh cache when switching policies; do not reuse a historical non-TTA cache for W1. Tau-addition and W1-plus-tau combinations were rejected and are not final or recommended configurations. The W1 entry point only exposes W1 V1.

## Submitted identities and numerical tolerance

| Configuration | Entry point / final policy | Kaggle submission | Submitted CSV SHA-256 |
| --- | --- | --- | --- |
| **W1 V1 primary final** | `scripts/w1_infer.py --final rf_r34_tta --dense-long2x-tta` | **58360434** (`test180.csv`, 1,180 masks) | `08c79e5c92d5ac3b90401485cc65253ed638fdf0b8f3e2a009fea260613e2ee6` |
| Historical RF+R34 alternative | `scripts/rf_infer.py --final rf_refiner_r34` | 58338486 (`candidate.csv`) | `c68d4195fa6f999c26ef17e158f11da4ac8dac6a230a8a13bc6b59de58e24d8f` |
| Historical RF alternative | `scripts/rf_infer.py --final rf` | 58304186 (`primary_test180.csv`) | `68f5bb21c7e7b3019d7a75c12a0a2f77da77bcc9a8d3a006ab11dd033eb40e48` |

The NVIDIA GB10 cold-reproduction receipt for W1 V1 records **PASS_EXACT: 1,180/1,180 masks identical by ID/RLE**, identical submitted CSV SHA-256, and **zero denied reads in each of the four audited stages**. The run used a new export directory and cache, reconstructed verified shard weights, and only official test JPEGs as competition inputs. Reference comparison happened after inference. This is evidence from the cold-verified W1 export, not a claim that the release packaging check reran full GPU inference.

The numerical tolerance remains **at least 99.5% masks exactly identical by ID/RLE and IoU at least 0.95 for every differing mask**. CSV row order and newline encoding are ignored when comparing masks. Historical RF reproduced 1,184/1,185 masks exactly (sole differing IoU 0.9847); RF+R34 reproduced 1,181/1,185 masks exactly (four differing IoUs 0.9835–0.9999). Small boundary differences can occur because GPU floating-point kernels are not guaranteed to be bitwise identical across runs and devices. Submitted CSV files are not included; their full hashes above identify the references.

To reproduce either historical alternative, use a separate new cache and the following four stages. For RF alone, stop after `score` and use `run_alternative/rf.csv`:

```bash
mkdir -p run_alternative
python scripts/rf_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_alternative/cache --output run_alternative/rf.csv --stage dense --final rf --device cuda
python scripts/rf_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_alternative/cache --output run_alternative/rf.csv --stage support --final rf --device cuda
python scripts/rf_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_alternative/cache --output run_alternative/rf.csv --stage score --final rf --device cuda
python scripts/rf_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_alternative/cache --output run_alternative/rf_refiner_r34.csv --stage refiner --final rf_refiner_r34 --device cuda
```

## Measured runtime and hardware

W1 V1 cold-stage receipts on NVIDIA GB10 / aarch64 record the following durations for all 180 test JPEGs. The run used CUDA, a shared GPU, one native CPU thread, and CPU nice 19. Stage timers exclude interpreter startup and shard reconstruction.

| W1 stage | Measured seconds | Approximate minutes |
| --- | ---: | ---: |
| Dense (long2x four-flip TTA) | 2,110.25 | 35.2 |
| RF-DETR support | 2,976.59 | 49.6 |
| RF scoring and selection | 671.30 | 11.2 |
| ResNet34 refiner | 194.83 | 3.2 |
| Sum of stage timers | 5,952.96 | 99.2 |

For comparison, the earlier non-TTA RF+R34 notebook run took about 94 minutes. Runtime depends on hardware, dependencies, and other workloads. The cold reference started with at least 45 GiB available RAM; this is a recorded condition, not a measured minimum requirement. Dense inference uses fp32; RF-DETR uses the configured bfloat16 path. Do not interpret timings from different shared-GPU runs as controlled speed comparisons.

## Training and notices

Inference uses the included trained weights; no retraining is needed. The repository does not contain all historical training data or every original training runner. See [TRAINING.md](TRAINING.md), [MODEL_CARD.md](MODEL_CARD.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for scope, lineage, and applicable terms. Original project code is licensed under Apache-2.0. Trained weights remain subject to upstream terms and are for non-commercial research use because the training data is licensed CC BY-NC 4.0.
