# Solar Filament Segmentation 2026

This repository contains the inference pipeline and weight shards for two submitted solutions to the Solar Filament Segmentation Challenge 2026. It includes no competition images, annotations, ground-truth masks, or private training data.

## Environment and weight reconstruction

Use Python 3.11 in a Conda environment. From the repository root:

```bash
conda env create -f environment.yml
conda activate solar_solution_rf
python merge_weights.py --root . --destination models
```

`merge_weights.py` verifies every shard against `weights_manifest.json`, reconstructs each model under `models/`, and verifies the reconstructed byte count and SHA-256. The merged `models/` directory is generated locally and is not committed.

Install Jupyter in the environment if it is not already available:

```bash
conda install -c conda-forge jupyterlab
```

## Run the notebook

Set the image and output directories, start Jupyter from the repository root, then run every cell in `notebooks/pipeline.ipynb`:

```bash
export SOLAR_IMAGES_DIR=/path/to/test_images
export SOLAR_OUTPUT_DIR="$PWD/run_release"
jupyter lab notebooks/pipeline.ipynb
```

The notebook verifies and merges the shards, then executes the four inference stages in order: `dense`, `support`, `score`, and `refiner`. It writes `rf.csv`, `rf_refiner_r34.csv`, per-stage logs, and completion receipts under `SOLAR_OUTPUT_DIR`. The notebook does not submit to Kaggle.

## Run the four CLI stages

Run each command from the repository root with the same Conda environment. Set `SOLAR_IMAGES_DIR` to the directory containing the test JPEGs. The cache directory is shared across stages and should be new for a cold run.

```bash
export SOLAR_IMAGES_DIR=/path/to/test_images
mkdir -p run_release
python scripts/rf_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_release/cache --output run_release/rf.csv --stage dense --final rf --device cuda
python scripts/rf_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_release/cache --output run_release/rf.csv --stage support --final rf --device cuda
python scripts/rf_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_release/cache --output run_release/rf.csv --stage score --final rf --device cuda
python scripts/rf_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_release/cache --output run_release/rf_refiner_r34.csv --stage refiner --final rf_refiner_r34 --device cuda
```

The first three stages build dense maps, RF-DETR support descriptors, and the RF-selected CSV. The fourth applies the five-fold ResNet34 refiner to the RF result. The `--final` values select the output policy; they do not change the Kaggle submission history.

## Submitted candidates and numerical comparison

| CLI policy | Kaggle submission | Submitted CSV SHA-256 |
| --- | --- | --- |
| `--final rf` | 58304186 (`primary_test180.csv`) | `68f5bb21c7e7b3019d7a75c12a0a2f77da77bcc9a8d3a006ab11dd033eb40e48` |
| `--final rf_refiner_r34` | 58338486 (`candidate.csv`) | `c68d4195fa6f999c26ef17e158f11da4ac8dac6a230a8a13bc6b59de58e24d8f` |

The cold-reproduction outputs are compared by image ID and decoded mask, not CSV byte order or newline encoding. On the GB10 reference run, RF reproduced 1,184 of 1,185 masks exactly; its sole differing mask had IoU 0.9847 against the submitted mask. RF plus R34 reproduced 1,181 of 1,185 masks exactly; the four differing masks had IoU from 0.9835 to 0.9999. Both runs passed the recorded numerical tolerance: at least 99.5% exact masks and at least 0.95 IoU for every differing mask. Small boundary differences can occur because GPU floating-point kernels are not guaranteed to be bitwise identical across runs and devices. The submitted CSV files are not included in this source export; their full SHA-256 values above identify the references.

## Runtime and hardware

A full four-stage notebook run took 5,632 seconds (about 94 minutes) on an NVIDIA GB10, aarch64, using CUDA, one native CPU thread, and a shared GPU. Stage durations below are derived from cumulative stage receipts and are approximate:

| Stage | Approximate elapsed time |
| --- | ---: |
| Dense | 19.6 minutes |
| RF-DETR support | 46.9 minutes |
| RF scoring and selection | 17.5 minutes |
| ResNet34 refiner | 9.8 minutes |
| Total | 93.9 minutes |

This is one measured run, not a performance guarantee. GPU model, available memory, driver, CUDA, PyTorch, and other workloads can affect runtime and numerical output. The reference run started with at least 45 GiB available RAM; this is a recorded condition, not a measured minimum requirement. Dense inference uses fp32; RF-DETR inference uses the configured bfloat16 path.

## Training and notices

Inference uses the included trained weights; no retraining is needed. The repository does not contain all historical training data or every original training runner. See [TRAINING.md](TRAINING.md), [MODEL_CARD.md](MODEL_CARD.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for scope, lineage, and applicable terms. Original project code is licensed under Apache-2.0. Trained weights remain subject to upstream terms and are for non-commercial research use because the training data is licensed CC BY-NC 4.0.
