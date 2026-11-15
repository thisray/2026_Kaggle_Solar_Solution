# Solar Filament Segmentation 2026

This repository contains the **W6 final** inference pipeline and trained weight shards for the Solar Filament Segmentation Challenge 2026. The primary final is Kaggle submission **58377502**, public score **0.40**, with **1,094 masks**. It includes no competition images, annotations, ground-truth masks, precomputed predictions, or CSV files.

## Environment and weights

Use Python 3.11 in Conda. From the repository root:

```bash
conda env create -f environment.yml
conda activate solar_solution_rf
python merge_weights.py --root . --destination models
```

`merge_weights.py` verifies every shard byte count and SHA-256 in `weights_manifest.json`, then verifies every reconstructed model. All shards are included and are at most 95 MiB. W6 adds five G_ctrl dense folds and its frozen temporal scorer/drop fits; the existing RF, dense, and R34 weights are unchanged. Generated `models/` and inference outputs are not committed.

## Reproduce W6

Use a new cache directory. Set the official test JPEG directory, then run the complete eight-stage CLI:

```bash
export SOLAR_IMAGES_DIR=/path/to/test_images
python scripts/demo_w6.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w6/cache --output run_w6/w6.csv --final w6 --dense-long2x-tta --postcal-w2 --device cuda
```

By default each stage uses the current Conda interpreter. If dependencies are in separate Conda environments, set `SOLAR_DENSE_PYTHON`, `SOLAR_SUPPORT_PYTHON`, `SOLAR_SCORE_PYTHON`, and `SOLAR_REFINER_PYTHON` to their Conda Python executables. The launcher removes inherited notebook/Python environment settings and clears an empty inherited `CUDA_VISIBLE_DEVICES` for GPU stages, matching the GB10 cold-replay repair; explicit nonempty device selections are preserved. Logs and `*_DEMO_DONE.json` receipts are written under the cache.

For staged execution, use the same images/models/cache/output and flags for each stage, in this exact order:

```bash
for stage in dense support score gctrl_plain_dense gctrl_dense gctrl_score refiner postcal_w2; do
  python scripts/w6_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w6_staged/cache --output run_w6_staged/w6.csv --stage "$stage" --final w6 --dense-long2x-tta --postcal-w2 --device cuda
done
```

With separate environments, invoke each stage using the corresponding interpreter instead: `dense`, `gctrl_plain_dense`, `gctrl_dense` use DENSE; `support` uses SUPPORT; `refiner` uses REFINER; `score`, `gctrl_score`, `postcal_w2` use SCORE. The launcher above handles environment cleanup. Do not change a recipe within a cache.

## Notebook

Install Jupyter inside Conda if needed, then run every cell in [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb):

```bash
conda install -c conda-forge jupyterlab
export SOLAR_IMAGES_DIR=/path/to/test_images
export SOLAR_OUTPUT_DIR="$PWD/run_w6_notebook"
jupyter lab notebooks/pipeline.ipynb
```

The notebook verifies and reconstructs all weights, executes the same eight-stage launcher, and writes `w6.csv` under `SOLAR_OUTPUT_DIR`. The optional interpreter variables above also apply. The notebook does not submit to Kaggle. The intermediate RF/R34 CSVs are upstream products, not the W6 final.

## Frozen W6 policy

W6 applies W2 COMBO_A_D postcalibration after W1 V1 (long2x dense four-flip TTA → RF → R34). Only the long2x arm uses the four dense views: identity, horizontal flip, vertical flip, and both flips. Predictions are unflipped, accumulated sequentially in NumPy float32, averaged, quantized per fold, and then averaged across five folds. The s05 and GONG arms keep their original inference.

RF-DETR support, fixed RF scoring/selection, and five-fold four-flip R34 inference retain the existing recipe. W2 then reprojects R34 probabilities at **0.45 instead of 0.40**, using original seed ownership. It removes a mask only when its G_ctrl overlap support count at **IoU ≥ 0.65** is **below 1** and its aligned RF score is **below 0.40**. G_ctrl uses its five raw dense checkpoints and frozen temporal selector/drop fits; its context remains non-TTA, including separately recomputed plain long2x maps. See [configs/w6_postcal.json](configs/w6_postcal.json), [configs/gctrl.json](configs/gctrl.json), and [configs/rf.json](configs/rf.json). These are fixed parameters, with no test-label access or online adaptation.

CSV schema is `filament_id,segmentation_rle`; IDs use the image stem and sequential instance number, with native 2048×2048 COCO compressed RLE. Empty images produce no rows.

## Submitted identities and numerical tolerance

| Configuration | Entry point / options | Kaggle submission | Submitted CSV SHA-256 |
| --- | --- | --- | --- |
| **W6 primary final** | `scripts/demo_w6.py --final w6 --dense-long2x-tta --postcal-w2` | **58377502**, 1,094 masks | `0e2dbd761885c96c996a738f47679173047df0d90a2deee35bb39b8455433809` |
| Historical W1 V1 | `scripts/w1_infer.py --final rf_r34_tta --dense-long2x-tta --stage all` | 58360434 | `08c79e5c92d5ac3b90401485cc65253ed638fdf0b8f3e2a009fea260613e2ee6` |
| Historical RF+R34 | `scripts/rf_infer.py --final rf_refiner_r34 --stage all` | 58338486 | `c68d4195fa6f999c26ef17e158f11da4ac8dac6a230a8a13bc6b59de58e24d8f` |
| Historical RF | `scripts/rf_infer.py --final rf --stage all` | 58304186 | `68f5bb21c7e7b3019d7a75c12a0a2f77da77bcc9a8d3a006ab11dd033eb40e48` |

Historical commands also require `--images`, `--models`, `--cache`, and `--output` as above, with a separate new cache for each policy.

The GB10 W6 cold replay passed **PASS_NUMERIC_TOLERANCE**: the same IDs, **1,093/1,094 exact RLEs (99.9086%)**, and one differing mask with **IoU 0.9993836671802774**, differing by two pixels. Reproduced CSV SHA-256: **`3414ec7997ac9097625a7d25d7883410affffdea21f4a23ba0436c3078fbf049`**. All eight audited stages recorded **zero denied reads**. The run began with a fresh export/cache, reconstructed verified shards, and used only official test JPEGs as competition inputs; comparison occurred after inference. This repository update verifies packaging against that cold export, without claiming another full GPU replay.

Acceptance requires **identical ID sets, at least 99.0% exact RLEs, and IoU ≥ 0.95 for every differing mask**. Mask comparison ignores CSV row order and newline serialization. GPU kernels can produce small floating-point boundary differences across runs/devices. The submitted and reproduced CSVs are identified by hash and are not bundled.

W6 full707 development evaluation (707 images / 1,154 records / 8,199 GT) is EV1 legacy **0.4616699503585168**, strict **0.46160876789772454**. This is conditional development evaluation, not fully nested validation: dense/refiner folds_v1 have temporal-group exposure on 370 images, with 337 nonexposed images. It is distinct from cold-reproduction evidence and official leaderboard performance.

## Measured runtime

The eight cold-stage wall timers on shared NVIDIA GB10 / aarch64, one native CPU thread and CPU nice 19, cover 180 test JPEGs:

| Stage | Seconds | Minutes |
| --- | ---: | ---: |
| Dense, long2x four-flip TTA | 5,156.61 | 85.94 |
| RF-DETR support | 6,028.97 | 100.48 |
| RF scoring/selection | 912.22 | 15.20 |
| G_ctrl plain dense context | 1,003.12 | 16.72 |
| G_ctrl dense | 1,490.41 | 24.84 |
| G_ctrl scoring | 826.69 | 13.78 |
| R34 refiner | 416.59 | 6.94 |
| W2 postcalibration | 19.91 | 0.33 |
| Sum | 15,854.52 | 264.24 |

The complete cold job took 15,874.65 seconds including orchestration. Stage timers include stage interpreter startup; shard reconstruction is outside the stage sum. These shared-GPU timings are not controlled speed comparisons with prior runs. The cold job started with at least 45 GiB available RAM, a recorded condition rather than a measured minimum. Dense uses fp32; RF-DETR uses the configured bfloat16 path. Hardware/dependency changes may affect runtime and numerical output.

## Training and terms

Inference uses the included trained weights; no retraining is needed. See [TRAINING.md](TRAINING.md), [MODEL_CARD.md](MODEL_CARD.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Original code is Apache-2.0. Trained weights retain upstream terms, including the bundled [DINOv3 agreement](licenses/DINOv3-LICENSE.md), and the non-commercial research restriction from CC BY-NC 4.0 training data. G_ctrl inherits the same DINOv3 source and terms as the existing DINOv3 producer.
