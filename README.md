# Solar Filament Segmentation 2026

The main final in this repository is **W14, Kaggle submission 58424527**, with **1,094 masks**. W14 uses all W6 weights unchanged and adds a per-mask binarization based on an expected-IoU ratio. SHA-256 of the submitted CSV:

`70030aa51bc56f082500e89c06c2d58ca64f3de3b8f489e098c21dfa1b7d0c01`

Technical report (4 pages plus references): report/technical_report.pdf

The repository contains the inference source code and the trained weights, split into shards. Competition images, annotations, ground truth, prediction CSVs and intermediate caches are not included.

## Environment and weights

Use Conda with Python 3.11. From the repository root, run:

```bash
conda env create -f environment.yml
conda activate solar_solution_rf
python merge_weights.py --root . --destination models
```

`merge_weights.py` checks the length and SHA-256 of all 100 shards against `weights_manifest.json`, then checks the 70 rebuilt models. Each shard is at most 95 MiB. W14 adds or changes no weights. The generated `models/` directory and inference outputs are not committed.

## Reproducing W14

Set the official test JPEG directory and run the eight-stage pipeline with a fresh cache:

```bash
export SOLAR_IMAGES_DIR=/path/to/test_images
python scripts/demo_w6.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w14/cache --output run_w14/w14.csv --final w6 --dense-long2x-tta --postcal-w2 --eiou-binarize --device cuda
```

`--final w14` is the same as `--final w6 --eiou-binarize`. The CLI keeps W6 as its default for compatibility with the earlier pipeline, so reproducing the main final requires the flags above or the W14 alias.

To run the stages one at a time, use the same images, models, cache and output paths and the same flags in every call:

```bash
for stage in dense support score gctrl_plain_dense gctrl_dense gctrl_score refiner postcal_w2; do
  python scripts/w6_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w14_staged/cache --output run_w14_staged/w14.csv --stage "$stage" --final w6 --dense-long2x-tta --postcal-w2 --eiou-binarize --device cuda
done
```

By default every stage uses the current Conda interpreter. If you install the stages in separate environments, set `SOLAR_DENSE_PYTHON`, `SOLAR_SUPPORT_PYTHON`, `SOLAR_SCORE_PYTHON` and `SOLAR_REFINER_PYTHON`: dense, gctrl_plain_dense and gctrl_dense use DENSE; support uses SUPPORT; refiner uses REFINER; score, gctrl_score and postcal_w2 use SCORE. The launcher clears inherited notebook and Python settings. GPU stages remove an empty `CUDA_VISIBLE_DEVICES` and keep a non-empty one as an explicit device choice. Logs and `*_DEMO_DONE.json` files are written to the cache. Do not switch recipes within the same cache.

## Notebook

[notebooks/pipeline.ipynb](notebooks/pipeline.ipynb) defaults to W14. After rebuilding the weights, it runs the same eight-stage launcher:

```bash
conda install -c conda-forge jupyterlab
export SOLAR_IMAGES_DIR=/path/to/test_images
export SOLAR_FINAL=w14
export SOLAR_OUTPUT_DIR="$PWD/run_w14_notebook"
jupyter lab notebooks/pipeline.ipynb
```

Run all cells in order; the output is `w14.csv`. To reproduce the previous version, set `SOLAR_FINAL=w6` and use a new output directory. The notebook does not submit to Kaggle.

## Fixed inference policy and evaluation

The W6 base is: W1 V1 long2x dense with four-flip TTA → fixed RF scoring and selection → five-fold four-flip R34 → W2 COMBO_A_D. Only the long2x dense arm uses the identity, horizontal, vertical and both-axis flips. Its outputs are flipped back to the original coordinates, accumulated in order in NumPy float32, averaged and quantized per fold, and then averaged over the five folds. The s05 and GONG arms are unchanged. W2 raises the R34 threshold from .40 to .45, and deletes a mask only when its G_ctrl support count at IoU≥.65 is below 1 and its aligned RF score is below .40. G_ctrl keeps its non-TTA context, including a separate plain long2x run. Configs: [configs/w6_postcal.json](configs/w6_postcal.json), [configs/gctrl.json](configs/gctrl.json), [configs/rf.json](configs/rf.json).

W14 keeps W6 and all of its weights and changes only the per-instance binarization of the native crop. Pixels are sorted by probability in decreasing order; with `S_k` the sum of the top k probabilities and `P` the sum of all probabilities, W14 keeps the k that maximizes `S_k/(k+P-S_k)`. This is the ratio of expected intersection to expected union, an approximation rather than the exact expected IoU. Pixels with equal probability are kept or dropped together, and ties in the objective go to the smallest k. W6's seed-connected components, ownership, exclusivity order and G_ctrl are kept. The method falls back to the seed when scale < .5, when a probability is not finite, or when the kept area is < 96 px. There is no calibration, clipping or new parameter.

full707 EV1 (707 images / 1,154 records / 8,199 GT): legacy **.46205893152894567**, strict **.46199775577515656**; versus W6, **+.00038898117042884817 / +.00038898787743202323** respectively. Paired bootstrap over physical images (5,000 resamples): P(Δ>0)=.9632, 95% CI about [-.000034, .000835]; on the two pre-specified halves, +.000458 / +.000290. The upstream temporal exposure is unchanged, so this is not fully nested. The gain is below the +.0005 threshold we had originally set for deploying it on its own; we chose W14 after weighing the overall strength of the evidence. No official score is claimed here.

Starting from upstream probabilities that had already been reproduced from a cold start, we reran projection + postcal: this partial cold reproduction gives **1,094/1,094 identical RLEs**, submitted CSV SHA-256 `70030aa51bc56f082500e89c06c2d58ca64f3de3b8f489e098c21dfa1b7d0c01`. This is not a full cold recomputation from the JPEGs. The flag is off by default; with it off, the code exactly reproduces the cold W6 run (SHA `3414ec7997ac9097625a7d25d7883410affffdea21f4a23ba0436c3078fbf049`). The known two-pixel difference from the submitted W6 and the numeric tolerance are described in the next section.

The W14 config is [configs/w14_candidate.json](configs/w14_candidate.json) (the original file name is kept to preserve lineage). The CSV columns are `filament_id,segmentation_rle`. Each ID is the image stem plus a running instance number, and masks are native 2048×2048 COCO compressed RLE. Empty images produce no rows.

## Submitted versions and reproduction limits

| Version | Kaggle submission | CSV SHA-256 |
| --- | --- | --- |
| **W14, main final** | **58424527** | `70030aa51bc56f082500e89c06c2d58ca64f3de3b8f489e098c21dfa1b7d0c01` |
| W6, previous final | 58377502 (public .40) | `0e2dbd761885c96c996a738f47679173047df0d90a2deee35bb39b8455433809` |
| W1 V1, earlier version | 58360434 | `08c79e5c92d5ac3b90401485cc65253ed638fdf0b8f3e2a009fea260613e2ee6` |

W14's **PASS_EXACT: 1,094/1,094 identical RLEs** is a partial cold reproduction: projection + postcal rerun from refiner probabilities that had already been reproduced from a cold start, not a full recomputation from the JPEGs. With the flag off, the code exactly reproduces the cold W6 run (1,094/1,094; SHA `3414ec7997ac9097625a7d25d7883410affffdea21f4a23ba0436c3078fbf049`). The cold W6 run compared with the W6 file submitted at the time is **PASS_NUMERIC_TOLERANCE**: 1,093/1,094 exact; the only difference is two pixels, IoU .9993836671802774. Denied reads were 0 in all eight original stages.

The full cold W6 run used a fresh export and cache, verified shards and the official test JPEGs; it was compared with the submitted file only after inference had finished. The numeric tolerance is the same set of IDs, at least 99.0% exact RLEs, and IoU≥.95 for every mask that differs. The comparison ignores row order and newline serialization. Different GPU kernels or dependency versions can cause floating-point differences at mask boundaries. W6 full707 EV1: legacy .4616699503585168, strict .46160876789772454. Both W14 and W6 inherit the temporal-group exposure of the dense/refiner folds_v1 (370 exposed / 337 nonexposed), so this is not fully nested validation. Development PQ, reproduction results and official leaderboard scores each measure different things.

## Runtime and licenses

On a shared GB10 (aarch64), the eight W6 stages took 15,854.52 s in total and the full orchestration 15,874.65 s (about 264.6 min), not counting the weight rebuild: dense 5,156.61 s, support 6,028.97 s, score 912.22 s, plain dense 1,003.12 s, G_ctrl dense 1,490.41 s, G_ctrl score 826.69 s, refiner 416.59 s, postcal 19.91 s. These are timings from the earlier cold W6 run, not a full cold W14 run or a controlled benchmark. At the time, available RAM was ≥45 GiB and native CPU threads=1 at nice 19; dense ran in fp32 and RF-DETR used bfloat16 as configured. Hardware and dependencies affect both runtime and numerical results.

No retraining is needed. Training and rights are described in [TRAINING.md](TRAINING.md), [MODEL_CARD.md](MODEL_CARD.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Our own code is released under Apache-2.0. The weights remain subject to upstream terms, including the [DINOv3 agreement](licenses/DINOv3-LICENSE.md) and the non-commercial research restriction of the CC BY-NC 4.0 training data. G_ctrl uses the same DINOv3 source and terms.
