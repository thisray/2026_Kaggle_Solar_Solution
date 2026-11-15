# Model Card

## Intended use

This system performs instance segmentation of solar filaments in 2048 × 2048 JPEG images for the Solar Filament Segmentation Challenge 2026. The primary final is W1 V1 (`rf_r34_tta`, Kaggle submission 58360434). It adds four-flip TTA only to the long2x dense arm; RF support, scorer/selection, and the R34 refiner are unchanged. The historical RF policy uses dense segmentation maps, RF-DETR Nano support descriptors, and a fixed scorer/selection policy. The `rf_refiner_r34` policy applies a five-fold ResNet34 refiner to the RF output.

The inference pipeline does not consume test ground truth, private annotations, external filament labels, or image timestamp metadata as model features. Filenames are used only to locate input images and identify output rows.

## Components and lineage

- Dense ensemble: five folds each of the R34 `long2x`, R34 `s05`, and DINOv3 ConvNeXt-small GONG fine-tuned producers. Quantized per-fold maps are averaged as configured by the release pipeline.
- Support: five-fold RF-DETR Nano predictions provide four support descriptors per candidate.
- Selection: five scorer folds use the fixed 83-feature input and the release configuration. Producer support fields are scored separately before scores are averaged.
- Optional refiner: five ResNet34 folds, four-flip averaging, float32 mean, and the configured 0.40 projection threshold.

The included checkpoints were stripped of optimizer state, RNG state, and training-path metadata before release. Each shard and reconstructed checkpoint is checked against `weights_manifest.json`.

## Evaluation and reproducibility

The full707 offline score recorded during development is EV1 0.4584661226677281 and strict 0.45840629411388906 for RF. These are historical development metrics, not the cold-inference comparison and not a guarantee of competition performance.

The W1 V1 GB10 cold run passed PASS_EXACT: 1,180/1,180 masks and the CSV SHA-256 matched exactly, with zero denied reads in all four audited stages. See README for the submitted identity and stage timings. The historical GB10 cold run reproduced 1,184/1,185 RF masks exactly and 1,181/1,185 RF+R34 masks exactly against the submitted CSVs. All differing masks met the documented minimum IoU of 0.95 (measured minimum 0.9835). GPU floating-point execution can create small boundary differences across runs or devices.

## Limitations

- The repository provides inference assets, not a turnkey reconstruction of the complete historical training process.
- Complete historical training datasets, clean-neighbor manifests, teacher targets, and all training runners are not included.
- Performance and numerical output may vary with hardware and software versions.
- Trained weights are restricted to non-commercial research use under the applicable upstream and training-data terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
