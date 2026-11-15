# Model Card

## Intended use

This system performs instance segmentation of solar filaments in 2048 × 2048 JPEG images for the Solar Filament Segmentation Challenge 2026. The primary final is W6 (`w6`, Kaggle submission 58377502): W1 V1 long2x four-flip dense TTA, RF support/scoring, R34 refinement, then frozen COMBO_A_D postcalibration. The R34 native projection threshold changes from 0.40 to 0.45. Masks with G_ctrl support IoU below 0.65 and aligned RF score below 0.40 are removed (required support count 1). W1 V1 (58360434), RF+R34 (58338486), and RF (58304186) remain historical alternatives.

The inference pipeline does not consume test ground truth, private annotations, external filament labels, or image timestamp metadata as model features. Filenames are used only to locate input images and identify output rows.

## Components and lineage

- Dense ensemble: five folds each of the R34 `long2x`, R34 `s05`, and DINOv3 ConvNeXt-small GONG fine-tuned producers. Quantized per-fold maps are averaged as configured by the release pipeline.
- Support: five-fold RF-DETR Nano predictions provide four support descriptors per candidate.
- Selection: five scorer folds use the fixed 83-feature input and the release configuration. Producer support fields are scored separately before scores are averaged.
- Refiner before W6 postcalibration: five ResNet34 folds, four-flip averaging, float32 mean, and the configured 0.40 projection threshold.

The included checkpoints were stripped of optimizer state, RNG state, and training-path metadata before release. Each shard and reconstructed checkpoint is checked against `weights_manifest.json`.

- G_ctrl: five raw DINOv3 ConvNeXt-small dense folds, original non-TTA context, and its frozen 79-feature temporal LightGBM selector/drop fits. The included dense tensors and fitted selectors are preserved from the cold-verified export. G_ctrl inherits the same DINOv3 and non-commercial training-data terms as the existing DINOv3 producer.

## Evaluation and reproducibility

The full707 offline score recorded during development is EV1 0.4584661226677281 and strict 0.45840629411388906 for RF. These are historical development metrics, not the cold-inference comparison and not a guarantee of competition performance.

W6 full707 EV1 legacy is 0.4616699503585168 and strict is 0.46160876789772454 (707 physical images, 1,154 annotation records, 8,199 GT). Dense/refiner folds_v1 and temporal RF folds do not form fully nested validation; 370 images have temporal-group exposure, with 337 nonexposed images. These development metrics are not an unbiased generalization estimate.

The W6 GB10 cold run passed PASS_NUMERIC_TOLERANCE: identical IDs, 1,093/1,094 exact RLEs, sole differing IoU 0.9993836671802774, and zero denied reads across eight stages. The reference and reproduced CSV hashes differ as documented in README. This packaging update preserves the cold-verified numerical functions; it does not claim another full GPU replay.

The historical W1 V1 GB10 cold run passed PASS_EXACT: 1,180/1,180 masks and the CSV SHA-256 matched exactly, with zero denied reads in all four audited stages. See README for the submitted identity and stage timings. The historical GB10 cold run reproduced 1,184/1,185 RF masks exactly and 1,181/1,185 RF+R34 masks exactly against the submitted CSVs. All differing masks met the documented minimum IoU of 0.95 (measured minimum 0.9835). GPU floating-point execution can create small boundary differences across runs or devices.

## Limitations

- The repository provides inference assets, not a turnkey reconstruction of the complete historical training process.
- Complete historical training datasets, clean-neighbor manifests, teacher targets, and all training runners are not included.
- Performance and numerical output may vary with hardware and software versions.
- Trained weights are restricted to non-commercial research use under the applicable upstream and training-data terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
