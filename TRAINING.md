# Training and Inference Reproduction

## Inference reproduction

The intended reproduction path uses the included trained weights; retraining is not required. Create the Conda environment described in `environment.yml`, reconstruct the weights with `python merge_weights.py --root . --destination models`, and follow the four-stage CLI or notebook instructions in `README.md`.

## Training lineage

`configs/training_lineage.json` records the available model and configuration lineage. The deployed system includes three dense producer families, RF-DETR Nano support models, scorer components, temporal-fit artifacts, and the optional ResNet34 refiner. `configs/` contains the release configurations and the baseline plan.

The historical training process cannot be fully rerun from this repository alone. The complete competition data, clean-neighbor manifests and targets for the GONG fine-tuning stage, all historical batch providers, and the full scorer training runner are not included. `scripts/train_refiner.py` and `scripts/train_rf_components.py` provide selected training components but require the original prepared inputs and caller-managed resource admission. They do not constitute an end-to-end trainer.

## Data and restrictions

Competition data and derived trained weights are subject to CC BY-NC 4.0 and the applicable upstream model terms. The trained weights are provided for non-commercial research use only. No competition images, ground-truth masks, external filament annotations, or private test labels are included in this repository.
