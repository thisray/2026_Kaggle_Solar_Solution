# Third-Party Notices and Weight Terms

## Original project code

Original code in this repository is licensed under Apache License 2.0, as provided in `LICENSE`. This grant does not replace third-party terms or grant rights to training data. The trained checkpoints are separately subject to the upstream model terms and the non-commercial restriction described below.

## RF-DETR Nano

The release uses RF-DETR Nano from Roboflow RF-DETR 1.5.2. The upstream project identifies the `rfdetr` package and Nano model weights as Apache-2.0. The included RF support checkpoints are fine-tuned project weights, not an unmodified Roboflow checkpoint. The RF-DETR code and pretrained weights were obtained through the package/model distribution; the original pretrained checkpoint is not separately bundled.

Upstream references:

- [RF-DETR repository and license](https://github.com/roboflow/rf-detr)
- [RF-DETR 1.5.2 release](https://github.com/roboflow/rf-detr/releases/tag/v1.5.2)

## DINOv3

The `exp_cnx_dino_gong_ft` checkpoints derive from the `timm/convnext_small.dinov3_lvd1689m` pretrained model at revision `9a0c319ae4531d36df6ef11fc253fdc4b503628a`. The full DINOv3 agreement is included verbatim at `licenses/DINOv3-LICENSE.md`. It grants rights to distribute derivative works under the agreement, requires a copy of the agreement to accompany distributed DINO materials and derivatives, and requires acknowledgment when publishing research performed using DINO materials. Retain that license file with any redistribution of these checkpoints. These trained checkpoints are additionally limited to non-commercial research by the training-data terms below.

Pretrained source: [revision-pinned timm model license](https://huggingface.co/timm/convnext_small.dinov3_lvd1689m/resolve/9a0c319ae4531d36df6ef11fc253fdc4b503628a/LICENSE.md).

## ResNet34 and ImageNet initialization

ResNet34 architectures and helper code are provided by `segmentation-models-pytorch` (MIT), `torchvision` (BSD-3-Clause), and `timm` (Apache-2.0), depending on the producer. The upstream code licenses do not by themselves settle rights in pretrained weights. The dense ResNet34 checkpoints and refiner were initialized from ImageNet-pretrained ResNet34 weights. The `timm` project notes that ImageNet pretrained weights are for non-commercial research and advises treating the dataset terms as applicable to such weights. Accordingly, the affected trained checkpoints are distributed only for non-commercial research use, subject to the relevant upstream terms.

Upstream references:

- [segmentation-models.pytorch license](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE)
- [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE)
- [torchvision ResNet34 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34)
- [timm code and pretrained-weight terms](https://github.com/huggingface/pytorch-image-models#licenses)

## Competition training data and trained weights

The competition training data is licensed CC BY-NC 4.0. Project-trained checkpoints were produced using that data and are provided with attribution to the competition data source, subject to the same non-commercial research limitation and any additional upstream terms. No competition images or annotations are redistributed here. See the [CC BY-NC 4.0 terms](https://creativecommons.org/licenses/by-nc/4.0/).

## Other dependencies

The environment installs third-party packages listed in `requirements.txt`. Their source-code licenses remain with their respective projects and distributions; this repository does not relicense those packages. The inspected package inventory is recorded in `licenses/dependency_inventory.json`. Relevant licenses include PyTorch (BSD-style), NumPy (BSD), SciPy (BSD), scikit-learn (BSD), LightGBM (MIT), OpenCV (Apache-2.0/third-party notices), pandas (BSD), Pillow (HPND), and pycocotools (BSD-style). Consult the exact package distribution for its complete notices.

## Scope

The statements above describe the sources and terms identified for this export. They are not legal advice and do not grant commercial rights in upstream assets or training data. The original code license does not override the restrictions on the trained weights.
