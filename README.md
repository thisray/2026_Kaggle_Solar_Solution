# Solar Filament Segmentation 2026

本 repo 的主要 final 為 **W14，Kaggle submission 58424527**，共 **1,094 masks**。W14 沿用 W6 全部權重，加入逐 mask 的期望 IoU 比值二值化。提交 CSV SHA-256：

`70030aa51bc56f082500e89c06c2d58ca64f3de3b8f489e098c21dfa1b7d0c01`

本 repo 包含推論原始碼與訓練權重分片；競賽影像、annotations、GT、預測 CSV 與中間 cache 不隨 repo 發布。

## 環境與權重

使用 Conda Python 3.11，於 repo 根目錄執行：

```bash
conda env create -f environment.yml
conda activate solar_solution_rf
python merge_weights.py --root . --destination models
```

`merge_weights.py` 依 `weights_manifest.json` 核對全部 100 個分片的長度及 SHA-256，再核對重建的 70 個模型。各分片不超過 95 MiB。W14 不新增或更動權重；產生的 `models/` 與推論輸出不提交。

## 重現 W14

設定官方 test JPEG 目錄，以全新 cache 執行八階段流程：

```bash
export SOLAR_IMAGES_DIR=/path/to/test_images
python scripts/demo_w6.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w14/cache --output run_w14/w14.csv --final w6 --dense-long2x-tta --postcal-w2 --eiou-binarize --device cuda
```

`--final w14` 等同 `--final w6 --eiou-binarize`。CLI 保留 W6 預設以相容既有流程；重現主要 final 時必須使用上述旗標或 W14 別名。

逐階段執行時，所有呼叫使用相同 images/models/cache/output 與旗標：

```bash
for stage in dense support score gctrl_plain_dense gctrl_dense gctrl_score refiner postcal_w2; do
  python scripts/w6_infer.py --images "$SOLAR_IMAGES_DIR" --models models --cache run_w14_staged/cache --output run_w14_staged/w14.csv --stage "$stage" --final w6 --dense-long2x-tta --postcal-w2 --eiou-binarize --device cuda
done
```

預設使用目前 Conda interpreter；分環境安裝時可設定 `SOLAR_DENSE_PYTHON`、`SOLAR_SUPPORT_PYTHON`、`SOLAR_SCORE_PYTHON`、`SOLAR_REFINER_PYTHON`。dense/gctrl_plain_dense/gctrl_dense 使用 DENSE；support 使用 SUPPORT；refiner 使用 REFINER；score/gctrl_score/postcal_w2 使用 SCORE。Launcher 清理繼承的 notebook/Python 設定，GPU stage 會清除空的 `CUDA_VISIBLE_DEVICES`，保留非空的明確裝置選擇；log 與 `*_DEMO_DONE.json` 寫在 cache。不可在同一 cache 切換 recipe。

## Notebook

[notebooks/pipeline.ipynb](notebooks/pipeline.ipynb) 預設 W14，重建權重後執行相同八階段 launcher：

```bash
conda install -c conda-forge jupyterlab
export SOLAR_IMAGES_DIR=/path/to/test_images
export SOLAR_FINAL=w14
export SOLAR_OUTPUT_DIR="$PWD/run_w14_notebook"
jupyter lab notebooks/pipeline.ipynb
```

依序執行全部 cell，輸出 `w14.csv`。設 `SOLAR_FINAL=w6` 並另用新輸出目錄可重現前一版；notebook 不提交 Kaggle。

## 固定推論政策與評估

W6 基底為 W1 V1 long2x dense 四翻轉 TTA → RF 固定評分／選取 → 五折四翻轉 R34 → W2 COMBO_A_D。只有 long2x dense arm 使用 identity／horizontal／vertical／both flips，反轉回原座標後以 NumPy float32 依序累加、平均、逐折量化，再平均五折；s05／GONG 維持原流程。W2 將 R34 門檻由 .40 改為 .45；僅在 G_ctrl IoU≥.65 的支持數<1 且對齊 RF score<.40 時刪除。G_ctrl 保持非 TTA context，包括獨立 plain long2x。設定見 [configs/w6_postcal.json](configs/w6_postcal.json)、[configs/gctrl.json](configs/gctrl.json)、[configs/rf.json](configs/rf.json)。

W14 沿用 W6 與全部權重，只改逐 instance 的 native-crop 二值化：依機率遞減排序，令 `S_k` 為前 k 個機率和、`P` 為全部機率和，最大化 `S_k/(k+P-S_k)`。這是期望交集／期望聯集的比值近似，不是 IoU 期望的精確公式；同機率完整納入，效用平手取最小 k。保留 W6 seed-connected components、ownership、互斥順序與 G_ctrl。scale < .5、非有限機率、保留面積 < 96 px 時回退 seed；沒有校準、夾限或新參數。

full707 EV1（707 images／1,154 records／8,199 GT）：legacy **.46205893152894567**、strict **.46199775577515656**；相對 W6 分別 **+.00038898117042884817／+.00038898787743202323**。physical-image paired bootstrap 5000，P(Δ>0)=.9632、95% CI 約 [-.000034,.000835]；預定兩半 +.000458／+.000290。上游 temporal exposure 不變，並非 fully nested。未達原先 +.0005 單獨部署門檻；commander 綜合證據強度選定 W14。此處不宣稱官方分數。

從已冷重現的上游機率接續 projection＋postcal，部分冷重現 **1,094/1,094 逐 RLE 相同**；提交 CSV SHA-256 `70030aa51bc56f082500e89c06c2d58ca64f3de3b8f489e098c21dfa1b7d0c01`。這不是完整 JPEG 冷重算。旗標預設關閉；關閉精確重現冷跑 W6（SHA `3414ec7997ac9097625a7d25d7883410affffdea21f4a23ba0436c3078fbf049`），與已提交 W6 的既知兩像素差異及數值容差見下節。

W14 設定見 [configs/w14_candidate.json](configs/w14_candidate.json)（保留原檔名以保存 lineage）。CSV 欄位為 `filament_id,segmentation_rle`，ID 為影像 stem 加連續 instance 編號，使用 native 2048×2048 COCO compressed RLE；空影像不產生列。

## 提交版本與重現界線

| 版本 | Kaggle submission | CSV SHA-256 |
| --- | --- | --- |
| **W14 主要 final** | **58424527** | `70030aa51bc56f082500e89c06c2d58ca64f3de3b8f489e098c21dfa1b7d0c01` |
| W6 前一版 final | 58377502（public .40） | `0e2dbd761885c96c996a738f47679173047df0d90a2deee35bb39b8455433809` |
| W1 V1 歷史版 | 58360434 | `08c79e5c92d5ac3b90401485cc65253ed638fdf0b8f3e2a009fea260613e2ee6` |

W14 的 **PASS_EXACT：1,094/1,094 逐 RLE 相同**，是從已冷重現的 refiner 機率接續投影＋postcal 的部分冷重現，並非完整 JPEG 重算。關閉旗標精確重現冷跑 W6（1,094/1,094；SHA `3414ec7997ac9097625a7d25d7883410affffdea21f4a23ba0436c3078fbf049`）。冷跑 W6 對當時提交 W6 為 **PASS_NUMERIC_TOLERANCE**：1,093/1,094 exact，唯一差異為兩像素、IoU .9993836671802774；原八階段 denied reads 均為 0。

完整冷跑 W6 使用新 export/cache、已驗分片與官方 test JPEG，與提交檔的比較在推論完成後才進行。數值容差為相同 ID 集合、至少 99.0% exact RLE、每個差異 mask IoU≥.95；比較忽略列序與換行序列化。跨 GPU kernel／依賴版本可能有邊界浮點差異。W6 full707 EV1 legacy .4616699503585168、strict .46160876789772454。W14 與 W6 都沿用 dense/refiner folds_v1 的 temporal-group exposure（370 exposed／337 nonexposed），不是 fully nested validation；開發 PQ、重現結果與官方榜分數各自有不同意義。

## 耗時與授權

共享 GB10／aarch64 上 W6 八階段合計 15,854.52 秒，完整 orchestration 15,874.65 秒（約 264.6 分）；權重重建另計。dense 5,156.61s、support 6,028.97s、score 912.22s、plain dense 1,003.12s、G_ctrl dense 1,490.41s、G_ctrl score 826.69s、refiner 416.59s、postcal 19.91s。此為 W6 既有冷跑計時，非 W14 完整冷跑或受控效能比較。當時可用 RAM≥45 GiB、native CPU threads=1／nice19；dense fp32，RF-DETR 依設定使用 bfloat16。硬體及依賴會影響耗時與數值。

不需重新訓練。訓練與權利說明見 [TRAINING.md](TRAINING.md)、[MODEL_CARD.md](MODEL_CARD.md)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。原創程式採 Apache-2.0；權重仍受上游條款約束，包括 [DINOv3 agreement](licenses/DINOv3-LICENSE.md) 及 CC BY-NC 4.0 訓練資料的非商業研究限制。G_ctrl 沿用相同 DINOv3 來源與條款。
