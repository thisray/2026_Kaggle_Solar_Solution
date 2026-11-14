"""Train-only, source-fold-grouped loading of registered GONG observations."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from solar import data


def load_neighbors(manifest, fold_of, train_stems, validation_fold):
    manifest = Path(manifest)
    summary = json.loads((manifest.parent / "summary.json").read_text())
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    expected = summary.get("manifest_hashes", {}).get(manifest.name, summary["manifest_sha256"])
    if expected != digest(manifest):
        raise ValueError("GONG manifest hash mismatch")
    if summary["official_annotations_sha256"] != digest(data.raw_root() / data.ANN_NAME):
        raise ValueError("GONG labels differ from current official train GT")
    allowed = set(train_stems)
    rows = {}
    # Validate all manifest rows before filtering validation-fold samples.
    test_times = [datetime.strptime(x[:14], "%Y%m%d%H%M%S")
                  for x in data.list_test_stems()]
    if not test_times:
        raise ValueError("Cannot verify GONG test exclusion: test filenames unavailable")
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        observation = datetime.strptime(row["neighbor_stem"][:14], "%Y%m%d%H%M%S")
        if any(abs((observation - t).total_seconds()) <= 3600 for t in test_times):
            raise ValueError("GONG observation within ±1h of test: " + row["neighbor_stem"])
        source = row["source_stem"]
        if source not in fold_of or row["fold"] != fold_of[source]:
            raise ValueError("GONG source fold mismatch")
        if row["usage"] != "training_only" or row["label_transform"] != [[1, 0, 0], [0, 1, 0]]:
            raise ValueError("Unsupported GONG label coordinate system")
        if source not in allowed or fold_of[source] == validation_fold:
            continue
        key = row["sample_id"]
        if key in rows:
            raise ValueError("Duplicate GONG sample")
        image_path = manifest.parent / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        target_paths = {channel: manifest.parent / path for channel, path in row["target_paths"].items()}
        if row.get("ignore_band_px", 0):
            target_paths["ignore"] = manifest.parent / row["ignore_path"]
        if any(not path.is_file() for path in target_paths.values()):
            raise FileNotFoundError(target_paths)
        rows[key] = dict(row, resolved_image_path=image_path, resolved_target_paths=target_paths)
    return rows, dict(summary, manifest_sha256=expected)


def read_neighbor(row, target_root):
    image = data.load_gray(row["resolved_image_path"])
    fg = data.load_gray(row["resolved_target_paths"]["fg"])
    spine = data.load_gray(row["resolved_target_paths"]["spine"])
    if image.shape != fg.shape or image.shape != spine.shape:
        raise ValueError("GONG image and transferred targets have different shapes")
    positive = np.flatnonzero(fg)
    if row.get("ignore_band_px", 0):
        mask = data.load_gray(row["resolved_target_paths"]["ignore"]) > 0
        fg, spine = fg.astype(np.float32), spine.astype(np.float32)
        fg[mask] = spine[mask] = -255
        positive = np.flatnonzero(fg > 0)
    return image, fg, spine, positive
