"""Physical-image grouped folds, stratified by station and rater multiplicity."""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

N_FOLDS = 5
SEED = 261008


def make_folds(records_by_stem: dict, n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, int]:
    strata = defaultdict(list)
    for stem, recs in sorted(records_by_stem.items()):
        station = stem[-2:]
        mult = min(len(recs), 3)
        strata[(station, mult)].append(stem)
    rng = random.Random(seed)
    fold_of = {}
    offset = 0
    for key in sorted(strata):
        stems = strata[key]
        rng.shuffle(stems)
        for i, stem in enumerate(stems):
            fold_of[stem] = (i + offset) % n_folds
        offset += len(stems)
    return fold_of


def write_folds(fold_of: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "fold"])
        for stem in sorted(fold_of):
            w.writerow([stem, fold_of[stem]])


def read_folds(path: Path) -> dict[str, int]:
    with open(path) as f:
        return {r["stem"]: int(r["fold"]) for r in csv.DictReader(f)}
