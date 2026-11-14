"""Compare a completed inference CSV with an optional reference; never used by inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment


def compare(actual, reference, stems):
    def masks(table, stem):
        selected = table[table.filament_id.str.rsplit('_', n=1).str[0] == stem]
        return [dict(size=[2048, 2048], counts=v.encode('ascii')) for v in selected.segmentation_rle]
    results = {}
    for stem in stems:
        a, b = masks(actual, stem), masks(reference, stem)
        ious = np.asarray(mu.iou(a, b, [0]*len(b))) if a and b else np.zeros((len(a), len(b)))
        x, y = linear_sum_assignment(-ious)
        matched = ious[x, y].tolist()
        results[stem] = dict(actual_masks=len(a), reference_masks=len(b),
            exact_order=[v['counts'] for v in a] == [v['counts'] for v in b],
            exact_mask_set=sorted(v['counts'] for v in a) == sorted(v['counts'] for v in b),
            matched_ious=matched, unmatched_actual=len(a)-len(x), unmatched_reference=len(b)-len(y),
            all_iou_ge_099=len(a)==len(b) and all(v >= .99 for v in matched))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--receipt', type=Path, help='Inference cache/done.json including empty images')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actual-pngs', type=Path)
    parser.add_argument('--reference-pngs', type=Path)
    args = parser.parse_args()
    if bool(args.actual) != bool(args.reference) or (args.actual and not args.receipt):
        parser.error('CSV comparison requires --actual, --reference and --receipt')
    if bool(args.actual_pngs) != bool(args.reference_pngs):
        parser.error('PNG comparison requires both PNG directories')
    if not args.actual and not args.actual_pngs:
        parser.error('Choose CSV and/or PNG comparison')
    result = {}
    if args.actual:
        stems = json.loads(args.receipt.read_text())['processed_images']
        result = compare(pd.read_csv(args.actual), pd.read_csv(args.reference), stems)
    passed = all(v['all_iou_ge_099'] for v in result.values())
    if args.actual_pngs:
        import cv2
        png_results = {}
        for path in sorted(args.actual_pngs.rglob('*.png')):
            rel = path.relative_to(args.actual_pngs)
            expected = args.reference_pngs / rel
            a, b = cv2.imread(str(path), 0), cv2.imread(str(expected), 0)
            if a is None or b is None or a.shape != b.shape:
                raise ValueError(f'Missing or incompatible PNG: {rel}')
            delta = np.abs(a.astype(float)-b.astype(float))
            png_results[str(rel)] = dict(exact=bool(np.array_equal(a,b)), max_u8=float(delta.max()), mean_u8=float(delta.mean()), changed_pixels=int((delta>0).sum()))
        if not png_results:
            parser.error('No actual PNGs found')
        result = dict(csv=result, png=png_results)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
