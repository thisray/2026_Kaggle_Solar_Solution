"""Frozen GONG5 inference from JPEGs and archived weights, without training artifacts."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def load(path):
    with Path(path).open('rb') as stream:
        value = pickle.load(stream)
    def limit(item):
        if isinstance(item, dict):
            for child in item.values():
                limit(child)
        elif hasattr(item, 'set_params') and hasattr(item, 'n_jobs'):
            item.set_params(n_jobs=THREADS)
    limit(value)
    return value


def valid_cache(marker, key):
    if not marker.exists():
        return False
    try:
        saved = json.loads(marker.read_text())
        return saved['key'] == key and all(
            Path(p).is_file() and sha(p) == digest for p, digest in saved['outputs'].items())
    except (KeyError, ValueError, OSError):
        return False


def dense(row, images, code_hash, device):
    import cv2
    import numpy as np
    import torch
    from solar.dense import normalized_image, save_probability, make_model
    from solar.foundation_dense import FOUNDATION_ARMS, FoundationDense
    checkpoint = MODEL / row['path']
    # Verify checkpoint before consulting any completed output.
    digest = sha(checkpoint)
    if digest != row['sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {checkpoint}')
    output = RUN / 'fold_maps' / row['arm'] / str(row['fold'])
    marker = output / 'done.json'
    key = dict(jpegs={s: sha(p) for s, p in images.items()}, checkpoint=digest,
               code=code_hash, config=sha(REPO / 'docs/final/inference_config.json'),
               device=device, threads=THREADS)
    if valid_cache(marker, key):
        return
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['plan']['arm'] == row['arm'] and ck['fold'] == row['fold']
    encoder = ck['plan']['encoder']
    model = (FoundationDense(encoder, pretrained=False) if encoder in FOUNDATION_ARMS.values()
             else make_model(pretrained=False, encoder=encoder))
    model.load_state_dict(ck['model'], strict=True)
    model = model.float().to(device).eval()
    del ck
    outputs = {}
    for stem, path in images.items():
        start = time.perf_counter()
        image = normalized_image(path)
        if row['arm'] == 'exp_s05':
            image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            pred = model(torch.from_numpy(np.ascontiguousarray(image[None, None])).to(device)).float().sigmoid()[0].cpu().numpy()
        if row['arm'] == 'exp_s05':
            pred = np.stack([cv2.resize(p, (2048, 2048), interpolation=cv2.INTER_LINEAR) for p in pred])
        save_probability(output, stem, pred)
        for channel in ('fg', 'spine'):
            path = output / f'{stem}_{channel}.png'
            outputs[str(path.resolve())] = sha(path)
        print('DENSE', row['arm'], row['fold'], stem, time.perf_counter()-start, flush=True)
    del model
    gc.collect()
    write(marker, dict(key=key, outputs=outputs))


def build_maps(stems):
    import cv2
    import numpy as np
    mapping = {'long2x':'exp_long2x', 's05':'exp_s05', 'gong':'exp_cnx_dino_gong_ft'}
    sources = {}
    for family, arm in mapping.items():
        directory = RUN / 'maps' / family
        for stem in stems:
            # Match discover(): average the five already quantized fold PNGs.
            probability = np.zeros((2, 2048, 2048), np.float32)
            for fold in range(5):
                for k, channel in enumerate(('fg', 'spine')):
                    path = RUN / 'fold_maps' / arm / str(fold) / f'{stem}_{channel}.png'
                    value = cv2.imread(str(path), 0)
                    assert value is not None, path
                    probability[k] += value.astype(np.float32)
            directory.mkdir(parents=True, exist_ok=True)
            for k, channel in enumerate(('fg', 'spine')):
                assert cv2.imwrite(str(directory / f'{stem}_{channel}.png'), np.rint(probability[k] / 5).astype(np.uint8))
        sources[family] = dict(test=str(directory), oof=None)
    return sources

def downstream(stems, sources):
    import numpy as np
    import pandas as pd
    import attack_base as base
    import attack_actions as actions
    import attack_drop as drop
    import l2_build_bank as builder
    import l2_add_scorer_features as enrich
    from solar import data, select
    manifest = CONFIG
    features = manifest['feature_columns']
    policy = CONFIG['policy']
    drop_policy = CONFIG['drop_policy']
    assert policy['column'] == 'baseline' and policy['method'] == 'all'
    assert not any(any(t in f.lower() for t in ('station', 'timestamp', 'annotator', 'rater')) for f in features)
    assert not {'stem','id','fold'} & set(features)
    parts, tables = {}, []
    for stem in stems:
        combined = dict(rles=[], thresholds=[], matrices=[], record_ids=[])
        for sid, family in enumerate(manifest['include']):
            directory = RUN / 'shards' / family
            directory.mkdir(parents=True, exist_ok=True)
            opts = manifest['family_inputs'][family]['options']
            builder.build_one((stem, sources[family]['test'], directory, 'test', False, 261009, False, opts))
            payload = load(directory / f'{stem}.pkl')
            if not payload['rows']:
                continue
            frame = pd.DataFrame(payload['rows'])
            entry = {k:payload[k] for k in ('rles','thresholds','matrices','record_ids')}
            enrich.BANK = {stem:entry}
            enrich.TABLES = {stem:frame}
            enrich.MANIFEST = dict(split='test', prob_dir=sources[family]['test'], inputs={stem:payload['input_hashes']})
            added, _ = enrich.enrich_one(stem)
            frame = frame.merge(pd.DataFrame(added), on=['stem','id'], validate='one_to_one')
            offset = len(combined['rles'])
            frame['id'] += offset
            frame['parent_a'] = np.where(frame.parent_a >= 0, frame.parent_a + offset, -1)
            frame['source_name'], frame['source_family'] = family, sid
            frame['selection_family'] = sid*16 + frame.family
            frame['in_a'] = frame.in_a & (sid == 0)
            combined['rles'].extend(entry['rles'])
            combined['thresholds'].extend(entry['thresholds'])
            tables.append(frame)
        parts[stem] = combined
    if not tables:
        return {stem: [] for stem in stems}
    table = pd.concat(tables, ignore_index=True).sort_values(['stem','id']).reset_index(drop=True)
    base.BANK, base.RECORDS, base.SOURCES = parts, {}, sources
    base.TABLES = {s:f.copy() for s,f in table.groupby('stem', sort=False)}
    extras, groups = [], {}
    for stem in stems:
        _, added, group = base.support_one(stem)
        extras.extend(added)
        groups[stem] = group
    table = table.join(pd.DataFrame(extras).set_index(['stem','id']), on=['stem','id'], validate='one_to_one')
    assert set(features) <= set(table)
    scores = []
    for fold in range(5):
        directory = BUNDLE / 'temporal_fit'
        frame = table.copy()
        prior = load(directory / f'prior_outer_{fold}.pkl')
        frame[['critic_hit','critic_miou']] = base.prior_predict(prior, frame, features)
        ranker = load(directory / f'rank_{fold}.pkl')
        rank = ranker.predict(frame[features + ['critic_hit','critic_miou']], num_threads=THREADS)
        model = load(directory / f'member0_fold{fold}.pkl')
        x = frame[model['features']]
        hurdle = np.clip(model['hit'].predict(x, num_threads=THREADS),0,1)*np.clip(model['quality'].predict(x, num_threads=THREADS),.5,1)
        scores.append(base.allocate(frame,hurdle,rank,groups))
    table['utility'] = select.family_utilities(np.mean(scores,axis=0),table.selection_family.to_numpy(),offsets={int(k):float(v) for k,v in policy['offsets'].items()})
    table.to_parquet(RUN / 'replayed_features_and_scores.parquet', index=False)
    actions.FEATURES, actions.POLICY = features, policy
    base.TABLES = {s:f for s,f in table.groupby('stem', sort=False)}
    models = [load(BUNDLE / 'temporal_fit_drop' / f'models_{f}.pkl') for f in range(5)]
    head, cutoff = drop_policy.split(':')
    predictions = {}
    for stem in stems:
        drop.SPATIAL.clear()
        state = actions.initial(stem)
        if state:
            x = drop.features_for(stem,state,base.TABLES[stem].utility.to_numpy())
            gain = np.mean([drop.predict(m,x,head) for m in models], axis=0)
            state = [v for v, keep in zip(state,gain <= float(cutoff)) if keep]
        actual = [v['rle'] for v in state]
        predictions[stem] = actual
    return predictions

def main():
    global RUN, MODEL, BUNDLE, CONFIG, THREADS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    THREADS = args.threads
    os.environ['SOLAR_FINAL_INFERENCE'] = '1'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(THREADS)
    import cv2
    import torch
    import pandas as pd
    from solar import data
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    cv2.setNumThreads(THREADS)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this environment')
    images = {p.stem: p.resolve() for p in sorted(args.images.iterdir())
              if p.is_file() and p.suffix.lower() in ('.jpeg', '.jpg')}
    paths = [p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in ('.jpeg', '.jpg')]
    if len(images) != len(paths) or not images:
        parser.error('Input must contain JPEGs with unique stems')
    for path in images.values():
        gray = data.load_gray(path)
        if gray.shape != (2048, 2048):
            raise ValueError(f'Expected 2048x2048 JPEG: {path}')
    # The shared feature functions resolve only the supplied single-image paths.
    data.test_image_path = lambda stem, root=None: images[stem]
    RUN, MODEL = args.cache.resolve(), args.weights.resolve()
    BUNDLE = MODEL
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPO / 'docs/final/RELEASE_MANIFEST.json').read_text())
    CONFIG = json.loads((REPO / 'docs/final/inference_config.json').read_text())
    for row in manifest['files']:
        path = (REPO if row['kind'] == 'source' else MODEL) / row['path']
        if sha(path) != row['sha256']:
            raise ValueError(f'Release hash mismatch: {path}')
    code_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    started = time.perf_counter()
    for row in manifest['files']:
        if row['kind'] == 'dense':
            dense(row, images, code_hash, args.device)
    sources = build_maps(list(images))
    predictions = downstream(list(images), sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(filament_id=f'{s}_{i}', segmentation_rle=r['counts'].decode())
                  for s, rs in predictions.items() for i, r in enumerate(rs, 1)],
                 columns=['filament_id', 'segmentation_rle']).to_csv(args.output, index=False)
    write(RUN / 'done.json', dict(status='DONE', source_sha=manifest['source_sha'],
          manifest_sha256=sha(REPO / 'docs/final/RELEASE_MANIFEST.json'),
          processed_images={s: len(predictions[s]) for s in images}, output_sha256=sha(args.output),
          elapsed_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
