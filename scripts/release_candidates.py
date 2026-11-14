def build(stems, sources):
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
    return table, parts, groups
