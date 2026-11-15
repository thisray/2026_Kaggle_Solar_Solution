"""Frozen G_ctrl raw dense inference and its own cross-fitted selector."""
import json
from pathlib import Path
import numpy as np


def dense_stage(a,images):
    import cv2
    import torch
    import solar_final_infer as parent
    from rf_infer import ROOT,sha,gate
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent.RUN=a.cache;parent.MODEL=a.models;parent.THREADS=1
    manifest=json.loads((ROOT/'weights_manifest.json').read_text())
    class GuardedImages(dict):
        def items(self):
            for key,value in super().items():gate();yield key,value
    for row in manifest['files']:
        if row['kind']=='gctrl_dense':parent.dense(row,GuardedImages(images),sha(ROOT/'source_manifest.json'),a.device)
    directory=a.cache/'maps/gctrl';directory.mkdir(parents=True,exist_ok=True)
    for stem in images:
        gate()
        for channel in ('fg','spine'):
            values=[cv2.imread(str(a.cache/f'fold_maps/gctrl/{fold}/{stem}_{channel}.png'),0) for fold in range(5)]
            assert all(v is not None for v in values)
            assert cv2.imwrite(str(directory/f'{stem}_{channel}.png'),np.rint(np.mean(values,axis=0,dtype=np.float32)).astype(np.uint8))


def score_stage(a,images):
    import cv2
    from solar import data,select
    import solar_final_infer as parent
    import attack_base as base
    import attack_actions as actions
    import attack_drop as drop
    import release_candidates
    from rf_infer import ROOT,load,save,gate
    cv2.setNumThreads(1);data.test_image_path=lambda stem,root=None:images[stem]
    config=json.loads((ROOT/'configs/gctrl.json').read_text())
    parent.RUN=a.cache/'gctrl';parent.MODEL=a.models;parent.BUNDLE=a.models;parent.THREADS=1;parent.CONFIG=config
    release_candidates.RUN=parent.RUN;release_candidates.CONFIG=config;release_candidates.load=parent.load
    sources={name:dict(test=str(a.cache/'maps'/('gctrl' if name=='gong' else name)),oof=None) for name in ('long2x','s05','gong')}
    table,parts,groups=release_candidates.build(list(images),sources)
    features=config['feature_columns'];scores=[]
    for fold in range(5):
        gate();frame=table.copy();directory=a.models/'gctrl_temporal_fit'
        prior=load(directory/f'prior_outer_{fold}.pkl')
        frame[['critic_hit','critic_miou']]=base.prior_predict(prior,frame,features)
        rank=load(directory/f'rank_{fold}.pkl').predict(frame[features+['critic_hit','critic_miou']],num_threads=1)
        model=load(directory/f'member0_fold{fold}.pkl');x=frame[model['features']]
        hurdle=np.clip(model['hit'].predict(x,num_threads=1),0,1)*np.clip(model['quality'].predict(x,num_threads=1),.5,1)
        scores.append(base.allocate(frame,hurdle,rank,groups))
    policy=config['policy'];table['utility']=select.family_utilities(np.mean(scores,axis=0),table.selection_family.to_numpy(),offsets={int(k):float(v) for k,v in policy['offsets'].items()})
    actions.FEATURES=features;actions.POLICY=policy;base.TABLES={s:f for s,f in table.groupby('stem',sort=False)}
    models=[load(a.models/f'gctrl_temporal_fit_drop/models_{f}.pkl') for f in range(5)]
    predictions={}
    for stem in images:
        gate();drop.SPATIAL.clear();state=actions.initial(stem)
        if state:
            x=drop.features_for(stem,state,base.TABLES[stem].utility.to_numpy())
            gain=np.mean([drop.predict(m,x,'gain15') for m in models],axis=0)
            state=[v for v,keep in zip(state,gain<=.15) if keep]
        predictions[stem]=[v['rle'] for v in state]
        print('GCTRL_SCORE',stem,len(state),flush=True)
    save(a.cache/'gctrl_predictions.pkl',predictions)
