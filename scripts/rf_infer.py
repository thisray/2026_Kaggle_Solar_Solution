"""Frozen RF pipeline: JPEG dense maps, RF support, scorer ensemble, optional refiner."""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
os.environ['SOLAR_FINAL_INFERENCE']='1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from merge_weights import sha, merge


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def load(path):
    with Path(path).open('rb') as stream:return pickle.load(stream)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:pickle.dump(value,stream,pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def image_paths(directory):
    paths=sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in ('.jpeg','.jpg'))
    result={p.stem:p for p in paths}
    assert result and len(result)==len(paths)
    return result


def gate():
    # Check host memory at finite image boundaries; the external scheduler may retry.
    if Path('/proc/meminfo').exists():
        available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024
        while available<45*1024**3:
            print('WAIT_HOST_MEMORY',available,flush=True);time.sleep(30)
            available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024


def dense_stage(a,images):
    import torch
    import cv2
    import solar_final_infer as parent
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent.RUN=a.cache;parent.MODEL=a.models;parent.THREADS=1
    manifest=json.loads((ROOT/'weights_manifest.json').read_text())
    class GuardedImages(dict):
        def items(self):
            for key,value in super().items():
                gate();yield key,value
    images=GuardedImages(images)
    for row in manifest['files']:
        if row['kind']=='dense':
            gate();parent.dense(row,images,sha(ROOT/'source_manifest.json'),a.device)
    parent.build_maps(list(images))


def rf_stage(a,images):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pycocotools import mask as mu
    from solar import data
    from solar.rf_family import candidate_alternatives,stitch_tiles
    from rfdetr import RFDETRSegNano
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for fold in range(5):
        gate()
        # Disable pretrained initialization downloads; the complete stored EMA is loaded below.
        wrapper=RFDETRSegNano(resolution=768,device=a.device,amp=False,group_detr=13,
                              gradient_checkpointing=True,pretrain_weights=None)
        wrapper.model.reinitialize_detection_head(2)
        model=wrapper.model.model
        ck=torch.load(a.models/f'rf_support/fold{fold}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['ema'],strict=True);model.eval();del ck
        positions=[0,512,1024,1280]
        for stem,path in images.items():
            gate();out=a.cache/f'rf_raw/fold{fold}/{stem}.pkl'
            if out.exists():continue
            gray=data.load_gray(path);raw=[]
            with torch.inference_mode():
                for y in positions:
                    for x in positions:
                        crop=gray[y:y+768,x:x+768]
                        tensor=torch.from_numpy(np.ascontiguousarray(crop)).float().to(a.device)/255
                        tensor=tensor[None].expand(3,-1,-1)
                        tensor=(tensor-tensor.new_tensor([.485,.456,.406])[:,None,None])/tensor.new_tensor([.229,.224,.225])[:,None,None]
                        with torch.autocast(a.device,dtype=torch.bfloat16,enabled=a.device=='cuda'):
                            output=model(tensor[None])
                        confidence=output['pred_logits'][0,:,1].sigmoid();keep=torch.where(confidence>=.05)[0]
                        if not len(keep):continue
                        masks=F.interpolate(output['pred_masks'][0,keep,None],size=(768,768),mode='bilinear',align_corners=False)[:,0]>0
                        for query,mask in zip(keep.tolist(),masks.cpu().numpy()):
                            if mask.sum()<20:continue
                            full=np.zeros((2048,2048),np.uint8);full[y:y+768,x:x+768]=mask
                            rle=mu.encode(np.asfortranarray(full))
                            raw.append(dict(query=query,tile=[x,y,768,768],rle=rle,box=mu.toBbox(rle).tolist(),score=float(confidence[query])))
            raw.sort(key=lambda v:-v['score'])
            save(out,candidate_alternatives(raw,stitch_tiles(raw)))
            print('RF_SUPPORT',fold,stem,flush=True)
        del model,wrapper;gc.collect()
        if a.device=='cuda':torch.cuda.empty_cache()


def score_stage(a,images):
    import numpy as np
    import pandas as pd
    import cv2
    from pycocotools import mask as mu
    from solar import data,select
    import solar_final_infer as parent
    import release_candidates
    import rf_descriptors
    import attack_base as base
    import attack_actions as actions
    import attack_drop as drop
    cv2.setNumThreads(1)
    data.test_image_path=lambda stem,root=None:images[stem]
    parent.RUN=a.cache;parent.MODEL=a.models;parent.BUNDLE=a.models;parent.THREADS=1
    config=json.loads((ROOT/'configs/rf.json').read_text())
    parent.CONFIG=config;release_candidates.RUN=a.cache;release_candidates.CONFIG=config
    release_candidates.load=parent.load
    sources={name:dict(test=str(a.cache/'maps'/name),oof=None) for name in ('long2x','s05','gong')}
    table,parts,groups=release_candidates.build(list(images),sources)
    features=config['feature_columns'];cols=rf_descriptors.COLUMNS
    matrices={}
    for stem,frame in table.groupby('stem',sort=False):
        values=[]
        for fold in range(5):
            unique={}
            for item in sorted(load(a.cache/f'rf_raw/fold{fold}/{stem}.pkl'),key=lambda v:-v['score']):
                if float(mu.area(item['rle']))>=32:unique.setdefault(item['rle']['counts'],item)
            values.append(rf_descriptors.descriptors(parts[stem]['rles'],list(unique.values())))
        matrices[stem]=np.stack(values)
    scores=[]
    for producer in range(5):
        frame=table.copy()
        for stem,part in table.groupby('stem',sort=False):frame.loc[part.index,cols]=matrices[stem][producer]
        member_scores=[]
        for fold in range(5):
            path=a.models/'temporal_fit'
            prior=load(path/f'prior_outer_{fold}.pkl')
            frame[['critic_hit','critic_miou']]=base.prior_predict(prior,frame,features)
            ranker=load(path/f'rank_{fold}.pkl')
            rank=ranker.predict(frame[features+['critic_hit','critic_miou']],num_threads=1)
            model=load(path/f'member0_fold{fold}.pkl');x=frame[model['features']]
            hurdle=np.clip(model['hit'].predict(x,num_threads=1),0,1)*np.clip(model['quality'].predict(x,num_threads=1),.5,1)
            member_scores.append(base.allocate(frame,hurdle,rank,groups))
        scores.append(np.mean(member_scores,axis=0))
    policy=config['policy']
    table['utility']=select.family_utilities(np.mean(scores,axis=0),table.selection_family.to_numpy(),offsets={int(k):v for k,v in policy['offsets'].items()})
    actions.FEATURES=features;actions.POLICY=policy
    base.TABLES={s:f for s,f in table.groupby('stem',sort=False)}
    models=[load(a.models/f'temporal_fit_drop/models_{f}.pkl') for f in range(5)]
    predictions={}
    for stem in images:
        drop.SPATIAL.clear();state=actions.initial(stem)
        if state:
            original=base.TABLES[stem].copy();gains=[]
            for producer in range(5):
                frame=original.copy();frame[cols]=matrices[stem][producer];base.TABLES[stem]=frame
                x=drop.features_for(stem,state,original.utility.to_numpy())
                gains.extend(drop.predict(m,x,'gain15') for m in models)
            base.TABLES[stem]=original
            state=[v for v,keep in zip(state,np.mean(gains,axis=0)<=.15) if keep]
        predictions[stem]=[v['rle'] for v in state]
    save(a.cache/'rf_predictions.pkl',predictions)
    emit(predictions,a.output)


def emit(predictions,path):
    import pandas as pd
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([dict(filament_id=f'{s}_{i}',segmentation_rle=r['counts'].decode()) for s,rs in predictions.items() for i,r in enumerate(rs,1)],columns=['filament_id','segmentation_rle']).to_csv(path,index=False)


def refiner_stage(a,images):
    import cv2
    import numpy as np
    import torch
    from release_refiner import inputs,project,make_model,predict
    seeds=load(a.cache/'rf_predictions.pkl');crops={};probabilities={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    for stem,path in images.items():
        crops[stem]=inputs(cv2.imread(str(path),0),cv2.imread(str(a.cache/f'maps/long2x/{stem}_fg.png'),0),cv2.imread(str(a.cache/f'maps/long2x/{stem}_spine.png'),0),seeds[stem])
        probabilities[stem]=[]
    members=['refiner_r34'] if a.final in ('rf_refiner_r34','rf_refiner_r34_defrag') else ['refiner_nm'] if a.final=='rf_refiner_nm' else ['refiner_r34','refiner_nm']
    means={}
    for member in members:
        # Original R34 producer used the CUDA/cuDNN defaults; NM explicitly disabled TF32.
        torch.backends.cudnn.allow_tf32=(member=='refiner_r34')
        probabilities={stem:[] for stem in images}
        for fold in range(5):
            gate();model=make_model('r34',False).float().cuda()
            ck=torch.load(a.models/f'{member}/fold{fold}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model']);del ck
            for stem in images:
                gate();q=predict(model,crops[stem][0],True);probabilities[stem].append(q)
            del model;gc.collect();torch.cuda.empty_cache()
        means[member]={stem:np.mean(probabilities[stem],axis=0,dtype=np.float32) for stem in images}
    result={}
    for stem in images:
        _,matrices,scales,_=crops[stem]
        if a.final=='rf_refiner_nm':
            q=means['refiner_nm'][stem].astype(np.float16).astype(np.float32)
        elif a.final=='rf_refiner_avg':
            left=means['refiner_r34'][stem].astype(np.float16).astype(np.float32)
            right=means['refiner_nm'][stem].astype(np.float16).astype(np.float32)
            q=((left+right)/2).astype(np.float16).astype(np.float32)
        else:q=means['refiner_r34'][stem]
        result[stem],_=project(q,matrices,scales,seeds[stem],.4)
    save(a.cache/(a.final+'_predictions.pkl'),result);emit(result,a.output)
    if a.final in ('rf_refiner_r34','rf_refiner_r34_defrag'):emit(result,a.cache/'r34.csv')
    if a.final=='rf_refiner_r34_defrag':
        from release_defrag import transform
        mode=json.loads((ROOT/'configs/rf_refiner_r34_defrag.json').read_text())['defrag_mode']
        result,_=transform(result,seeds,mode);emit(result,a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--stage',choices=['dense','support','score','refiner','all'],default='all')
    p.add_argument('--final',choices=['rf','rf_refiner_r34','rf_refiner_r34_defrag','rf_refiner_nm','rf_refiner_avg'],default='rf')
    a=p.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
    if a.final in ('rf_refiner_nm','rf_refiner_avg'):
        config=json.loads((ROOT/'configs'/f'{a.final}.json').read_text())
        if config.get('release_status')!='READY':raise RuntimeError('Candidate awaits Worker B READY receipt and frozen configuration')
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for row in manifest['files']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Source hash mismatch: '+row['path'])
    for row in json.loads((ROOT/'weights_manifest.json').read_text())['files']:
        if sha(a.models/row['path'])!=row['sha256']:raise ValueError('Model hash mismatch: '+row['path'])
    images=image_paths(a.images);started=time.time()
    identity=dict(source=sha(ROOT/'source_manifest.json'),weights=sha(ROOT/'weights_manifest.json'),images={s:sha(p) for s,p in images.items()},device=a.device)
    marker=a.cache/'CACHE_IDENTITY.json'
    if marker.exists() and json.loads(marker.read_text())!=identity:
        raise ValueError('Cache identity differs; choose a fresh cache directory')
    write(marker,identity)
    for name,function in [('dense',dense_stage),('support',rf_stage),('score',score_stage),('refiner',refiner_stage)]:
        if name=='refiner' and a.final=='rf':continue
        if a.stage in ('all',name):
            function(a,images)
            import importlib.metadata as metadata
            import platform
            import resource
            write(a.cache/(name+'_DONE.json'),dict(status='PASS',images=len(images),elapsed_seconds=time.time()-started,final=a.final,architecture=platform.machine(),python=sys.version,packages={k:metadata.version(k) for k in ('torch','numpy','segmentation-models-pytorch')},peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,source_manifest_sha256=sha(ROOT/'source_manifest.json'),weights_manifest_sha256=sha(ROOT/'weights_manifest.json')))


if __name__=='__main__':main()
