"""Composable W2 postcalibration stages for an isolated W1 release export."""
import argparse
import inspect
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]

def install_dense(arms):
    import solar_final_infer as parent
    source=inspect.getsource(parent.dense)
    needle='pred = model(torch.from_numpy(np.ascontiguousarray(image[None, None])).to(device)).float().sigmoid()[0].cpu().numpy()'
    replacement='''tensor = torch.from_numpy(np.ascontiguousarray(image[None, None])).to(device)
            if row['arm'] in W6_TTA_ARMS:
                total = None
                for dims in ((), (3,), (2,), (2, 3)):
                    value = model(tensor.flip(dims) if dims else tensor).float().sigmoid()
                    value = (value.flip(dims) if dims else value)[0].cpu().numpy()
                    assert np.isfinite(value).all()
                    if total is None: total = np.zeros_like(value, dtype=np.float32)
                    total += value
                pred = total / 4
            else:
                pred = model(tensor).float().sigmoid()[0].cpu().numpy()'''
    assert source.count(needle)==1
    source=source.replace(needle,replacement).replace('device=device, threads=THREADS)', 'device=device, threads=THREADS, w6_tta_arms=sorted(W6_TTA_ARMS))')
    mapping=dict(long2x='exp_long2x',s05='exp_s05',gong='exp_cnx_dino_gong_ft')
    parent.W6_TTA_ARMS={mapping[k] for k in arms}
    exec(compile(source,'w6_dense_float32_mean4','exec'),parent.__dict__)

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('images','models','cache','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--eiou-binarize',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--postcal-w2',action='store_true')
    p.add_argument('--dense-tta-arms',choices=['long2x'],default='long2x')
    p.add_argument('--dense-long2x-tta',action='store_true',help='Four-flip long2x TTA; implied by W6.')
    p.add_argument('--final',choices=['w6','w14'],default='w6')
    p.add_argument('--stage',choices=['dense','support','score','refiner','gctrl_plain_dense','gctrl_dense','gctrl_score','postcal_w2'],required=True)
    a=p.parse_args();arms=a.dense_tta_arms.split(',') if a.dense_tta_arms else []
    assert set(arms)<={'long2x','s05','gong'}
    import rf_infer as rf
    a.cache.mkdir(parents=True,exist_ok=True)
    recipe=dict(dense_tta_arms=arms,postcal_w2=a.postcal_w2)
    if a.final=='w14':a.eiou_binarize=True
    if a.eiou_binarize:recipe['eiou_binarize']=True
    marker=a.cache/'W6_RECIPE.json'
    if marker.exists():assert json.loads(marker.read_text())==recipe
    rf.write(marker,recipe)
    images=rf.image_paths(a.images);a.final='rf_refiner_r34'
    if a.stage in ('dense','support','score','refiner'):
        if a.stage=='dense':install_dense(arms)
        sys.argv=[str(ROOT/'scripts/rf_infer.py'),'--images',str(a.images),'--models',str(a.models),'--cache',str(a.cache),'--output',str(a.output),'--device',a.device,'--stage',a.stage,'--final',a.final]
        rf.main();return
    assert a.postcal_w2,'Extra stages require --postcal-w2'
    plain=a.cache/'gctrl_plain';plain.mkdir(exist_ok=True)
    if a.stage=='gctrl_plain_dense':
        # G_ctrl always keeps its original non-TTA long2x/s05 context.
        # Unchanged arms are reused only within this fresh cold run.
        import solar_final_infer as parent
        for name,arm in [('s05','exp_s05'),('gong','exp_cnx_dino_gong_ft')]:
            if name in arms:continue
            (plain/'fold_maps').mkdir(exist_ok=True)
            target=plain/'fold_maps'/arm
            if not target.exists():target.symlink_to((a.cache/'fold_maps'/arm).resolve(),target_is_directory=True)
        original=parent.dense
        def needed(row,images,code_hash,device):
            if row['arm']=='exp_long2x' or (row['arm']=='exp_s05' and 's05' in arms) or (row['arm']=='exp_cnx_dino_gong_ft' and 'gong' in arms):
                return original(row,images,code_hash,device)
        parent.dense=needed;a.cache=plain;rf.dense_stage(a,images);return
    if a.stage in ('gctrl_dense','gctrl_score'):
        from release_gctrl import dense_stage,score_stage
        a.cache=plain
        (dense_stage if a.stage=='gctrl_dense' else score_stage)(a,images);return
    from release_w2 import Projection,aligned_scores,keep_consensus,support
    seeds=rf.load(a.cache/'rf_predictions.pkl');baseline=rf.load(a.cache/'rf_refiner_r34_predictions.pkl')
    scores=rf.load(a.cache/'rf_scores.pkl');gctrl=rf.load(plain/'gctrl_predictions.pkl');out={}
    config=json.loads((ROOT/'configs/w6_postcal.json').read_text())
    for stem in images:
        rf.gate();q,mm,ss=rf.load(a.cache/'w6_refiner'/f'{stem}.pkl')
        from w14_component import project_refiner
        shaped=project_refiner(q,mm,ss,seeds[stem],scores[stem],Projection,config['rules'][0],a.eiou_binarize)
        assigned=aligned_scores(shaped,baseline[stem],scores[stem])
        out[stem]=keep_consensus(shaped,assigned,{'G_ctrl':support(shaped,gctrl[stem])},config['rules'][1])
    rf.save(a.cache/'w6_predictions.pkl',out);rf.emit(out,a.output)
    rf.write(a.cache/'W6_DONE.json',dict(status='PASS',images=len(images),masks=sum(map(len,out.values()))))

if __name__=='__main__':main()
