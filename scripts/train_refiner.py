JOB='standalone-refiner-training'
import gc
import json
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
from release_refiner import make_model,predict,project
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with tmp.open('wb') as stream: pickle.dump(value, stream, pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

def load(path):
    with Path(path).open('rb') as stream: return pickle.load(stream)

def array(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream: np.save(stream, value, allow_pickle=False)
    tmp.replace(path)

def event(root, phase, **values):
    row = dict(job_id=JOB, phase=phase, epoch=time.time(), pid=os.getpid(), **values)
    write(Path(root)/'events'/f'{time.time_ns()}-{phase}.json', row)
    write(Path(root)/'STATUS.json', row)
    return row
def batch(root, keys, step, seed):
    import torch
    xs, ys = [], []
    rng = np.random.default_rng(seed + step)
    for stem, cid in keys:
        x = np.load(root/'pixels'/stem/'x.npy',mmap_mode='r')[cid].astype(np.float32)
        y = np.load(root/'teachers'/(stem+'.npy'),mmap_mode='r')[cid].astype(np.float32)
        for dim in (-1,-2):
            if rng.random() < .5: x=np.flip(x,dim); y=np.flip(y,dim)
        xs.append(x.copy()); ys.append(y.copy())
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

def sample_keys(manifest, fold):
    keys = [(s,i) for s in manifest['fit'] if manifest['folds'][s] != fold
            for i in range(manifest['rows'][s]['instances'])]
    assert all(s not in manifest['H']+manifest['D'] for s,_ in keys)
    assert all(manifest['folds'][s]!=fold for s,_ in keys)
    return keys

def schedule(keys, manifest, fold, step, seed):
    if fold == 0 and step < 200:
        pool = [k for k in keys if k[0] in manifest['F32']]
        offset = step
    else: pool=keys; offset=step-(200 if fold==0 else 0)
    n = len(pool); per_epoch=math.ceil(n/16); epoch,index=divmod(offset,per_epoch)
    order=np.random.default_rng(seed+fold*10000+epoch).permutation(n)
    return [pool[int(order[(index*16+i)%n])] for i in range(16)]

def evidence(model, root, manifest, fold, step, out, loss=None, gradients=None):
    import torch
    baseline=load(root/'protected_A.pkl'); stem=next(s for s in manifest['F32'] if manifest['rows'][s]['instances'])
    assert manifest['folds'][stem]!=0 and stem in manifest['fit']
    x=np.load(root/'pixels'/stem/'x.npy'); matrix=np.load(root/'pixels'/stem/'matrices.npy'); scales=np.load(root/'pixels'/stem/'scales.npy')
    probs=predict(model,x); owned,detail=project(probs,matrix,scales,baseline[stem],.4)
    destination=out/f'evidence-{step:04d}'; destination.mkdir(exist_ok=True)
    array(destination/'probs.npy',probs); save(destination/'owned.pkl',{stem:owned})
    effects={}
    if len(x):
        tensor=torch.from_numpy(x[:1].astype(np.float32)).cuda().requires_grad_(True)
        model.eval(); q=model(tensor).sigmoid(); q.mean().backward()
        input_grad=tensor.grad.abs().mean((0,2,3)).detach().cpu().tolist()
        with torch.inference_mode():
            for channel in range(5):
                altered=tensor.detach().clone(); altered[:,channel]=0
                effects[str(channel)]=float((model(altered).sigmoid()-q.detach()).abs().mean())
        model.zero_grad(set_to_none=True)
    else:input_grad=[]
    write(destination/'proof.json',dict(step=step,stem=stem,fold=fold,loss=loss,shape=list(x.shape),
          input_hash=sha(root/'pixels'/stem/'x.npy'),teacher_hash=sha(root/'teachers'/(stem+'.npy')),
          probability_hash=sha(destination/'probs.npy'),owned_hash=sha(destination/'owned.pkl'),
          parameter_gradient=gradients,input_gradient=input_grad,input_zero_channel_effect=effects,
          probability_range=[float(probs.min()),float(probs.max())] if probs.size else [],
          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          peak_vram_bytes=torch.cuda.max_memory_allocated(),**detail))

def train(config, member, fold, segment_end):
    import torch
    torch.set_num_threads(1); torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    root=Path(config['root']); manifest=json.loads((root/'MANIFEST.json').read_text())
    out=root/'models'/member/f'fold{fold}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'TRAIN_DONE.json').exists():return True
    seed=config['seed']; np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    keys=sample_keys(manifest,fold); total=10*math.ceil(len(keys)/16)
    if fold==0 and total<200:raise ValueError('First200 exceeds fixed10epoch budget')
    checkpoint_path=out/'last.pt'
    model=make_model(member,not checkpoint_path.exists()).float().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total)
    start=0; elapsed_prior=0.; active_prior=0.
    if checkpoint_path.exists():
        ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert ck['manifest_sha256']==sha(root/'MANIFEST.json') and ck['total']==total and ck['member']==member and ck['fold']==fold
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng']); start=ck['step']; elapsed_prior=ck['productive_seconds']; active_prior=ck['active_seconds']
    else:
        # Capture the actual legal ImageNet initialization before any update.
        save(out/'TRAIN_KEYS.pkl',keys)
        write(out/'BUDGET.json',dict(member=member,fold=fold,train_physical=len(set(s for s,_ in keys)),
              train_instances=len(keys),total_updates=total,effective_batch=16,epochs=10,seed=seed,
              commissioning_updates=200 if fold==0 else 0,all_profile_updates_included=True,
              new_GT_scope='F503 minus physical heldfold only',selection='last raw; no heldout validation selection'))
        if member=='r34' and fold==0:evidence(model,root,manifest,fold,0,out)
    started=time.monotonic(); productive=0.; losses=[]
    event(root,'TRAIN_START',member=member,fold=fold,start_step=start,target_updates=total)
    def checkpoint(completed):
        ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                step=completed,total=total,member=member,fold=fold,manifest_sha256=sha(root/'MANIFEST.json'),
                productive_seconds=elapsed_prior+productive,active_seconds=active_prior+time.monotonic()-started,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        temp=out/'last.tmp';torch.save(ck,temp);temp.replace(checkpoint_path)
    for step in range(start,total):
        low_memory=False  # The caller owns resource admission for standalone training.
        # Save every productive update at a voluntary boundary, including non-multiples of20.
        if time.time()>segment_end-180 or (root/'STOP').exists() or low_memory:
            if step==start:return False
            checkpoint(step)
            break
        tick=time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
        selected=schedule(keys,manifest,fold,step,seed)
        loss_value=0.
        micro=config.get('microbatch',16)
        if 16%micro:raise ValueError('Microbatch must divide effective16')
        for i in range(0,16,micro):
            x,y=batch(root,selected[i:i+micro],step,seed);x=x.cuda();y=y.cuda()
            q=model(x); p=q.sigmoid()
            dice=1-(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(q,y)+.5*dice.mean())/(16/micro)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite productive loss')
            loss.backward();loss_value+=float(loss.detach())
        grads={name:float(parameter.grad.abs().mean()) for name,parameter in model.named_parameters()
               if parameter.grad is not None and (name.endswith('weight') and ('conv1.' in name or 'segmentation_head' in name))}
        if not grads or not any(v>0 for v in grads.values()):raise RuntimeError('Missing productive parameter gradient')
        optimizer.step(); scheduler.step(); torch.cuda.synchronize(); productive+=time.monotonic()-tick
        losses.append(loss_value); completed=step+1
        with (out/'loss.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=completed,loss=loss_value,lr=scheduler.get_last_lr()[0]))+'\n')
        if completed%20==0 or completed in (1,total) or time.time()>segment_end-240:
            checkpoint(completed)
            write(out/'PROGRESS.json',dict(step=completed,target=total,loss=loss_value,
                  updates_per_second=(completed-start)/max(productive,1e-9),productive_seconds=elapsed_prior+productive,
                  checkpoint_sha256=sha(checkpoint_path),elapsed_segment=time.monotonic()-started))
        if member=='r34' and fold==0 and completed in (1,20,100,200):
            evidence(model,root,manifest,fold,completed,out,loss_value,grads)
            event(root,'FIRST_MEANINGFUL_EVIDENCE' if completed==1 else 'MID_RUN_MILESTONE',member=member,fold=fold,step=completed)
        if member=='r34' and fold==0 and completed==200:
            speed=200/(active_prior+time.monotonic()-started)
            budgets=json.loads((root/'DATA_READY.json').read_text())['folds']
            updates=sum(v['target_updates'] for v in budgets.values())
            # Conservative predeclared allowance: ConvNeXt2x R34 and inference+I/O2h.
            hours_r34=max(0,updates-200)/speed/3600
            hours_both=(max(0,updates-200)+2*updates)/speed/3600+2
            both=time.time()+hours_both*3600<=config['target_epoch']
            write(root/'COMMISSION200.json',dict(status='PRODUCTIVE_200_COMPLETE',updates=200,unique_restart=False,
                  first100_loss=[json.loads(v)['loss'] for v in (out/'loss.jsonl').read_text().splitlines()[:100]],
                  checkpoint_sha256=sha(checkpoint_path),actual_update_per_second=speed,
                  productive_kernel_update_per_second=200/(elapsed_prior+productive),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_vram_bytes=torch.cuda.max_memory_allocated(),
                  expected_remaining_r34_fivefold_hours=hours_r34,expected_remaining_tenmodel_hours=hours_both,
                  second_member='RUN_FIXED_CONVNEXT' if both else 'NOT_RUN_ETA',
                  inference_allowance_hours=2,convnext_multiplier=2,not_gain=True))
    else:
        write(out/'TRAIN_DONE.json',dict(status='SUCCESS',step=total,checkpoint_sha256=sha(checkpoint_path),member=member,fold=fold,last_raw=True))
    del model,optimizer,scheduler; gc.collect();torch.cuda.empty_cache()
    return (out/'TRAIN_DONE.json').exists()
