"""VSP-Net V1/V2：冻结留站折上的事件级垂向联合预测。"""
from __future__ import annotations
import argparse, importlib.util, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[2]
BASE_PATH=ROOT/'脚本/04_模型训练/70_train_multitask_fttransformer_v1.py'
spec=importlib.util.spec_from_file_location('m1_frozen',BASE_PATH)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
OUT=ROOT/'runs/vertical_structure_prediction_v1'
CFG=json.loads((ROOT/'runs/multitask_fttransformer_v1/experiment_config.json').read_text(encoding='utf-8'))
TASKS=base.TASKS
DEPTHS=['surface','middle','bottom']
INDICATORS=['chla','tn','tp']
DEPTH_ID={v:i for i,v in enumerate(DEPTHS)}

class VSPNet(nn.Module):
    def __init__(self,n_features,cfg):
        super().__init__()
        d=cfg['d_model']
        self.backbone=base.FTTransformer(n_features,10,cfg)
        # The M1 output heads are unused; retain only its exact tokenizer/encoder.
        self.backbone.heads=nn.Identity()
        self.indicator=nn.ModuleDict({q:nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(cfg['dropout'])) for q in INDICATORS})
        self.depth_embed=nn.Embedding(3,d//4)
        self.vertical=nn.ModuleDict({q:nn.Sequential(nn.Linear(d+d//4,d//2),nn.GELU(),
            nn.Dropout(cfg['dropout']),nn.Linear(d//2,1)) for q in INDICATORS})
        self.secchi=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(d//2,1))

    def forward(self,x,missing,has_depth):
        b,n_depth,n_feature=x.shape
        x=x.reshape(b*n_depth,n_feature);missing=missing.reshape(b*n_depth,n_feature)
        enc=self.backbone
        token=x.unsqueeze(-1)*enc.weight+enc.bias+missing.unsqueeze(-1)*enc.missing
        cls=enc.cls.expand(len(x),-1,-1)
        h=enc.norm(enc.encoder(torch.cat([cls,token],dim=1))[:,0]).reshape(b,3,-1)
        event=(h*has_depth.unsqueeze(-1)).sum(1)/has_depth.sum(1,keepdim=True).clamp(min=1)
        level=self.depth_embed(torch.arange(3,device=x.device)).unsqueeze(0).expand(b,-1,-1)
        result=[]
        for q in INDICATORS:
            u=self.indicator[q](event)
            out=self.vertical[q](torch.cat([u[:,None,:]+h,level],dim=-1)).squeeze(-1)
            result.append(out)
        # Task order: three Chl-a, Secchi, three TN, three TP.
        return torch.cat([result[0],self.secchi(h[:,0,:]),result[1],result[2]],dim=1)

def build_event_arrays(obs,x,missing,y,valid,train_idx,val_idx,test_ids):
    obs_values=obs.astype(str).to_numpy()
    event=np.array([z.rsplit('_',1)[0] for z in obs_values],dtype=object)
    depth=np.array([DEPTH_ID[z.rsplit('_',1)[1]] for z in obs_values])
    events,inv=np.unique(event,return_inverse=True)
    if len(np.unique(np.column_stack([inv,depth]),axis=0))!=len(obs):raise ValueError('重复采样事件水层')
    n=len(events); f=x.shape[1]
    xe=np.zeros((n,3,f),np.float32);me=np.ones((n,3,f),np.float32)
    ye=np.zeros((n,10),np.float32);ve=np.zeros((n,10),np.float32);has=np.zeros((n,3),np.float32)
    xe[inv,depth]=x;me[inv,depth]=missing;has[inv,depth]=1
    # Each task is associated with exactly one water layer; Secchi uses surface.
    task_depth=np.array([0,1,2,0,0,1,2,0,1,2])
    for j in range(10):
        sel=(depth==task_depth[j])&valid[:,j]
        ye[inv[sel],j]=y[sel,j];ve[inv[sel],j]=1
    tr=np.unique(inv[train_idx]);va=np.unique(inv[val_idx])
    if np.intersect1d(tr,va).size:raise ValueError('同一事件跨训练和验证')
    test_event={task:inv[obs.get_indexer(ids)] for task,ids in test_ids.items()}
    return xe,me,ye,ve,has,tr,va,test_event

def predict(model,arrays,indices,cfg,device):
    xe,me,_,_,has=arrays[:5];out=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),cfg['batch_size']):
            ix=indices[start:start+cfg['batch_size']]
            args=[torch.from_numpy(a[ix]).to(device) for a in (xe,me,has)]
            out.append(model(*args).cpu().numpy())
    return np.concatenate(out)

def vertical_loss(pred,y,valid,mean,std,scale):
    # Predictions/targets are standardized taskwise; compare differences in original units.
    rawp=pred*std+mean;rawy=y*std+mean
    losses=[]
    for qi,start in enumerate((0,4,7)):
        for a,b in ((start,start+1),(start+1,start+2)):
            mask=(valid[:,a]>0)&(valid[:,b]>0)
            if mask.any():
                error=((rawp[mask,a]-rawp[mask,b])-(rawy[mask,a]-rawy[mask,b]))/scale[qi]
                losses.append(nn.functional.smooth_l1_loss(error,torch.zeros_like(error),beta=1.0))
            else:losses.append(pred.sum()*0)
    return sum(losses)/6

def run_one(frame,manifests,targets,names,region,fold,seed,variant,device):
    obs,xf,y,label_ok,train_idx,val_idx,test_ids,forbidden,val_stations=base.fold_arrays(
        frame,manifests,targets,fold,seed,CFG)
    x,missing=base.prepare_x(xf.loc[:,names].to_numpy(float),train_idx)
    ys,mean,std=base.prepare_y(y,train_idx,label_ok)
    arrays=build_event_arrays(obs,x,missing,ys,label_ok,train_idx,val_idx,test_ids)
    xe,me,ye,ve,has,tr,va,te=arrays
    # The normalization source is the same eligible training labels as M1.
    scales=[]
    for start in (0,4,7):
        vals=np.concatenate([y[train_idx,j][label_ok[train_idx,j]] for j in range(start,start+3)])
        scales.append(max(float(vals.std()),1e-6))
    mean_t=torch.tensor(mean,dtype=torch.float32,device=device)
    std_t=torch.tensor(std,dtype=torch.float32,device=device)
    scale_t=torch.tensor(scales,dtype=torch.float32,device=device)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=VSPNet(len(names),CFG).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    def loader(ix,shuffle):
        z=TensorDataset(*[torch.from_numpy(a[ix]) for a in (xe,me,ye,ve,has)])
        return DataLoader(z,batch_size=CFG['batch_size'],shuffle=shuffle,num_workers=0,pin_memory=device.type=='cuda')
    train=loader(tr,True);val=loader(va,False)
    best=float('inf');state=None;epoch_used=0;patience=0;started=time.perf_counter()
    for epoch in range(CFG['max_epochs']):
        model.train()
        for xb,mb,yb,vb,hb in train:
            xb,mb,yb,vb,hb=[z.to(device,non_blocking=True) for z in (xb,mb,yb,vb,hb)]
            opt.zero_grad(set_to_none=True);pred=model(xb,mb,hb)
            loss=base.masked_loss(pred,yb,vb,'M1')
            if variant=='V2':loss=loss+.1*vertical_loss(pred,yb,vb,mean_t,std_t,scale_t)
            if not torch.isfinite(loss):raise ValueError('损失非有限')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        # Early stopping exactly follows M1's sample-average masked base loss.
        model.eval();total=0.;denom=0.
        with torch.no_grad():
            for xb,mb,yb,vb,hb in val:
                xb,mb,yb,vb,hb=[z.to(device,non_blocking=True) for z in (xb,mb,yb,vb,hb)]
                total+=float(((model(xb,mb,hb)-yb).square()*vb).sum());denom+=float(vb.sum())
        current=total/max(denom,1)
        if current<best-1e-5:
            best=current;epoch_used=epoch+1;patience=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience+=1
            if patience>=CFG['patience']:break
    if state is None:raise RuntimeError('未取得有效权重')
    model.load_state_dict(state)
    rows=[]
    for j,task in enumerate(TASKS):
        ids=test_ids[task];p=predict(model,arrays,te[task],CFG,device)[:,j]*std[j]+mean[j]
        p=np.maximum(p,0);truth=y[obs.get_indexer(ids),j]
        assert np.isfinite(truth).all() and len(p)==len(ids)
        rows.append(pd.DataFrame({'obs_id':ids,'task':task,'fold':fold,'seed':seed,'variant':variant,
            'y_true':truth,'y_pred':p,'station_id':frame.loc[ids,'station'].to_numpy(),
            'water_zone':region.loc[ids,'water_zone_en'].to_numpy()}))
    weight=OUT/'weights'/f'{variant}_{fold}_seed{seed}.pt';weight.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state':state,'features':names,'tasks':TASKS,'target_mean':mean,
        'target_std':std,'vertical_scale':scales,'config':CFG,'best_epoch':epoch_used},weight)
    record={'variant':variant,'fold':fold,'seed':seed,'best_epoch':epoch_used,'validation_loss':best,
        'training_events':len(tr),'validation_events':len(va),'forbidden_test_stations':len(forbidden),
        'seconds':time.perf_counter()-started,'weight':str(weight.relative_to(ROOT))}
    return pd.concat(rows,ignore_index=True),record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(1,6))
    parser.add_argument('--seed',type=int,choices=CFG['seeds']);parser.add_argument('--variant',choices=['V1','V2'])
    a=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    cfg,names,frame,manifests,targets,region=base.read_assets()
    if cfg!=CFG:raise ValueError('M1配置发生变化')
    if not torch.cuda.is_available():raise RuntimeError('GPU不可用')
    torch.set_num_threads(4);device=torch.device('cuda')
    for fold in ([a.fold] if a.fold else range(1,6)):
      for seed in ([a.seed] if a.seed is not None else CFG['seeds']):
       for variant in ([a.variant] if a.variant else ['V1','V2']):
        path=OUT/'predictions'/f'{variant}_fold_{fold}_seed{seed}.csv'
        if path.exists() and path.stat().st_size>100:print('REUSE',path.name,flush=True);continue
        rows,record=run_one(frame,manifests,targets,names,region,f'fold_{fold}',seed,variant,device)
        path.parent.mkdir(parents=True,exist_ok=True);rows.to_csv(path,index=False,encoding='utf-8-sig')
        with (OUT/'training_registry.jsonl').open('a',encoding='utf-8') as fh:
            fh.write(json.dumps(record,ensure_ascii=False)+'\n')
        print('DONE',path.name,'seconds',round(record['seconds'],1),flush=True)

if __name__=='__main__':main()
