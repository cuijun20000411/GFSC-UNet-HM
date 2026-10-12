"""D3验证公共组件。数据隔离/统计仅依赖NumPy；训练时复用原始D3引擎。"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time
from dataclasses import asdict
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, distance_transform_edt

VERSION = "d3_validation_20260915_v1"
MODELS = ("D3", "D3A10")
DEPENDENCIES = ("rh4_final_strict_light.py", "fusion_final_strict.py",
                "f3_direct_ablation_final_strict.py", "d3_direct_ablation_final_strict.py")

def normalize(x):
    return json.loads(json.dumps(x, default=lambda v: v.item() if isinstance(v, np.generic)
                                else v.tolist() if isinstance(v, np.ndarray) else str(v), sort_keys=True))

def digest(x):
    return hashlib.sha256(json.dumps(normalize(x), sort_keys=True).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(normalize(value), ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp,path)

def write_csv(path, df):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); df.to_csv(tmp,index=False); os.replace(tmp,path)

def csv_list(value):
    return [x.strip() for x in value.split(',') if x.strip()]

def common_parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--data-root',default=os.environ.get('SCUNET_DATA_ROOT','/etc/cj/boundary/duoyuan/data'))
    p.add_argument('--pollutants',default='no2,o3,pm25')
    p.add_argument('--d3-root',default='/etc/cj/boundary/f3_direct_ablation_final_strict_v1')
    p.add_argument('--a10-root',default='/etc/cj/boundary/d3_direct_ablation_final_strict_v1')
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--log-interval',type=int,default=10)
    p.add_argument('--validation-mask-seed',type=int,default=42)
    p.add_argument('--fold-start',type=int,default=0)
    p.add_argument('--max-folds',type=int,default=0,help='0表示运行全部剩余折；小值用于试跑')
    p.add_argument('--summary-only',action='store_true')
    p.add_argument('--preflight',action='store_true',help='只检查数据/协议并保存划分，不训练')
    return p

class FieldData:
    def __init__(self, root, pollutant):
        self.pollutant=pollutant; self.folder=Path(root)/pollutant
        self.paths=[]
        def load(n,**kw):
            self.paths.append(self.folder/n); return np.load(self.folder/n,**kw)
        raw=self.folder/'conc_img_raw.npy'; self.legacy=None
        if raw.exists(): self.conc=load(raw.name,mmap_mode='r')
        else:
            self.conc=load('conc_img_full.npy',mmap_mode='r')
            self.paths.append(self.folder/'conc_scaler.pkl')
            self.legacy=joblib.load(self.paths[-1])
        self.cov=load('cov_img_full.npy',mmap_mode='r')
        self.coord=load('coord_img.npy')
        self.positions=load('station_positions.npy').astype(np.int64)
        self.original_times=load('valid_time_indices.npy').astype(np.int64)
        self.splits={k:np.asarray(v,dtype=np.int64) for k,v in load('time_splits.npy',allow_pickle=True).item().items()}
        # 严格使用有效区域文件，避免把补齐网格误当作实际研究区域。
        self.valid=load('valid_grid_mask.npy').astype(np.float32)
        self.shape=tuple(self.conc.shape[1:]);self.feature_dim=int(self.cov.shape[1]);self.n=len(self.positions)
        if self.shape!=(32,32):raise ValueError(f'{pollutant}: expected original 32x32 grid, got {self.shape}')
        expected={'no2':39,'o3':39,'pm25':40}[pollutant]
        if self.feature_dim!=expected:raise ValueError(f'{pollutant}: expected {expected} covariate channels')
        if self.cov.shape[0]!=len(self.conc) or self.cov.shape[2:]!=self.shape or self.coord.shape!=(*self.shape,2):
            raise ValueError('Concentration/covariate/coordinate shapes disagree')
        if self.valid.shape!=self.shape or not np.isin(self.valid,[0,1]).all():raise ValueError('Invalid domain mask')
        if len(self.original_times)!=len(self.conc) or len(np.unique(self.original_times))!=len(self.conc):raise ValueError('Nonunique/missing time IDs')
        if self.n<3 or self.positions.shape!=(self.n,2) or len(np.unique(self.positions,axis=0))!=self.n:
            raise ValueError('Invalid station positions')
        if (self.positions<0).any() or (self.positions>=np.array(self.shape)).any():raise ValueError('Station outside array')
        rr,cc=self.positions.T
        if not (self.valid[rr,cc]>0).all():raise ValueError('Station outside valid domain')
        if not np.isfinite(self.coord).all():raise ValueError('Nonfinite coordinates')
        prev=-1
        for mode in ('train','val','test'):
            ids=self.splits[mode]
            if not len(ids) or (np.diff(ids)<=0).any() or ids.min()<=prev or ids.max()>=len(self.conc):
                raise ValueError('Time splits must be nonempty, chronological and disjoint')
            prev=int(ids.max())
        self._scalers={}
        self.identity={p.name:file_hash(p) for p in self.paths}

    def raw(self,t):
        a=np.asarray(self.conc[int(t)],dtype=np.float32).copy()
        if self.legacy is not None:a=a*float(self.legacy.scale_[0])+float(self.legacy.mean_[0])
        return a.astype(np.float32)

    def candidates(self,heldout):
        held=tuple(sorted(set(int(i) for i in heldout)))
        if not held or any(i<0 or i>=self.n for i in held):raise ValueError('Invalid held-out IDs')
        keep=np.asarray([i for i in range(self.n) if i not in held],dtype=int)
        if len(keep)<2:raise ValueError('At least two non-held-out stations are required')
        return keep

    def scaler(self,heldout):
        key=tuple(sorted(heldout))
        if key not in self._scalers:
            ids=self.candidates(key);rr,cc=self.positions[ids].T
            # 只收集训练时段、非留出站的标签；任何留出区域真值均不参与拟合。
            vals=np.concatenate([self.raw(t)[rr,cc] for t in self.splits['train']])
            vals=vals[np.isfinite(vals)]
            if not len(vals):raise ValueError('No finite training labels outside held-out region')
            mean=float(vals.mean());std=float(vals.std())
            self._scalers[key]=(mean,std if std>=1e-8 else 1.0)
        return self._scalers[key]

    def expected_keys(self,heldout):
        rows=[]
        for t in self.splits['test']:
            a=self.raw(t)
            for i in heldout:
                y=float(a[tuple(self.positions[i])])
                if np.isfinite(y): rows.append((int(i),int(self.original_times[t]),y))
        return pd.DataFrame(rows,columns=['station','time_idx','true'])

def neighbor_features(sparse,observed,radius,mean,std):
    # 与原rh4_final_strict_light.py的observed_neighbor_features逐式一致。
    size=2*radius+1;area=float(size*size)
    values=np.where(observed,sparse,0).astype(np.float32)
    count=uniform_filter(observed.astype(np.float32),size=size,mode='constant')*area
    total=uniform_filter(values,size=size,mode='constant')*area
    squares=uniform_filter(values*values,size=size,mode='constant')*area
    local=total/np.maximum(count,1)
    sd=np.sqrt(np.maximum(squares/np.maximum(count,1)-local*local,0))
    local[count<.5]=mean;sd[count<.5]=std
    return np.stack([local,sd,np.full_like(local,mean)]).astype(np.float32)

class FoldDataset:
    """一个fold只传入整组heldout；接口兼容原训练引擎。"""
    def __init__(self,data,heldout,mode,config):
        self.data=data;self.heldout=tuple(sorted(heldout));self.mode=mode;self.config=config
        self.candidates=data.candidates(self.heldout)
        self.time_indices=data.splits[mode];self.train_time_indices=data.splits['train']
        self.station_positions=data.positions;self.total_stations=data.n;self.base_channels=7
        self.F_dim=data.feature_dim;self.cov_img_full=data.cov;self.valid_grid_mask=data.valid
        self.fold_mean,self.fold_std=data.scaler(self.heldout)
        self.mask_ratio=config.START_MASK_RATIO
    def __len__(self):return len(self.time_indices)
    def set_mask_ratio(self,r):self.mask_ratio=float(r)
    def selected(self,t):
        n=min(len(self.candidates),max(1,int(round(len(self.candidates)*self.mask_ratio))))
        if self.mode=='train':return random.sample(self.candidates.tolist(),n)
        return np.random.default_rng(self.config.VALIDATION_MASK_SEED+int(t)).choice(self.candidates,n,replace=False).tolist()
    def numpy_sample(self,idx):
        t=int(self.time_indices[idx]);raw=self.data.raw(t)
        norm=((raw-self.fold_mean)/self.fold_std).astype(np.float32)
        sparse=np.full(self.data.shape,np.nan,dtype=np.float32)
        rr,cc=self.station_positions[self.candidates].T;sparse[rr,cc]=norm[rr,cc]
        sparse[~np.isfinite(sparse)]=np.nan
        target=np.full_like(sparse,np.nan)
        target_ids=self.heldout if self.mode=='test' else self.selected(t)
        for i in target_ids:
            r,c=self.station_positions[i];target[r,c]=norm[r,c] if np.isfinite(norm[r,c]) else np.nan;sparse[r,c]=np.nan
        observed=np.isfinite(sparse)&(self.valid_grid_mask>0)
        vals=sparse[observed];mean=float(vals.mean()) if len(vals) else 0.
        std=float(vals.std()) if len(vals)>1 else 1.
        if std<1e-6:std=1.
        filled=np.where(observed,sparse,mean).astype(np.float32)
        near=neighbor_features(sparse,observed,self.config.FIXED_NEIGHBOR_RADIUS,mean,std)
        dist=distance_transform_edt(~observed).astype(np.float32)
        dist/=max(float(dist[self.valid_grid_mask>0].max()),1.);dist*=self.valid_grid_mask
        base=np.concatenate([filled[None],near,self.data.coord.transpose(2,0,1).astype(np.float32),dist[None]])
        cov=np.asarray(self.data.cov[t],dtype=np.float32).copy()
        if not np.isfinite(cov).all():raise ValueError(f'Nonfinite covariates at time {t}; fix preprocessing')
        return {'input':np.concatenate([base,cov]),'channel_mask':observed[None].astype(np.float32),
                'known_values':np.nan_to_num(sparse,nan=mean,posinf=mean,neginf=mean)[None],
                'target':target[None],'valid_grid_mask':self.valid_grid_mask[None],
                'infer_mask':((self.valid_grid_mask>0)&~observed)[None].astype(np.float32),
                'time_idx':int(self.data.original_times[t]),'array_idx':t}
    def __getitem__(self,idx):
        import torch
        return {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in self.numpy_sample(idx).items()}

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if not len(y) or not (np.isfinite(y).all() and np.isfinite(p).all()):raise ValueError('Empty/nonfinite predictions')
    e=p-y;sse=float(np.square(e).sum());sst=float(np.square(y-y.mean()).sum())
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.square(e).mean())),
            'R2':1-sse/sst if len(y)>1 and sst>0 else float('nan')}

def validate_predictions(frame, data, heldout):
    cols=['station','time_idx']
    if frame.duplicated(cols).any():raise ValueError('Duplicate station-time predictions')
    expected=data.expected_keys(heldout)
    merged=expected.merge(frame,on=cols,how='outer',suffixes=('_expected',''),indicator=True)
    if not (merged['_merge']=='both').all():raise ValueError('Missing/extra test station-time predictions')
    if not np.isfinite(frame['prediction']).all():raise ValueError('Nonfinite test prediction')
    if not np.allclose(merged['true_expected'],merged['true'],rtol=2e-6,atol=2e-5):raise ValueError('Prediction truths do not match current data')

@contextlib.contextmanager
def output_lock(root):
    # 进程崩溃后内核自动释放锁；避免同一结果目录并发写入。
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('w') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Another process is writing {root}')
        yield

def source_identity():
    folder=Path(__file__).parent
    return {p.name:file_hash(p) for p in [folder/x for x in (*DEPENDENCIES,'d3_validation_common.py',
            'd3_spatial_block_final_strict.py','d3_multiseed_final_strict.py')] if p.exists()}

def reference_dir(args,model):
    root=Path(args.d3_root if model=='D3' else args.a10_root)
    return root if root.name==model else root/model

def check_reference_config(proto,args,model,pollutant,check_training=True):
    # seed必须明确为42；额外种子的模型不得伪装成已有seed=42结果。
    expected={'seed':42,'pollutant':pollutant,'scb_channel_attention':False,
        'importance_beta':0.,'curriculum_masking':True,'tv_multiplier':1.,'laplacian_multiplier':1.,
        'prediction_clipping':False,'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
        'learning_rate':1e-4,'weight_decay':1e-5,'start_mask_ratio':.05,'final_mask_ratio':.25,
        'curriculum_epoch':50,'group_width':8,'group_dropout':.25,'gate_initial_bias':-3.,
        'tv_weights':{'no2':.01,'o3':.10,'pm25':.05},
        'laplacian_weights':{'no2':.01,'o3':.01,'pm25':.01},
        'bottleneck_fusion':model=='D3','decoder_fusion':model=='D3'}
    if not check_training:
        for k in ('epochs','patience','batch_size'):expected.pop(k)
    errors={k:{'expected':v,'observed':proto.get(k)} for k,v in expected.items() if normalize(proto.get(k))!=normalize(v)}
    if proto.get('ablation',{}).get('code')!=model:errors['ablation_code']=proto.get('ablation')
    if check_training and args.validation_mask_seed!=42:errors['validation_mask_seed']='Existing reference used 42'
    if errors:raise ValueError(f'{model}/{pollutant} reference config mismatch: {errors}')

def groups_from_reference(args,data):
    # 分组保持与最终D3完全一致，不能为每个外层块重新挑选结构。
    path=reference_dir(args,'D3')/f'protocol_{data.pollutant}.json'
    proto=read_json(path)
    check_reference_config(proto,args,'D3',data.pollutant,check_training=False)
    groups=proto['resolved_groups']
    flat=[int(i) for g in groups for i in g['indices']]
    if sorted(flat)!=list(range(data.feature_dim)):raise ValueError('Reference groups do not partition all channels')
    if any(g['kind'] not in ['spatial','global'] for g in groups):raise ValueError('Invalid group kind')
    return groups,file_hash(path)

class NeuralEngine:
    def __init__(self,args,seed):
        try:
            import d3_direct_ablation_final_strict as d3
        except ModuleNotFoundError as exc:
            raise RuntimeError('请在原PyTorch环境运行，并将压缩包全部文件置于同一目录。缺少依赖: '+str(exc)) from exc
        self.d3=d3;self.e=d3.engine;self.core=d3.core;self.fusion=d3.fusion
        d3.GLOBAL_SEED=int(seed);d3.install_engine_overrides()
        self.seed=int(seed);self.args=args
        c=d3.Config();self.config=c
        # 显式固定D3配置，避免旧shell中FUSION_*变量改变实验。
        fixed={'DATA_ROOT_DIR':args.data_root,'EPOCHS':args.epochs,'PATIENCE':args.patience,
            'BATCH_SIZE':args.batch_size,'NUM_WORKERS':args.num_workers,'LOG_INTERVAL':args.log_interval,
            'VALIDATION_MASK_SEED':args.validation_mask_seed,'LR':1e-4,'WEIGHT_DECAY':1e-5,
            'START_MASK_RATIO':.05,'FINAL_MASK_RATIO':.25,'CURRICULUM_EPOCH':50,
            'FIXED_NEIGHBOR_RADIUS':2,'HUBER_WEIGHT':.8,'MSE_WEIGHT':.2,'IMPORTANCE_BETA':0.,
            'DIFFUSION_COEFF':.1,'TV_WEIGHTS':{'no2':.01,'o3':.10,'pm25':.05},
            'LAPLACIAN_WEIGHTS':{'no2':.01,'o3':.01,'pm25':.01},'GROUP_WIDTH':8,
            'GROUP_DROPOUT':.25,'GATE_BIAS':-3.,'USE_SCB_ATTENTION':False,
            'REUSE_CHECKPOINTS':True,'MODEL_VERSION':VERSION}
        for k,v in fixed.items():setattr(c,k,v)
        self.d3.SOURCE_SHA256=digest(source_identity());self.e.SOURCE_SHA256=self.d3.SOURCE_SHA256
        self.device=self.e.DEVICE
        if self.e.DEVICE!=self.core.DEVICE:
            raise RuntimeError('Training engine and data-transfer device disagree')
        print(f'device={self.device} | training seed={seed} | validation mask seed={args.validation_mask_seed}',flush=True)

    def train_predict(self,data,heldout,fold,model_name,root,fingerprint):
        import torch,gc
        ab=self.d3.D3Ablation('D3','Full D3 without channel attention') if model_name=='D3' else self.d3.ABLATIONS['D3A10']
        groups_raw,_=groups_from_reference(self.args,data)
        groups=[self.fusion.GroupSpec(g['name'],tuple(g['indices']),g['kind']) for g in groups_raw]
        effective,width=self.e.effective_groups_and_width(self.config,ab,groups,data.feature_dim)
        datasets={mode:FoldDataset(data,heldout,mode,self.config) for mode in ('train','val','test')}
        old_factory=self.e.make_dataset;old_fp=self.e.experiment_fingerprint
        self.e.make_dataset=lambda config,pollutant,station_idx,mode,ablation,variant:datasets[mode]
        self.e.experiment_fingerprint=lambda *a,**k:fingerprint
        ckpt=Path(root)/'best_model.pth'
        try:
            marker=self.e.completion_marker(ckpt)
            if marker.exists() and read_json(marker).get('experiment_fingerprint')!=fingerprint:
                raise RuntimeError(f'Existing checkpoint belongs to different configuration: {root}')
            model,info=self.e.train_fold_model(self.config,data.pollutant,fold,ab,groups,effective,width,
                                               'frozen D3 reference groups',ckpt)
            test=datasets['test'];loader=self.core.make_loader(test,self.config,shuffle=False,batch_size=1)
            records=[];gate=[];start=time.perf_counter()
            if self.device.type=='cuda':torch.cuda.synchronize()
            with torch.no_grad():
                for batch in loader:
                    inp,obs,known,target,valid=self.core.move_batch(batch)
                    out=model(inp,obs,known)
                    for i in heldout:
                        r,c=data.positions[i];truth=float(target[0,0,r,c].cpu())
                        if not np.isfinite(truth):continue
                        pred=float(out[0,0,r,c].cpu())
                        if not np.isfinite(pred):raise RuntimeError('Nonfinite neural prediction')
                        records.append({'station':i,'time_idx':int(batch['time_idx'].item()),
                                        'true':truth*test.fold_std+test.fold_mean,
                                        'prediction':pred*test.fold_std+test.fold_mean})
                    gate.append(model.gate_means())
            if self.device.type=='cuda':torch.cuda.synchronize()
            info.update({'test_loop_seconds':time.perf_counter()-start,'fold_train_mean':test.fold_mean,
                         'fold_train_std':test.fold_std})
            if gate:
                for k in gate[0]:info['gate_'+k+'_mean']=float(np.mean([x[k] for x in gate]))
            frame=pd.DataFrame(records);validate_predictions(frame,data,heldout)
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            return frame,info
        finally:self.e.make_dataset=old_factory;self.e.experiment_fingerprint=old_fp

def run_signature(args,data,model,seed,heldout,fold,extra=None):
    groups,refhash=groups_from_reference(args,data)
    return digest({'version':VERSION,'sources':source_identity(),'data':data.identity,
        'model':model,'seed':seed,'heldout':list(heldout),'fold':fold,'groups':groups,'reference_hash':refhash,
        'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
        'num_workers':args.num_workers,'validation_mask_seed':args.validation_mask_seed,'extra':extra})

def load_completed(root,fingerprint,data,heldout):
    root=Path(root);marker=root/'complete.json'
    if not marker.exists():return None
    m=read_json(marker)
    if m.get('fingerprint')!=fingerprint:raise RuntimeError(f'Configuration/data changed at {root}; choose a new output directory')
    f=pd.read_csv(root/'predictions.csv');validate_predictions(f,data,heldout)
    if len(f)!=m['n_predictions'] or file_hash(root/'predictions.csv')!=m['prediction_sha256']:
        raise RuntimeError('Completed prediction file was changed')
    return f,read_json(root/'training_info.json')

def save_completed(root,fingerprint,frame,info):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    write_csv(root/'predictions.csv',frame);write_json(root/'training_info.json',info)
    write_json(root/'complete.json',{'fingerprint':fingerprint,'n_predictions':len(frame),
                                   'prediction_sha256':file_hash(root/'predictions.csv'),'status':'completed'})

def summarize(root,frames,status_rows):
    root=Path(root);out=root/'metrics';out.mkdir(parents=True,exist_ok=True)
    status=pd.DataFrame(status_rows);write_csv(out/'completion_status.csv',status)
    if not frames:return pd.DataFrame()
    pred=pd.concat(frames,ignore_index=True)
    keys=['model','pollutant','seed']
    if pred.duplicated(keys+['station','time_idx']).any():raise ValueError('Duplicate merged predictions')
    station=[];folds=[];aggregates=[]
    for group,s in pred.groupby(keys+['fold','station'],sort=True):
        station.append(dict(zip(keys+['fold','station'],group),n_test=len(s),**metrics(s.true,s.prediction)))
    stations=pd.DataFrame(station)
    for group,s in pred.groupby(keys+['fold'],sort=True):
        folds.append(dict(zip(keys+['fold'],group),n_test=len(s),n_stations=s.station.nunique(),**metrics(s.true,s.prediction)))
    for group,s in pred.groupby(keys,sort=True):
        subset=stations
        st=status
        for k,v in zip(keys,group):subset=subset[subset[k]==v];st=st[st[k]==v]
        row=dict(zip(keys,group),n_stations=s.station.nunique(),n_test_observations=len(s),
                 completed_folds=int(st.complete.sum()),expected_folds=len(st),complete=bool(len(st) and st.complete.all()))
        row.update({'pooled_'+k:v for k,v in metrics(s.true,s.prediction).items()})
        for m in ['MAE','RMSE','R2']:row['mean_station_'+m]=float(subset[m].mean())
        row['median_station_R2']=float(subset.R2.median());aggregates.append(row)
    agg=pd.DataFrame(aggregates)
    write_csv(out/'aggregate_metrics.csv',agg);write_csv(out/'station_metrics.csv',stations)
    write_csv(out/'fold_metrics.csv',pd.DataFrame(folds));write_csv(root/'predictions'/'test_predictions.csv',pred)
    # 相同污染物/种子的已完成方法必须覆盖相同站点-时间真值。
    for (p,seed),group in agg[agg.complete].groupby(['pollutant','seed']):
        ref=None
        for name in group.model:
            s=pred[(pred.pollutant==p)&(pred.seed==seed)&(pred.model==name)].sort_values(['station','time_idx'])
            a=s[['station','time_idx','true']].to_numpy()
            if ref is not None and (ref.shape!=a.shape or not np.allclose(ref,a,rtol=2e-6,atol=2e-5)):
                raise RuntimeError('Completed methods do not share test samples')
            ref=a
    return agg

def attach(frame,model,pollutant,seed,fold):
    out=frame.copy();out['model']=model;out['pollutant']=pollutant;out['seed']=seed;out['fold']=fold
    out['error']=out.prediction-out.true
    return out

def fold_range(args,n):
    stop=n if args.max_folds==0 else min(n,args.fold_start+args.max_folds)
    if not 0<=args.fold_start<stop:raise ValueError('Invalid fold range')
    return range(args.fold_start,stop)
