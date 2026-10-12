"""空间分块留出：重新训练D3/D3A10，整块站点在该折完全隔离。"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from d3_validation_common import *

BASELINES=('SpatialMean','IDW','KNN','ExtraTrees','RandomForest','XGBoost')

def make_blocks(data,args):
    if not 2<=args.blocks<data.n:raise ValueError('Invalid number of spatial blocks')
    coords=data.coord[data.positions[:,0],data.positions[:,1]].astype(float)
    source='coord_img.npy at station positions (stored coordinate units)'
    if args.coordinates_csv:
        tab=pd.read_csv(args.coordinates_csv)
        if not {'station','x','y'}.issubset(tab):raise ValueError('Coordinates CSV requires station,x,y')
        if tab.station.duplicated().any() or sorted(tab.station.tolist())!=list(range(data.n)):
            raise ValueError('Coordinates CSV must cover each station ID once')
        coords=tab.sort_values('station')[['x','y']].to_numpy(float)
        source='user-supplied x,y in common projected units'
    if not np.isfinite(coords).all() or len(np.unique(coords,axis=0))<args.blocks:
        raise ValueError('Invalid coordinates for spatial clustering')
    # 不按各轴分别标准化，避免改变x/y的相对距离。
    km=KMeans(n_clusters=args.blocks,random_state=args.split_seed,n_init=20).fit(coords)
    order=np.lexsort((km.cluster_centers_[:,1],km.cluster_centers_[:,0]))
    remap={int(old):new for new,old in enumerate(order)}
    labels=np.array([remap[int(v)] for v in km.labels_])
    parts=[np.flatnonzero(labels==k).tolist() for k in range(args.blocks)]
    for part in parts:data.candidates(part)
    payload={'method':'coordinate-only KMeans; labels ordered by centroid x then y',
             'n_blocks':args.blocks,'split_seed':args.split_seed,'coordinate_source':source,
             'coordinates':coords.tolist(),'station_positions':data.positions.tolist(),'heldout_by_fold':parts}
    payload['split_signature']=digest(payload)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    path=root/'spatial_split.json'
    if path.exists() and read_json(path)!=payload:
        raise RuntimeError('Spatial partition/config changed; use a new output directory')
    write_json(path,payload)
    write_csv(root/'station_blocks.csv',pd.DataFrame({'station':np.arange(data.n),'block':labels,
               'x':coords[:,0],'y':coords[:,1],'grid_row':data.positions[:,0],'grid_col':data.positions[:,1]}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,6))
    for k in range(args.blocks):
        ids=np.flatnonzero(labels==k);ax.scatter(coords[ids,0],coords[ids,1],label=f'Block {k}: {len(ids)} stations')
        for i in ids:ax.annotate(str(i),coords[i],xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.set(xlabel='Station x',ylabel='Station y',title='Spatial folds fixed before model fitting')
    ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=9);fig.tight_layout()
    fig.savefig(root/'spatial_blocks.png',dpi=200);plt.close(fig)
    return parts,payload

def interpolation(coords,values,target,method,param):
    if not len(values):raise ValueError('No contemporaneous input station available')
    if method=='SpatialMean':return float(np.mean(values))
    distance=np.linalg.norm(coords-target,axis=1)
    if np.any(distance<1e-8):return float(np.mean(values[distance<1e-8]))
    if method=='KNN':return float(np.mean(values[np.argsort(distance)[:min(int(param),len(values))]]))
    w=1/np.maximum(distance,1e-6)**float(param)
    return float(np.sum(w*values)/w.sum())

def tabular_feature(coords,values,target,coord_feature,covariates,k,shape):
    # 与baseline_final_strict.py的nearest_features保持同一输入定义。
    distances=np.linalg.norm(coords-target[None],axis=1)
    order=np.argsort(distances)[:min(k,len(distances))]
    nearest=values[order].astype(np.float32)
    d=(distances[order]/max(float(np.hypot(shape[0]-1,shape[1]-1)),1.)).astype(np.float32)
    nearest=np.pad(nearest,(0,k-len(nearest)),constant_values=0.)
    d=np.pad(d,(0,k-len(d)),constant_values=1.)
    w=1/np.maximum(distances,1e-6)
    summary=[np.sum(w*values)/w.sum(),values.mean(),values.std(),values.min(),values.max()]
    return np.concatenate([nearest,d,np.asarray(summary,dtype=np.float32),
        np.nan_to_num(coord_feature,nan=0.),np.nan_to_num(covariates,nan=0.)]).astype(np.float32)

def tabular_records(data,heldout,mode,k):
    safe=data.candidates(heldout);mean,std=data.scaler(heldout)
    features=[];records=[];ys=[]
    for t in data.splits[mode]:
        raw=data.raw(t);normalized=(raw-mean)/std
        targets=heldout if mode=='test' else safe
        for target in targets:
            r,c=data.positions[target];truth=float(raw[r,c])
            if not np.isfinite(truth):continue
            known=[int(i) for i in safe if i!=target and np.isfinite(raw[tuple(data.positions[i])])]
            if not known:raise ValueError(f'No input observations at time {t}')
            xy=data.positions[known].astype(np.float32);values=normalized[tuple(data.positions[known].T)]
            target_xy=data.positions[target].astype(np.float32)
            features.append(tabular_feature(xy,values,target_xy,data.coord[r,c],data.cov[t,:,r,c],k,data.shape))
            ys.append((truth-mean)/std)
            records.append({'station':int(target),'time_idx':int(data.original_times[t]),'true':truth,
                            'known_coords':xy,'known_values':values,'target_coord':target_xy})
    if not records:raise ValueError(f'No {mode} records')
    return np.asarray(features),np.asarray(ys),records

def baseline_predict(data,heldout,model,args,fold,root):
    start=time.perf_counter();mean,std=data.scaler(heldout)
    xt,yt,test=tabular_records(data,heldout,'test',args.k_neighbors)
    info={'fold_train_mean':mean,'fold_train_std':std,'distance_space':'grid row/column, same as original baseline',
          'parameter_count':None}
    if model in ['IDW','KNN','SpatialMean']:
        chosen=None
        if model!='SpatialMean':
            xv,yv,val=tabular_records(data,heldout,'val',args.k_neighbors)
            candidates=[1.,1.5,2.] if model=='IDW' else [3,4,6,8]
            scores={}
            for q in candidates:
                pp=np.array([interpolation(z['known_coords'],z['known_values'],z['target_coord'],model,q) for z in val])
                scores[str(q)]=float(np.mean(np.abs(pp-yv)))
            chosen=min(candidates,key=lambda q:scores[str(q)])
            info.update({'selected_parameter':chosen,'validation_MAE_by_parameter':scores})
        pred=np.array([interpolation(z['known_coords'],z['known_values'],z['target_coord'],model,chosen) for z in test])
    else:
        x,y,train=tabular_records(data,heldout,'train',args.k_neighbors)
        seed=int((args.seed+1009*fold+sum(map(ord,data.pollutant)))%(2**31-1))
        options=dict(n_estimators=args.trees,max_features=.7,min_samples_leaf=2,random_state=seed,n_jobs=args.ml_jobs)
        if model=='ExtraTrees':est=ExtraTreesRegressor(**options)
        elif model=='RandomForest':est=RandomForestRegressor(**options)
        elif model=='XGBoost':
            try:from xgboost import XGBRegressor
            except ImportError as exc:raise RuntimeError('XGBoost requested but not installed') from exc
            est=XGBRegressor(objective='reg:squarederror',n_estimators=400,max_depth=6,learning_rate=.03,
                             subsample=.8,colsample_bytree=.8,reg_lambda=1.,random_state=seed,n_jobs=args.ml_jobs)
        else:raise ValueError(model)
        est.fit(x,y);pred=est.predict(xt);joblib.dump(est,Path(root)/'baseline.joblib')
        info['estimator_parameters']=est.get_params();info['n_training_samples']=len(y)
    frame=pd.DataFrame([{k:z[k] for k in ('station','time_idx','true')} for z in test])
    frame['prediction']=pred*std+mean;validate_predictions(frame,data,heldout)
    info['train_and_predict_seconds']=time.perf_counter()-start
    return frame,info

def main():
    p=common_parser(__doc__)
    p.add_argument('--output',default='/etc/cj/boundary/d3_spatial_block_validation_v1')
    p.add_argument('--models',default='D3,D3A10,SpatialMean,IDW,KNN,ExtraTrees')
    p.add_argument('--blocks',type=int,default=5)
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--coordinates-csv',default='',help='可选：station,x,y；x/y应采用相同投影单位')
    p.add_argument('--k-neighbors',type=int,default=8)
    p.add_argument('--trees',type=int,default=300)
    p.add_argument('--ml-jobs',type=int,default=4)
    args=p.parse_args();models=csv_list(args.models);pollutants=csv_list(args.pollutants)
    if len(set(models))!=len(models) or set(models)-set((*MODELS,*BASELINES)):raise ValueError('Unknown/duplicate model')
    if args.trees<1 or args.k_neighbors<1:raise ValueError('Invalid tree/neighbor count')
    data={name:FieldData(args.data_root,name) for name in pollutants}
    first=next(iter(data.values()))
    for d in data.values():
        if not np.array_equal(d.positions,first.positions) or not np.allclose(d.coord,first.coord):
            raise ValueError('Pollutants must share station IDs, grid positions and coordinates')
        groups_from_reference(args,d)
    with output_lock(args.output):
        parts,split=make_blocks(first,args)
        print('Spatial block sizes:',[len(x) for x in parts],flush=True)
        write_json(Path(args.output)/'run_config.json',{'version':VERSION,'args':vars(args),'data':{k:v.identity for k,v in data.items()},
                    'split_signature':split['split_signature'],'sources':source_identity(),
                    'protocol':'retrain after excluding whole block from concentration inputs, train/validation labels and scaler',
                    'covariates':'precomputed covariates remain available over the full domain; D3 groups frozen'})
        if args.preflight:
            for d in data.values():
                for held in parts:d.scaler(held)
            print('Preflight passed. Review spatial_blocks.png before full training.');return
        engine=None;frames=[];status=[]
        selected=set(fold_range(args,args.blocks))
        for pollutant,d in data.items():
            for model in models:
                for fold,held in enumerate(parts):
                    root=Path(args.output)/model/pollutant/f'block_{fold:02d}';root.mkdir(parents=True,exist_ok=True)
                    fp=run_signature(args,d,model,args.seed,held,fold,extra={'split':split['split_signature'],
                           'k_neighbors':args.k_neighbors,'trees':args.trees,'ml_jobs':args.ml_jobs})
                    done=load_completed(root,fp,d,held)
                    if done is None and not args.summary_only and fold in selected:
                        print(f'\n{model} {pollutant} block {fold}: held out {held}',flush=True)
                        if model in MODELS:
                            if engine is None:engine=NeuralEngine(args,args.seed)
                            frame,info=engine.train_predict(d,held,fold,model,root,fp)
                        else:frame,info=baseline_predict(d,held,model,args,fold,root)
                        info.update({'heldout_stations':held,'training_stations':d.candidates(held).tolist()})
                        save_completed(root,fp,frame,info);done=(frame,info)
                        print(metrics(frame.true,frame.prediction),flush=True)
                    status.append({'model':model,'pollutant':pollutant,'seed':args.seed,'fold':fold,
                                   'heldout_station_count':len(held),'complete':done is not None})
                    if done is not None:frames.append(attach(done[0],model,pollutant,args.seed,fold))
                summarize(args.output,frames,status)
        result=summarize(args.output,frames,status)
        if len(result):print(result.to_string(index=False))
        print('DONE: read metrics/completion_status.csv; partial runs are explicitly marked.')

if __name__=='__main__':main()
