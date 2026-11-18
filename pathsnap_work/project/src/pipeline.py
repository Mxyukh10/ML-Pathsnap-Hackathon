from pathlib import Path
import argparse, json, time, warnings
import numpy as np, pandas as pd, joblib
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'data'/'Student folder'
FEATURES=['dist_m','heading_diff','speed_mps','dt_s','edge_len_m','edge_bearing_sin','edge_bearing_cos','gps_bearing_sin','gps_bearing_cos','endpoint_min_m']

def read_trip(folder):
    tid=folder.name
    track=pd.read_csv(folder/f'{tid}.track',sep=r'\s+',header=None,names=['lon','lat','ts'])
    nodes=pd.read_csv(folder/f'{tid}.nodes',sep=r'\s+',header=None,names=['lon','lat'])
    arcs=pd.read_csv(folder/f'{tid}.arcs',sep=r'\s+',header=None,names=['u','v']).astype(int)
    route=pd.read_csv(folder/f'{tid}.route',sep=r'\s+',header=None,names=['arc']).iloc[:,0].astype(int).to_numpy()
    return track,nodes,arcs,route

def project(lon,lat,lon0,lat0):
    x=(np.asarray(lon)-lon0)*111320*np.cos(np.deg2rad(lat0)); y=(np.asarray(lat)-lat0)*110540
    return np.column_stack([x,y])

def point_seg_dist(P,A,B):
    V=B-A; den=np.sum(V*V,axis=1); den=np.maximum(den,1e-9)
    W=P[:,None,:]-A[None,:,:]
    t=np.clip(np.sum(W*V[None,:,:],axis=2)/den[None,:],0,1)
    Q=A[None,:,:]+t[:,:,None]*V[None,:,:]
    return np.sqrt(np.sum((P[:,None,:]-Q)**2,axis=2))

def build_trip(folder,k=8):
    tr,nodes,arcs,route=read_trip(folder)
    tr=tr.replace([np.inf,-np.inf],np.nan).dropna().drop_duplicates('ts').sort_values('ts').iloc[::3].reset_index(drop=True)
    if len(tr)<2 or len(nodes)<2 or len(arcs)<1: return pd.DataFrame(), {'trip':folder.name,'observations':len(tr),'reason':'insufficient data'}
    lon0=float(tr.lon.mean()); lat0=float(tr.lat.mean())
    P=project(tr.lon,tr.lat,lon0,lat0); N=project(nodes.lon,nodes.lat,lon0,lat0)
    uv=arcs[['u','v']].to_numpy(); valid=(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<len(N))&(uv[:,1]<len(N))
    uv=uv[valid]; arc_ids=np.where(valid)[0]
    A=N[uv[:,0]]; B=N[uv[:,1]]; V=B-A
    lens=np.linalg.norm(V,axis=1); bearings=np.arctan2(V[:,1],V[:,0]); mids=(A+B)/2
    tree=cKDTree(mids); kk=min(k,len(mids)); _,near=tree.query(P,k=kk)
    if kk==1: near=near[:,None]
    # Exact point-to-segment distance for the candidate matrix (vectorised).
    W=P[:,None,:]-A[near]; VV=V[near]; den=np.maximum(np.sum(VV*VV,axis=2),1e-9)
    frac=np.clip(np.sum(W*VV,axis=2)/den,0,1); proj=A[near]+frac[:,:,None]*VV
    dist=np.sqrt(np.sum((P[:,None,:]-proj)**2,axis=2))
    # Approximate per-observation reference label by nearest segment among the supplied route arcs.
    rids=np.unique(route[(route>=0)&(route<len(arcs))]); rids=rids[np.isin(rids,arc_ids)]
    ref=np.full(len(P),-1,dtype=int)
    if len(rids):
        rpos=np.searchsorted(arc_ids,rids); rmid=mids[rpos]; rtree=cKDTree(rmid)
        _,rnear=rtree.query(P,k=min(6,len(rids)))
        if np.ndim(rnear)==1: rnear=rnear[:,None]
        for q in range(rnear.shape[1]):
            rp=rpos[rnear[:,q]]; ww=P-A[rp]; vv=V[rp]; dd=np.maximum(np.sum(vv*vv,axis=1),1e-9)
            tt=np.clip(np.sum(ww*vv,axis=1)/dd,0,1); qq=A[rp]+tt[:,None]*vv
            d=np.linalg.norm(P-qq,axis=1)
            if q==0: bestd=d; bestid=rids[rnear[:,q]]
            else:
                mask=d<bestd; bestd[mask]=d[mask]; bestid[mask]=rids[rnear[mask,q]]
        ref=bestid
    dt=np.diff(tr.ts.to_numpy(),prepend=tr.ts.iloc[0]); dt=np.maximum(dt,1e-3)
    delta=np.diff(P,axis=0,prepend=P[[0]]); gps_b=np.arctan2(delta[:,1],delta[:,0]); speed=np.linalg.norm(delta,axis=1)/dt
    diff=np.arctan2(np.sin(gps_b[:,None]-bearings[near]),np.cos(gps_b[:,None]-bearings[near]))
    endpoint=np.minimum(np.linalg.norm(P[:,None,:]-A[near],axis=2),np.linalg.norm(P[:,None,:]-B[near],axis=2))
    labels=(arc_ids[near]==ref[:,None]).astype(int)
    miss=int(np.sum((ref>=0)&(~np.any(labels==1,axis=1))))
    n,m=near.shape
    data={
      'dist_m':dist.ravel(),'heading_diff':np.abs(diff).ravel(),'speed_mps':np.repeat(speed,m),'dt_s':np.repeat(dt,m),
      'edge_len_m':lens[near].ravel(),'edge_bearing_sin':np.sin(bearings[near]).ravel(),'edge_bearing_cos':np.cos(bearings[near]).ravel(),
      'gps_bearing_sin':np.repeat(np.sin(gps_b),m),'gps_bearing_cos':np.repeat(np.cos(gps_b),m),'endpoint_min_m':endpoint.ravel(),
      'y':labels.ravel(),'trip_id':np.repeat(folder.name,n*m),'obs_idx':np.repeat(np.arange(n),m),'arc_id':arc_ids[near].ravel()}
    return pd.DataFrame(data), {'trip':folder.name,'observations':len(P),'candidate_rows':len(data['y']),'candidate_misses':miss,'candidate_miss_rate':miss/max(1,len(P)),'route_arcs':len(route),'nodes':len(nodes),'arcs':len(arcs)}

def load_partition(part,k=12):
    out=[]; stats=[]
    for folder in sorted((DATA/part).iterdir()):
        if not folder.is_dir(): continue
        try:
            df,s=build_trip(folder,k); stats.append(s)
            if len(df): out.append(df)
        except Exception as e: stats.append({'trip':folder.name,'error':str(e)})
    return (pd.concat(out,ignore_index=True) if out else pd.DataFrame()),stats

def metrics(model,X,y):
    pred=model.predict(X); p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    out={'accuracy':float(accuracy_score(y,pred)),'precision':float(p),'recall':float(r),'f1':float(f),'macro_f1':float(f1_score(y,pred,average='macro'))}
    if hasattr(model,'predict_proba'):
        try:
            probs=model.predict_proba(X)[:,1]
            # Candidate-level metrics are reported separately from pointwise top-1 accuracy.
            out['log_loss']=float(log_loss(y,model.predict_proba(X),labels=[0,1]))
        except: pass
    return out

def top1_accuracy(model, df):
    X=df[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
    prob=model.predict_proba(X)[:,1]
    z=df[['trip_id','obs_idx','y']].copy(); z['prob']=prob
    chosen=z.loc[z.groupby(['trip_id','obs_idx'])['prob'].idxmax()]
    return float(chosen['y'].mean()), int(len(chosen))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--k',type=int,default=8); ap.add_argument('--max-rows',type=int,default=180000); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    t0=time.time(); print('Loading training trips...',flush=True)
    train,stats=load_partition('train',args.k); print('Train candidate rows:',len(train),'positive rate:',train.y.mean(),flush=True)
    if len(train)==0 or train.y.nunique()<2: raise RuntimeError('Training data has fewer than two classes. Inspect route alignment/candidate generation.')
    # Deterministic cap for laptop-friendly training; preserve positive examples.
    if len(train)>args.max_rows:
        pos=train[train.y==1]; neg=train[train.y==0].sample(n=max(1,args.max_rows-len(pos)),random_state=42) if len(pos)<args.max_rows else pos.sample(args.max_rows,random_state=42)
        train=pd.concat([pos,neg]).sample(frac=1,random_state=42).reset_index(drop=True)
    X=train[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0); y=train.y.astype(int); groups=train.trip_id
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42); a,b=next(splitter.split(X,y,groups))
    models={
      'LogisticRegression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,class_weight='balanced',random_state=42)),
      'DecisionTree':DecisionTreeClassifier(max_depth=14,min_samples_leaf=4,class_weight='balanced',random_state=42),
      'RandomForest':RandomForestClassifier(n_estimators=100,max_depth=18,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=42),
      'ExtraTrees':ExtraTreesClassifier(n_estimators=100,max_depth=18,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42),
      'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=20,learning_rate=.08,l2_regularization=1.0,random_state=42)
    }
    results=[]; best_name=None; best_score=-1; best_model=None
    for name,m in models.items():
        st=time.time(); m.fit(X.iloc[a],y.iloc[a]); elapsed=time.time()-st
        met=metrics(m,X.iloc[b],y.iloc[b]); met['top1_accuracy'],met['gps_observations']=top1_accuracy(m,train.iloc[b]); met.update({'model':name,'fit_seconds':round(elapsed,2),'train_rows':len(a),'validation_rows':len(b),'validation_trips':int(groups.iloc[b].nunique())})
        results.append(met); print(json.dumps(met),flush=True)
        if met['f1']>best_score: best_score=met['f1']; best_name=name; best_model=m
    # Refit selected model on all sampled training rows and save artifacts.
    best_model.fit(X,y); joblib.dump({'model':best_model,'features':FEATURES,'candidate_k':args.k,'model_name':best_name},ROOT/'models'/'pathsnap_model.joblib')
    pd.DataFrame(results).to_csv(ROOT/'results'/'model_comparison.csv',index=False)
    pd.DataFrame(stats).to_csv(ROOT/'results'/'train_trip_eda.csv',index=False)
    # Evaluate on student_test only, never evaluator_test.
    print('Building student-test candidates...',flush=True)
    test,tstats=load_partition('student_test',args.k)
    if len(test):
        Xt=test[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0); yt=test.y.astype(int)
        tm=metrics(best_model,Xt,yt); tm['top1_accuracy'],tm['gps_observations']=top1_accuracy(best_model,test); tm.update({'selected_model':best_name,'rows':len(test),'trips':int(test.trip_id.nunique())})
        (ROOT/'results'/'student_test_metrics.json').write_text(json.dumps(tm,indent=2))
        pd.DataFrame(tstats).to_csv(ROOT/'results'/'student_test_trip_eda.csv',index=False)
        print('STUDENT_TEST',json.dumps(tm),flush=True)
    summary={'selected_model':best_name,'validation_f1':best_score,'training_rows_used':len(train),'candidate_k':args.k,'elapsed_seconds':round(time.time()-t0,2),'note':'Reference labels are approximated by nearest spatial segment among unique arc IDs in the supplied route; there are no explicit per-observation labels. This is a weak alignment assumption and should be validated.'}
    (ROOT/'results'/'run_summary.json').write_text(json.dumps(summary,indent=2)); print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__': main()
