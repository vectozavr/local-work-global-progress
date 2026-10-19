"""Learn a persistent correction from scalar verifier residuals.
No analytic true correction is exposed to learned-policy code. NumPy only.
"""
from pathlib import Path
import json, math,itertools
import numpy as np
OUT=Path(__file__).resolve().parent
DATA=json.loads((OUT.parent/'known-correction/results.json').read_text())
target=np.array(DATA['target']);weights=np.array(DATA['weights']);private=np.array(DATA['private_target'])
N,D=target.shape;W=DATA['coupling'];mu=DATA['mu'];rates=np.array(DATA['rates']);REPS=1000;SEED=20260926
local_a=weights*(1-2*private);M=2*weights

def coupling(x):return W*(x[...,:-1,:]!=x[...,1:,:]).sum(axis=(-2,-1))
def oracle(x):return ((x!=target)*weights).sum(axis=(-2,-1))+coupling(x)
def known_boundary(x):
    a=np.zeros(x.shape,dtype=float)
    a[...,:-1,:]+=W*(1-2*x[...,1:,:]);a[...,1:,:]+=W*(1-2*x[...,:-1,:])
    return a

def experiment(H,p,mode):
    rng=np.random.default_rng(SEED+H+int(100*p));x=np.broadcast_to(1-target,(REPS,N,D)).copy()
    hat=np.zeros_like(x,dtype=float);known=np.zeros_like(x,dtype=bool)
    finished=np.zeros(REPS,dtype=int);probes=np.zeros(REPS,dtype=int);verifications=np.zeros(REPS,dtype=int)
    fstar=float(oracle(target));start=float(oracle(x[0])-fstar);trace=[]
    q_b=1-(1-rates/D)**H;q=float(q_b.min());alpha=mu*q/2;beta=float((1+q_b).max()/2)
    gamma=p/D if p else 0;weight=2*beta/gamma if gamma else 0;r=min(alpha,gamma/2) if gamma else 0
    V0=start+weight*float(M.sum())
    for epoch in range(1,257):
        before=oracle(x);active=(finished==0)
        if not active.any():break
        boundary=known_boundary(x)
        if mode=='refresh' or (mode=='identify_once' and epoch==1):
            # Strong baseline: recovers every coefficient from scalar probes each epoch.
            for b in range(N):
                for j in range(D):
                    y=x.copy();y[:,b,j]=1-y[:,b,j];s=1-2*x[:,b,j]
                    residual=oracle(y)-before-s*(local_a[b,j]+boundary[:,b,j])
                    hat[active,b,j]=residual[active]/s[active];known[active,b,j]=True
                    probes+=active
        elif mode=='learned':
            for b in range(N):
                unknown=~known[:,b,:]
                take=active & unknown.any(1) & (rng.random(REPS)<p)
                # Uniform among unobserved coordinates; no repeated queries after identification.
                random=rng.random((REPS,D));random[~unknown]=-1
                j=random.argmax(1);rr=np.arange(REPS)
                y=x.copy();y[rr,b,j]=1-y[rr,b,j];s=1-2*x[rr,b,j]
                residual=oracle(y)-before-s*(local_a[b,j]+boundary[rr,b,j])
                innovation=residual-hat[rr,b,j]*s
                hat[rr[take],b,j[take]]+=innovation[take]/s[take]
                known[rr[take],b,j[take]]=True;probes+=take
        # The policy now uses only local data, observed residuals, and known coupling/context.
        a=local_a+hat+boundary;y=x.copy()
        for _ in range(H):
            j=rng.integers(D,size=(REPS,N));rr,bb=np.indices((REPS,N));success=rng.random((REPS,N))<rates
            y[rr,bb,j]=np.where(success,(a[rr,bb,j]<0).astype(int),y[rr,bb,j])
        gains=np.zeros((REPS,N))
        for b in range(N):
            candidate=x.copy();candidate[:,b]=y[:,b]
            changed=active&(candidate[:,b]!=x[:,b]).any(1)
            gains[:,b]=np.maximum(0,before-oracle(candidate));verifications+=changed
        sums=np.stack([gains[:,::2].sum(1),gains[:,1::2].sum(1)],1);color=sums.argmax(1)
        for b in range(N):
            take=active&(color==b%2)&(gains[:,b]>1e-9);x[take,b]=y[take,b]
        gap=oracle(x)-fstar;assert (gap>=-1e-9).all();assert np.max(oracle(x)-before)<1e-9
        newly=active&(gap<1e-9);finished[newly]=epoch
        # Offline diagnostics may access hidden truth, but do not affect the policy.
        theta_true=weights*(1-2*target)-local_a
        error=np.abs(hat-theta_true);assert np.max(error[known],initial=0)<1e-10
        R=((~known)*M).sum(axis=(1,2));assert np.all(error.sum((1,2))<=R+1e-9)
        trace.append(dict(epoch=epoch,mean_gap=float(gap.mean()),mean_radius=float(R.mean()),
                          stopped_trajectories=int((finished>0).sum())))
    solved=(finished>0)
    out=dict(mode=mode,H=H,p=p,repetitions=REPS,solved=int(solved.sum()),mean_residual_gap=float((oracle(x)-fstar).mean()),
             mean_probes=float(probes.mean()),max_probes=int(probes.max()),mean_verifications=float(verifications.mean()),trace=trace)
    if solved.all():
        # C=20 per synchronization; unit cost per verifier call, serially charged; c=1 per parallel local slot.
        cost=finished*(H+20)+probes+verifications+1
        out.update(mean_epochs=float(finished.mean()),se_epochs=float(finished.std(ddof=1)/math.sqrt(REPS)),
                   mean_queries=float((1+probes+verifications).mean()),mean_modeled_cost=float(cost.mean()),
                   se_modeled_cost=float(cost.std(ddof=1)/math.sqrt(REPS)))
    if mode=='learned':assert probes.max()<=N*D
    return out
rows=[]
for H in [8,32,64]:
    for p in [1.,.5,.25]:rows.append(experiment(H,p,'learned'))
    rows.append(experiment(H,1.,'refresh'))
    rows.append(experiment(H,1.,'identify_once'))
rows.append(experiment(32,0.,'uncorrected'))
# Exact one-step joint-coefficient audit, independent of trajectories.
checks=0
for alpha in [.001,.02,.2,.8]:
 for gamma in [.001,.02,.2,1.]:
  beta=.7;omega=2*beta/gamma;r=min(alpha,gamma/2)
  assert (beta+omega)*(1-gamma)<=omega*(1-r)+1e-12
  assert 1-alpha<=1-r+1e-12;checks+=1
# Random projection identity and excitation contraction for coordinate measurements.
rng=np.random.default_rng(SEED);projection_error=0.
for _ in range(500):
 e=rng.normal(size=12);a=rng.normal(size=12);new=e-a*(a@e)/(a@a)
 projection_error=max(projection_error,abs(new@new-(e@e-(a@e)**2/(a@a))))
assert projection_error<1e-10
result=dict(seed=SEED,task_source='../known-correction/results.json',mu=mu,
            joint_coefficient_checks=checks,projection_identity_max_error=projection_error,rows=rows,
            accounting='C=20 includes communication/context/integration; c=1 per parallel local slot; every scalar verifier query costs an additional unit. Initial value charged once. Probe and candidate calls reuse the base value. Offline audit calls excluded. All costs modeled.',
            scope='Same fixed binary task, hidden static unary correction, known pair coupling, exact scalar verifier, no LLM calls. Includes identify-once baseline. Trajectories are stopped at the offline-known optimum; radius traces for this stopped policy are not a joint-contraction test.')
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in result.items() if k!='rows'},'rows':[{k:v for k,v in x.items() if k!='trace'} for x in rows]},indent=2))
