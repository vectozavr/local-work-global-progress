"""Finite-state correction and communication study. No LLM calls.
Run with Python and NumPy. Fixed seed; all time quantities are modeled.
"""
from pathlib import Path
import itertools,json,math
import numpy as np
OUT=Path(__file__).resolve().parent
SEED=20260925; N=6; D=8; REPS=2000; W=0.2
rng=np.random.default_rng(SEED)
target=rng.integers(0,2,(N,D)); weights=rng.uniform(1.0,1.5,(N,D))
private=target.copy();private[::2]=1-private[::2]
rates=np.array([.5,.6,.7,.8,.9,1.0])

def objective(x):
    return ((x!=target)*weights).sum(axis=(-2,-1))+W*(x[...,:-1,:]!=x[...,1:,:]).sum(axis=(-2,-1))

# Exact global conditional coefficients; constant terms are unnecessary.
def coefficients(x):
    a=np.broadcast_to(weights*(1-2*target),x.shape).copy()
    a[...,:-1,:]+=W*(1-2*x[...,1:,:])
    a[...,1:,:]+=W*(1-2*x[...,:-1,:])
    return a
local_a=weights*(1-2*private)
# Dominance gives the unique target optimum; independently verify by feature enumeration.
fstar=float(objective(target));mu=1.;audit_states=0
for j in range(D):
    vals={}
    for bits in itertools.product([0,1],repeat=N):
        x=np.array(bits);vals[bits]=float(((x!=target[:,j])*weights[:,j]).sum()+W*(x[:-1]!=x[1:]).sum())
    opt=min(vals.values());assert abs(vals[tuple(target[:,j])]-opt)<1e-10
    for bits,value in vals.items():
        gains=[]
        for b in range(N):
            y=list(bits);y[b]=1-y[b];gains.append(max(0,value-vals[tuple(y)]))
        if value>opt+1e-10:mu=min(mu,sum(gains)/(value-opt))
        audit_states+=1
# Separability across features makes the smallest feature ratio the full-block constant.
assert mu>0
x0=np.broadcast_to(1-target,(REPS,N,D)).copy();gap0=float(objective(x0[0])-fstar)
# Audit correction identity on sampled bases and arbitrary local values.
err=0.
for _ in range(200):
    x=rng.integers(0,2,(N,D));a=coefficients(x);b=int(rng.integers(N));u=rng.integers(0,2,D)
    y=x.copy();y[b]=u
    c=a[b]-local_a[b]
    corrected_change=float(np.dot(local_a[b]+c,u-x[b]))
    err=max(err,abs(corrected_change-(objective(y)-objective(x))))
assert err<1e-10

def trial(H,corrected=True):
    rng=np.random.default_rng(SEED+H)
    x=x0.copy();finished=np.zeros(REPS,dtype=int);history=[]
    for epoch in range(1,513 if corrected else 129):
        before=objective(x);active=before>fstar+1e-9
        if not active.any():break
        a=coefficients(x) if corrected else np.broadcast_to(local_a,x.shape)
        y=x.copy()
        for _ in range(H):
            index=rng.integers(D,size=(REPS,N));success=rng.random((REPS,N))<rates
            r,b=np.indices((REPS,N));choice=(a[r,b,index]<0).astype(int)
            old=y[r,b,index];y[r,b,index]=np.where(success,choice,old)
        gains=np.zeros((REPS,N))
        for b in range(N):
            candidate=x.copy();candidate[:,b]=y[:,b]
            gains[:,b]=np.maximum(0,before-objective(candidate))
        sums=np.stack([gains[:,::2].sum(1),gains[:,1::2].sum(1)],axis=1)
        color=np.argmax(sums,axis=1)
        for b in range(N):
            use=active&(color==b%2)&(gains[:,b]>1e-10)
            x[use,b]=y[use,b]
        after=objective(x)
        assert np.max(after-before)<1e-9
        assert np.max(np.abs(before-after-sums.max(1)))<1e-9
        newly=(finished==0)&(after<=fstar+1e-9);finished[newly]=epoch
        history.append(float(np.mean(after-fstar)))
    result=dict(H=H,corrected=corrected,solved=int((finished>0).sum()),repetitions=REPS,
                residual_gap_mean=float(np.mean(objective(x)-fstar)),mean_gap_by_epoch=history)
    if corrected:
        assert (finished>0).all()
        result.update(mean_epochs=float(finished.mean()),se_epochs=float(finished.std(ddof=1)/math.sqrt(REPS)),
                      mean_local_slots=float(H*finished.mean()),
                      cost_by_overhead={str(C):float((H+C)*finished.mean()) for C in [0,5,20,100]})
    return result
rows=[trial(h) for h in [1,2,4,8,16,32,64,128]];uncorrected=trial(8,False)
rho=float(rates.min()/D);lam=-math.log1p(-rho)
bounds=[]
for C in [0,5,20,100]:
    candidates=[]
    for h in range(1,257):
        q=-math.expm1(-lam*h);epochs=math.ceil(2/(mu*q)*math.log(100))
        candidates.append((h,(h+C)/q,epochs,(h+C)*epochs))
    best=min(candidates,key=lambda z:z[1]);actual=min(candidates,key=lambda z:z[3])
    bounds.append(dict(overhead=C,optimal_relaxed_integer_H=best[0],relaxed_factor=best[1],
                       optimal_ceiled_bound_H=actual[0],certified_modeled_time=actual[3]))
# Anchored approximate-correction inequality: exact surrogate minimization q=1.
approx_checks=0
for _ in range(10000):
    f=rng.uniform(-2,2,5);e=rng.uniform(-.2,.2,5);e[0]=0
    chosen=int(np.argmin(f+e));Dstar=f[0]-f.min();delta=max(abs(e))
    verified=max(0,f[0]-f[chosen])
    assert verified+1e-10>=Dstar-2*delta
    approx_checks+=1
result=dict(seed=SEED,n=N,d=D,repetitions=REPS,coupling=W,target=target.tolist(),
            private_target=private.tolist(),weights=weights.tolist(),rates=rates.tolist(),
            optimum=fstar,initial_gap=gap0,mu=mu,rho_min=rho,exact_feature_states=audit_states,
            correction_identity_max_error=err,approximation_checks=approx_checks,
            corrected=rows,uncorrected=uncorrected,certificate_optimizers=bounds,
            scope='Synthetic randomized coordinate-search agents. Modeled parallel local slot cost 1; all shared setup/verification/communication cost C per epoch. No LLM calls or hardware-speedup claim.')
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
summary={k:v for k,v in result.items() if k not in ['corrected','uncorrected','weights','target','private_target']}
summary['corrected']=[{k:v for k,v in row.items() if k!='mean_gap_by_epoch'} for row in rows]
summary['uncorrected']={k:v for k,v in uncorrected.items() if k!='mean_gap_by_epoch'}
print(json.dumps(summary,indent=2))
