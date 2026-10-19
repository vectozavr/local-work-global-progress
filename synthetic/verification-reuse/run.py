"""Learn a persistent correction from scalar verifier residuals.
Controlled extension: reuse proposal measurements when one unknown coefficient remains. NumPy only.
"""
from pathlib import Path
import json, math,itertools
import numpy as np
OUT=Path(__file__).resolve().parent
DATA=json.loads((OUT.parent/'known-correction/results.json').read_text())
target=np.array(DATA['target']);weights=np.array(DATA['weights']);private=np.array(DATA['private_target'])
N,D=target.shape;W=DATA['coupling'];mu=DATA['mu'];rates=np.array(DATA['rates']);REPS=1000;SEED=20260926
local_a=weights*(1-2*private);M=2*weights
degree=np.full(N,2);degree[[0,-1]]=1
incident=W*degree[:,None]
mu_certificate=float(np.min((weights-incident)/(weights+incident)))
assert mu_certificate>0

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
    reused=np.zeros(REPS,dtype=int)
    fstar=float(oracle(target));start=float(oracle(x[0])-fstar);trace=[]
    q_b=1-(1-rates/D)**H;q=float(q_b.min());alpha=mu*q/2;beta=float((1+q_b).max()/2)
    gamma=p/D if p else 0;weight=2*beta/gamma if gamma else 0;r=min(alpha,gamma/2) if gamma else 0
    V0=start+weight*float(M.sum())
    for epoch in range(1,257):
        before=oracle(x);active=(finished==0)
        if not active.any():break
        boundary=known_boundary(x)
        if mode=='refresh' or (mode=='identify_once' and epoch==1):
            # Calibration controls: recover all coefficients each epoch or once at initialization.
            for b in range(N):
                for j in range(D):
                    y=x.copy();y[:,b,j]=1-y[:,b,j];s=1-2*x[:,b,j]
                    residual=oracle(y)-before-s*(local_a[b,j]+boundary[:,b,j])
                    hat[active,b,j]=residual[active]/s[active];known[active,b,j]=True
                    probes+=active
        elif mode in ('learned','learned_reuse'):
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
            candidate_value=oracle(candidate)
            gains[:,b]=np.maximum(0,before-candidate_value);verifications+=changed
            if mode=='learned_reuse':
                displacement=y[:,b]-x[:,b]
                unknown_support=(displacement!=0)&(~known[:,b,:])
                reusable=changed&(unknown_support.sum(1)==1)
                j=unknown_support.argmax(1); rr=np.arange(REPS)
                # All measurements are taken at the original base. Negative-gain
                # proposals are informative too; reuse never changes this epoch's proposals.
                residual=candidate_value-before-(displacement*(local_a[b]+boundary[:,b])).sum(1)
                residual-=(displacement*np.where(known[:,b],hat[:,b],0)).sum(1)
                rsel=rr[reusable];jsel=j[reusable]
                hat[rsel,b,jsel]=residual[reusable]/displacement[rsel,jsel]
                known[rsel,b,jsel]=True;reused+=reusable
        sums=np.stack([gains[:,::2].sum(1),gains[:,1::2].sum(1)],1);color=sums.argmax(1)
        for b in range(N):
            take=active&(color==b%2)&(gains[:,b]>1e-9);x[take,b]=y[take,b]
        gap=oracle(x)-fstar;assert (gap>=-1e-9).all();assert np.max(oracle(x)-before)<1e-9
        # Observable stopping: exact affine surrogate minima plus certified coefficient radii.
        # Recompute known context at the new base; no target or optimum enters this test.
        a_current=local_a+hat+known_boundary(x)
        surrogate_improvement=(a_current*x-np.minimum(a_current,0)).sum(axis=(1,2))
        remaining_radius=((~known)*M).sum(axis=(1,2))
        certificate=(surrogate_improvement+remaining_radius)/mu_certificate
        assert np.all(gap<=certificate+1e-9)
        newly=active&(certificate<1e-9);finished[newly]=epoch
        # Offline diagnostics may access hidden truth, but do not affect the policy.
        theta_true=weights*(1-2*target)-local_a
        error=np.abs(hat-theta_true);assert np.max(error[known],initial=0)<1e-10
        R=((~known)*M).sum(axis=(1,2));assert np.all(error.sum((1,2))<=R+1e-9)
        trace.append(dict(epoch=epoch,mean_gap=float(gap.mean()),mean_radius=float(R.mean()),
                          mean_certificate=float(certificate.mean()),stopped_trajectories=int((finished>0).sum())))
    solved=(finished>0)
    out=dict(mode=mode,H=H,p=p,repetitions=REPS,solved=int(solved.sum()),mean_residual_gap=float((oracle(x)-fstar).mean()),
             mean_reused_coefficients=float(reused.mean()),mean_probes=float(probes.mean()),max_probes=int(probes.max()),mean_verifications=float(verifications.mean()),trace=trace)
    if solved.all():
        # C=20 per synchronization; unit cost per verifier call, serially charged; c=1 per parallel local slot.
        cost=finished*(H+20)+probes+verifications+1
        out.update(mean_epochs=float(finished.mean()),se_epochs=float(finished.std(ddof=1)/math.sqrt(REPS)),
                   mean_queries=float((1+probes+verifications).mean()),mean_modeled_cost=float(cost.mean()),
                   se_modeled_cost=float(cost.std(ddof=1)/math.sqrt(REPS)))
    if mode in ('learned','learned_reuse'):assert np.max(probes+reused)<=N*D
    return out
rows=[]
for H in [8,32,64]:
    for mode in ['learned','learned_reuse','identify_once']:
        rows.append(experiment(H,1.,mode))
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
result=dict(seed=SEED,task_source='../known-correction/results.json',mu=mu,mu_certificate=mu_certificate,
            joint_coefficient_checks=checks,projection_identity_max_error=projection_error,rows=rows,
            accounting='C=20 includes communication/context/integration; c=1 per parallel local slot; every scalar verifier query costs an additional unit. Initial value charged once. Probe and candidate calls reuse the base value. Offline audit calls excluded. All costs modeled.',
            scope='Same fixed binary task; p=1. Reuse only when a verified displacement contains exactly one still-unknown coefficient after removing known contributions. No additional oracle call for reuse. Projection/arithmetic cost is included in modeled overhead, not measured. No LLM calls. Includes identify-once baseline. Trajectories stop using exact surrogate minima, known coefficient bounds, and the analytical dominance constant computed from known weights; the optimum is used only in offline diagnostics. Stopped traces are not a joint-contraction test. Certificate arithmetic is included in modeled overhead, not timed.')
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in result.items() if k!='rows'},'rows':[{k:v for k,v in x.items() if k!='trace'} for x in rows]},indent=2))
