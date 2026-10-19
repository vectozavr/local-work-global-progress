"""Independent finite-state checks of the stopping certificate, the dominance bound and the geometry characterization; NumPy only."""
import itertools
import numpy as np
rng=np.random.default_rng(2026092601)
n,d=3,3
blockstates=np.array(list(itertools.product([0,1],repeat=d)))
for _ in range(2000):
    target=rng.integers(0,2,(n,d));private=rng.integers(0,2,(n,d))
    weights=rng.uniform(1,2,(n,d));w=.2
    incident=w*np.array([1,2,1])[:,None]
    mu=np.min((weights-incident)/(weights+incident))
    x=rng.integers(0,2,(n,d))
    def F(z): return ((z!=target)*weights).sum()+w*(z[:-1]!=z[1:]).sum()
    gap=F(x)-F(target);D=[]
    known=rng.random((n,d))<.5
    local=weights*(1-2*private);theta=weights*(1-2*target)-local
    hat=np.where(known,theta,0);radius=np.where(known,0,2*weights).sum(1)
    boundary=np.zeros((n,d));boundary[:-1]+=w*(1-2*x[1:]);boundary[1:]+=w*(1-2*x[:-1])
    Ahat=local+hat+boundary;lower=[]
    for b in range(n):
        changes=[];surrogates=[]
        for u in blockstates:
            z=x.copy();z[b]=u
            f=F(z)-F(x);s=Ahat[b]@(u-x[b])
            assert abs(f-s)<=radius[b]+1e-10
            changes.append(f);surrogates.append(s)
        D.append(-min(changes));lower.append(min(surrogates))
        analytic=(np.minimum(Ahat[b],0)-Ahat[b]*x[b]).sum()
        assert abs(analytic-lower[-1])<1e-10
    assert sum(D)>=mu*gap-1e-10
    assert gap<=(sum(radius)-sum(lower))/mu+1e-10
print('2,000 independently enumerated block problems: dominance, surrogate minima, and stopping certificate passed.')
for _ in range(2000):
    theta=rng.normal(size=7);hat=rng.normal(size=7)
    for j in range(10):
        a=rng.normal(size=7);y=a@theta
        previous=np.linalg.norm(hat-theta)
        hat=hat+(y-a@hat)/(a@a)*a
        assert np.linalg.norm(hat-theta)<=previous+1e-12
print('20,000 extra exact projections: parameter-error nonincrease passed.')
states=list(itertools.product([0,1],repeat=3))
for _ in range(2000):
    values=dict(zip(states,rng.integers(-10,11,len(states))))
    best=min(values.values());ratios=[];trap=False
    for x,value in values.items():
        if value==best:continue
        gains=[]
        for b in range(3):
            z=list(x);z[b]=1-z[b]
            gains.append(max(0,value-values[tuple(z)]))
        trap=trap or sum(gains)==0
        ratios.append(sum(gains)/(value-best))
    assert (not trap)==(not ratios or min(ratios)>0)
print('2,000 arbitrary finite objectives: block-geometry characterization passed.')
