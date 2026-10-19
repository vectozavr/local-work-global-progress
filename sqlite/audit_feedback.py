"""Replay raw feedback and check learned fan-outs against exact offline joins."""
import json,random,statistics
import join_study as j
import sqlite_profile as sp
original=json.loads((j.ROOT/'results'/'stat4-results.json').read_text())['runs']
recorded=json.loads((j.ROOT/'results'/'instrumentation-results.json').read_text())['results']
all_audits=[];central=[]
class AuditDB(sp.DB):
 events=[]
 def run(self,sql,profile=True):
  r=super().run(sql,profile)
  if profile and 'EXPLAIN' not in sql:self.events.append({'sql':sql,**r})
  return r
for seed in [51031,68437,89759]:
 catalog=[]
 for s in range(j.S):
  rng=random.Random(seed+s*18181);ah=set(rng.sample(range(j.G),j.G//2));ch=set(rng.sample(range(j.G),j.G//2));catalog.append({'ah':sorted(ah),'al':sorted(set(range(j.G))-ah),'ch':sorted(ch),'cl':sorted(set(range(j.G))-ch)})
 truth=[{} for _ in range(j.S)]
 for row in original:
  if row['seed']==seed and row['repetition']==0:
   for q,answers in zip(j.jobs(seed,catalog,row['length'],row['regime']),row['answers']):
    for s in range(j.S):truth[s][tuple(q[s])]=answers[s][1:]
 for regime,length in [('native_friendly',48),('native_friendly',192),('balanced',48)]:
  for rep in range(3):
   j.DB=sp.DB
   r=j.run(j.ROOT/f'join-data-{seed}.db',truth,catalog,[1642496]*3,seed,length,regime,'gated_factorized_central');r['repetition']=rep;central.append(r)
   target=next(x for x in recorded if x['seed']==seed and x['regime']==regime and x['length']==length and x['policy']=='gated_local' and x['repetition']==rep)
   assert json.loads(json.dumps(r['trace']))==target['trace'], 'Decision trace mismatch'
   assert r['vm_steps']==target['vm_steps'], 'VM work mismatch'
  if length!=48:continue
  AuditDB.events=[];j.DB=AuditDB
  r=j.run(j.ROOT/f'join-data-{seed}.db',truth,catalog,[1642496]*3,seed,length,regime,'gated_local')
  auditdb=sp.DB(j.ROOT/f'join-data-{seed}.db');checked=0;unique=set();gated=0
  for row,event in zip(r['trace'],AuditDB.events):
   assert event['rows']==[[s,*truth[s].get(tuple(row['query'][s]),[0,0])] for s in range(j.S)]
   if event['scans']:gated+=1
   for scan in event['scans']:
    name=scan['name']
    if not name.startswith('b'):continue
    s=int(name[1]);side='left' if name.endswith('_aid') else 'right';tab,col=('a','aid') if side=='left' else ('c','cid');g=row['query'][s][0 if side=='left' else 1]
    count=auditdb.run(f'SELECT COUNT(*) FROM {tab}{s} AS d JOIN b{s} AS b ON b.{col}=d.id WHERE d.grp={g}',False)['rows'][0][0]
    assert count==scan['visits'];checked+=1;unique.add((s,side,g))
  auditdb.close()
  all_audits.append({'seed':seed,'regime':regime,'reports':length,'profiled_reports':gated,'verified_measurements':checked,'distinct_observed_coordinates':len(unique),'events':AuditDB.events.copy()})
(j.ROOT/'results'/'feedback-audit.json').write_text(json.dumps({'audits':all_audits,'central_factorized_runs':central},indent=2))
print(json.dumps([{k:v for k,v in a.items() if k!='events'} for a in all_audits],indent=2))
print('Central factorized actions and VM work exactly match local agents on all 27 replays.')
