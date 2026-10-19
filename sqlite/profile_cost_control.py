"""Fair instrumentation control: non-learning policies disable scan-status counters."""
import ctypes as C,json,random,statistics
import sqlite_profile as sp
import join_study as j
sp.lib.sqlite3_db_config.argtypes=[sp.P,C.c_int];sp.lib.sqlite3_db_config.restype=C.c_int
ENABLED=True
class ConfiguredDB(sp.DB):
 def __init__(self,path):
  super().__init__(path);out=C.c_int()
  assert sp.lib.sqlite3_db_config(self.db,1018,C.c_int(int(ENABLED)),C.byref(out))==0
  assert out.value==int(ENABLED)
j.DB=ConfiguredDB
original=json.loads((j.ROOT/'results'/'stat4-results.json').read_text())['runs']
policies=['native','fixed_left','sample128','sample2048','full_join_histogram','active_join_count_cached','bounded_probe_cached','plan_identify_cached','online_local','gated_local','gated_central','frozen_after4']
rr=[];rng=random.Random(707091)
for seed in [51031,68437,89759]:
 catalog=[]
 for s in range(j.S):
  gen=random.Random(seed+s*18181);ah=set(gen.sample(range(j.G),j.G//2));ch=set(gen.sample(range(j.G),j.G//2));catalog.append({'ah':sorted(ah),'al':sorted(set(range(j.G))-ah),'ch':sorted(ch),'cl':sorted(set(range(j.G))-ch)})
 truth=[{} for _ in range(j.S)]
 for r in original:
  if r['seed']!=seed or r['repetition']!=0:continue
  for q,answer in zip(j.jobs(seed,catalog,r['length'],r['regime']),r['answers']):
   for s in range(j.S):truth[s][tuple(q[s])]=answer[s][1:]
 for regime,length in [('native_friendly',48),('native_friendly',192),('balanced',48)]:
  for rep in range(3):
   order=policies.copy();rng.shuffle(order)
   for policy in order:
    ENABLED=policy in ['online_local','gated_local','gated_central','frozen_after4']
    r=j.run(j.ROOT/f'join-data-{seed}.db',truth,catalog,[1642496]*3,seed,length,regime,policy);r['repetition']=rep;r['scanstatus_enabled']=ENABLED;rr.append(r)
 print('Instrument-cost control',seed,flush=True)
summary={}
for regime,length in [('native_friendly',48),('native_friendly',192),('balanced',48)]:
 key=f'{regime}_{length}';summary[key]={}
 for policy in policies:
  r=[r for r in rr if r['regime']==regime and r['length']==length and r['policy']==policy]
  summary[key][policy]={'median_seconds':statistics.median(x['elapsed_seconds'] for x in r),'median_vm_steps':statistics.median(x['vm_steps'] for x in r),'all_correct':all(x['successes']==length for x in r),'seed_medians':{str(seed):statistics.median(x['elapsed_seconds'] for x in r if x['seed']==seed) for seed in [51031,68437,89759]}}
(j.ROOT/'results'/'instrumentation-results.json').write_text(json.dumps({'results':rr,'note':'Scan status disabled for controls that do not use it. Expected answers replayed from STAT4 control; initial full study used independent Python generator reference.'},indent=2));(j.ROOT/'results'/'instrumentation-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
