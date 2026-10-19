"""Fresh-seed confirmation after freezing gated feedback and bounded calibration."""
import json,random,statistics
import join_study as j
ROOT=j.ROOT;j.HEAVY_DEGREE=400
POLICIES=['native','fixed_left','fixed_right','sample128','sample2048','sample16384','full_join_histogram','active_join_count_cached','bounded_probe_cached','plan_identify_cached','online_local','gated_local','gated_central','frozen_after4']
rr=[];setup=[];rng=random.Random(199773)
for seed in [51031,68437,89759]:
 path,truth,catalog,edgesizes,seconds=j.build(seed);setup.append({'seed':seed,'seconds':seconds,'edgesizes':edgesizes,'bytes':path.stat().st_size})
 print('Built confirmation database',seed,edgesizes,flush=True)
 for regime,length in [('native_friendly',48),('native_friendly',192),('balanced',48)]:
  for rep in range(3):
   order=POLICIES.copy();rng.shuffle(order)
   for p in order:
    r=j.run(path,truth,catalog,edgesizes,seed,length,regime,p);r['repetition']=rep;rr.append(r)
   print('Completed',seed,regime,length,rep,flush=True)
 (ROOT/'results'/'confirmation-results.json').write_text(json.dumps({'results':rr,'setup':setup,'degree':j.HEAVY_DEGREE},indent=2))
summary={}
for regime,length in [('native_friendly',48),('native_friendly',192),('balanced',48)]:
 key=f'{regime}_{length}';summary[key]={}
 for p in POLICIES:
  r=[r for r in rr if r['regime']==regime and r['length']==length and r['policy']==p]
  summary[key][p]={'median_seconds':statistics.median(x['elapsed_seconds'] for x in r),'median_vm_steps':statistics.median(x['vm_steps'] for x in r),'median_calibration_seconds':statistics.median(x['calibration_seconds'] for x in r),'all_correct':all(x['successes']==length for x in r),'seed_medians':{str(seed):statistics.median(x['elapsed_seconds'] for x in r if x['seed']==seed) for seed in [51031,68437,89759]}}
(ROOT/'results'/'confirmation-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
