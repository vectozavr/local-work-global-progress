"""Additional strong control: native SQLite with STAT4, on separate database copies."""
import ctypes as C,json,random,statistics,shutil
import sqlite_profile as sp
import join_study as j
old=sp.lib;new=C.CDLL(str(j.ROOT/f'vendor/libsqlite_stat4.{sp.EXT}'))
for name in ['sqlite3_open_v2','sqlite3_close','sqlite3_prepare_v2','sqlite3_step','sqlite3_finalize','sqlite3_column_count','sqlite3_column_type','sqlite3_column_int64','sqlite3_column_double','sqlite3_column_text','sqlite3_stmt_scanstatus','sqlite3_stmt_status','sqlite3_errmsg','sqlite3_libversion']:
 getattr(new,name).argtypes=getattr(old,name).argtypes;getattr(new,name).restype=getattr(old,name).restype
sp.lib=new
new.sqlite3_db_config.argtypes=[sp.P,C.c_int];new.sqlite3_db_config.restype=C.c_int
class UnprofiledDB(sp.DB):
 def __init__(self,path):
  super().__init__(path);out=C.c_int(-1)
  assert new.sqlite3_db_config(self.db,1018,C.c_int(0),C.byref(out))==0 and out.value==0
sp.DB=UnprofiledDB
runs=[];stats=[]
for seed in [51031,68437,89759]:
 original=j.ROOT/f'join-data-{seed}.db';path=j.ROOT/f'stat4-copy-{seed}.db';shutil.copy2(original,path)
 db=sp.DB.__new__(sp.DB);db.db=sp.P();assert new.sqlite3_open_v2(str(path).encode(),C.byref(db.db),2,None)==0
 r=db.run('ANALYZE',False);stats.append({'seed':seed,'analyze_seconds':r['seconds'],'vm_steps':r['vm_steps']});db.close()
 catalog=[]
 for s in range(j.S):
  rng=random.Random(seed+s*18181);ah=set(rng.sample(range(j.G),j.G//2));ch=set(rng.sample(range(j.G),j.G//2))
  catalog.append({'ah':sorted(ah),'al':sorted(set(range(j.G))-ah),'ch':sorted(ch),'cl':sorted(set(range(j.G))-ch)})
 for regime,length in [('native_friendly',48),('native_friendly',192),('balanced',48)]:
  stream=j.jobs(seed,catalog,length,regime)
  for rep in range(3):
   db=sp.DB(path);seconds=0;steps=0;answers=[]
   for q in stream:
    r=db.run(' UNION ALL '.join(j.frag(s,*q[s],'native') for s in range(j.S)),False);seconds+=r['seconds'];steps+=r['vm_steps'];answers.append(r['rows'])
   db.close();runs.append({'seed':seed,'regime':regime,'length':length,'repetition':rep,'sql_seconds':seconds,'vm_steps':steps,'answers':answers})
 print('STAT4 complete',seed,flush=True);path.unlink()
(j.ROOT/'results'/'stat4-results.json').write_text(json.dumps({'calibration':stats,'runs':runs},indent=2))
summary={}
for regime,length in [('native_friendly',48),('native_friendly',192),('balanced',48)]:
 r=[r for r in runs if r['regime']==regime and r['length']==length]
 summary[f'{regime}_{length}']={'median_sql_seconds_existing_statistics':statistics.median(x['sql_seconds'] for x in r),'median_vm_steps':statistics.median(x['vm_steps'] for x in r)}
(j.ROOT/'results'/'stat4-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
