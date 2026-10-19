"""Join-degree feedback: independent sources, real three-table joins, static data."""
import json,random,sqlite3,time,statistics,itertools,sys
from collections import Counter,defaultdict
from pathlib import Path
from sqlite_profile import DB,lib
ROOT=Path(__file__).resolve().parent
(ROOT/'results').mkdir(exist_ok=True)   # outputs of confirm.py, stat4_control.py, profile_cost_control.py, audit_feedback.py
G=4096;NODE_PER_GROUP=2;HEAVY_DEGREE=100;S=3
POLICIES=['native','fixed_left','fixed_right','sample128','sample2048','sample16384','full_join_histogram','active_join_count_cached','bounded_probe_cached','plan_identify_cached','online_local','online_central','gated_local','gated_central','frozen_after4']

def build(seed):
 path=ROOT/f'join-data-{seed}.db'
 if path.exists():path.unlink()
 start=time.perf_counter();c=sqlite3.connect(path);c.execute('PRAGMA journal_mode=OFF');c.execute('PRAGMA synchronous=OFF')
 truth=[];catalog=[];edgesizes=[];generators=[]
 for source in range(S):
  rng=random.Random(seed+source*18181);ah=set(rng.sample(range(G),G//2));ch=set(rng.sample(range(G),G//2))
  catalog.append({'ah':sorted(ah),'al':sorted(set(range(G))-ah),'ch':sorted(ch),'cl':sorted(set(range(G))-ch)});generators.append(rng)
 needed=[set() for _ in range(S)]
 for regime in ['native_friendly','sparse_outliers','balanced']:
  for q in jobs(seed,catalog,192,regime):
   for source,pair in enumerate(q):needed[source].add(tuple(pair))
 for s in range(S):
  rng=generators[s];ah=catalog[s]['ah'];al=catalog[s]['al'];ch=catalog[s]['ch'];cl=catalog[s]['cl']
  c.executescript(f'CREATE TABLE a{s}(id INTEGER PRIMARY KEY,grp INTEGER);CREATE TABLE c{s}(id INTEGER PRIMARY KEY,grp INTEGER);CREATE TABLE b{s}(id INTEGER PRIMARY KEY,aid INTEGER,cid INTEGER,amount INTEGER);')
  nodes=[(i,i//NODE_PER_GROUP) for i in range(G*NODE_PER_GROUP)]
  c.executemany(f'INSERT INTO a{s} VALUES(?,?)',nodes);c.executemany(f'INSERT INTO c{s} VALUES(?,?)',nodes)
  rows=[];ref=defaultdict(lambda:[0,0]);ahset=set(ah)
  for aid,ag in nodes:
   degree=HEAVY_DEGREE if ag in ahset else 1
   for _ in range(degree):
    cg=rng.choice(ch if rng.random()<.97 else cl);cid=cg*NODE_PER_GROUP+rng.randrange(NODE_PER_GROUP);amt=rng.randrange(1,10000)
    rows.append((len(rows),aid,cid,amt))
    if (ag,cg) in needed[s]:ref[ag,cg][0]+=1;ref[ag,cg][1]+=amt
  c.executemany(f'INSERT INTO b{s} VALUES(?,?,?,?)',rows)
  c.executescript(f'CREATE INDEX a{s}_grp ON a{s}(grp);CREATE INDEX c{s}_grp ON c{s}(grp);CREATE INDEX b{s}_aid ON b{s}(aid);CREATE INDEX b{s}_cid ON b{s}(cid);')
  truth.append(ref);edgesizes.append(len(rows))
 c.execute('ANALYZE');c.commit();c.close()
 return path,truth,catalog,edgesizes,time.perf_counter()-start

def frag(s,ag,cg,mode):
 if mode=='left':tables=f'a{s} AS a CROSS JOIN b{s} AS b INDEXED BY b{s}_aid CROSS JOIN c{s} AS c'
 elif mode=='right':tables=f'c{s} AS c CROSS JOIN b{s} AS b INDEXED BY b{s}_cid CROSS JOIN a{s} AS a'
 else:tables=f'a{s} AS a JOIN b{s} AS b ON b.aid=a.id JOIN c{s} AS c ON b.cid=c.id'
 return f'SELECT {s},COUNT(*),COALESCE(SUM(b.amount),0) FROM {tables} WHERE a.grp={int(ag)} AND c.grp={int(cg)} AND b.aid=a.id AND b.cid=c.id'

def jobs(seed,catalog,length,regime):
 rng=random.Random(seed+818181);base=[]
 for j in range(48):
  q=[]
  for s in range(S):
   d=catalog[s]
   # Most requests agree with the default right-to-left plan; rare outliers share a few high-fanout C groups.
   outlier=(j+s)%6==0 if regime in ['sparse_outliers','native_friendly'] else (j+s)%2==0
   if regime=='native_friendly':
    ag=d['ah'][(j//6+s)%2] if outlier else rng.choice(d['al'])
    cg=rng.choice(d['cl']) if outlier else rng.choice(d['ch'])
   else:
    ag=rng.choice(d['al']) if outlier else rng.choice(d['ah'])
    cg=d['ch'][(j//6+s)%2] if outlier else rng.choice(d['cl'])
   q.append((ag,cg))
  base.append(q)
 return [base[j%48] for j in range(length)]

def run(path,truth,catalog,edgesizes,seed,length,regime,policy):
 db=DB(path);stream=jobs(seed,catalog,length,regime);cache={};bounds={};plans={};costs=[];trace=[];updates=0
 # Determine native direction from the engine, not from the hidden degree table.
 native=[]
 for s in range(S):
  r=db.run('EXPLAIN QUERY PLAN '+frag(s,*stream[0][s],'native'),False)['rows']
  native.append('left' if 'b'+str(s)+'_aid' in str(r) else 'right')
 start=time.perf_counter()
 def execute(sql,role,profile=False):
  r=db.run(sql,profile);costs.append({'role':role,'seconds':r['seconds'],'vm_steps':r['vm_steps']});return r
 if policy.startswith('sample'):
  m=int(policy[6:]);rng=random.Random(seed+97414)
  for s,n in enumerate(edgesizes):
   ids=rng.sample(range(n),m)
   r=execute(f"SELECT a.grp,c.grp FROM b{s} AS b JOIN a{s} AS a ON a.id=b.aid JOIN c{s} AS c ON c.id=b.cid WHERE b.id IN ({','.join(map(str,ids))})",'calibration')
   for pos,side in enumerate(['left','right']):
    cc=Counter(row[pos] for row in r['rows'])
    for g in range(G):cache[s,side,g]=n*cc[g]/m
 if policy=='full_join_histogram':
  for s in range(S):
   for side,tab,col in [('left','a','aid'),('right','c','cid')]:
    r=execute(f'SELECT d.grp,COUNT(*) FROM {tab}{s} AS d JOIN b{s} AS b ON b.{col}=d.id GROUP BY d.grp','calibration')
    for g,count in r['rows']:cache[s,side,g]=count
 for j,q in enumerate(stream):
  expected=[[s,*truth[s].get(tuple(q[s]),[0,0])] for s in range(S)]
  if policy=='plan_identify_cached':
   actual=[];actions=[]
   for s,(ag,cg) in enumerate(q):
    key=(s,ag,cg)
    if key not in plans:
     modes=[native[s],'right' if native[s]=='left' else 'left']
     rr=[execute(frag(s,ag,cg,a),'productive' if i==0 else 'calibration') for i,a in enumerate(modes)]
     assert rr[0]['rows']==rr[1]['rows']==[expected[s]]
     plans[key]=modes[min(range(2),key=lambda i:rr[i]['vm_steps'])];actual.append(rr[0]['rows'][0])
    else:actual.append(execute(frag(s,ag,cg,plans[key]),'productive')['rows'][0])
    actions.append(plans[key])
   assert actual==expected
  else:
   if policy=='active_join_count_cached':
    for s,(ag,cg) in enumerate(q):
     for side,g,tab,col in [('left',ag,'a','aid'),('right',cg,'c','cid')]:
      key=(s,side,g)
      if key not in cache:cache[key]=execute(f'SELECT COUNT(*) FROM {tab}{s} AS d JOIN b{s} AS b ON b.{col}=d.id WHERE d.grp={g}','calibration')['rows'][0][0]
   bounded_actions=None
   if policy=='bounded_probe_cached':
    first=[];pending=[]
    for source,pair in enumerate(q):
     order=[native[source],'right' if native[source]=='left' else 'left']
     a=min(order,key=lambda side:cache.get((source,side,pair[0 if side=='left' else 1]),max(edgesizes[source]/G,bounds.get((source,side,pair[0 if side=='left' else 1]),0))))
     first.append(a);g=pair[0 if a=='left' else 1];key=(source,a,g)
     if key not in cache:
      tab,col=('a','aid') if a=='left' else ('c','cid')
      pending.append((key,f'SELECT {source},COUNT(*) FROM {tab}{source} AS d JOIN b{source} AS b ON b.{col}=d.id WHERE d.grp={g}'))
    if pending:
     obs=execute(' UNION ALL '.join(sql for _,sql in pending),'calibration')['rows']
     for (key,_),row in zip(pending,obs):cache[key]=row[1]
    pending=[];bounded_actions=first.copy()
    for source,pair in enumerate(q):
     a=first[source];other='right' if a=='left' else 'left';g=pair[0 if other=='left' else 1];key=(source,other,g)
     current=cache[source,a,pair[0 if a=='left' else 1]]
     if key in cache:
      if cache[key]<current:bounded_actions[source]=other
     elif current>0 and bounds.get(key,0)<current:
      tab,col=('a','aid') if other=='left' else ('c','cid')
      pending.append((key,current,f'SELECT {source},COUNT(*) FROM (SELECT 1 FROM {tab}{source} AS d JOIN b{source} AS b ON b.{col}=d.id WHERE d.grp={g} LIMIT {current+1})'))
    if pending:
     obs=execute(' UNION ALL '.join(sql for _,_,sql in pending),'calibration')['rows']
     for (key,current,_),row in zip(pending,obs):
      count=row[1]
      if count<=current:
       cache[key]=count
       if count<current:bounded_actions[key[0]]=key[1]
      else:bounds[key]=max(bounds.get(key,0),count)
   scores=[{side:cache.get((s,side,q[s][pos]),edgesizes[s]/G) for pos,side in enumerate(['left','right'])} for s in range(S)]
   orders=[[native[s],'right' if native[s]=='left' else 'left'] for s in range(S)]
   if bounded_actions is not None:actions=bounded_actions
   elif policy=='native':actions=['native']*S
   elif policy.startswith('fixed_'):actions=[policy[6:]]*S
   elif policy in ['online_central','gated_central']:actions=list(min(itertools.product(*orders),key=lambda a:sum(scores[s][a[s]] for s in range(S))))
   else:actions=[min(orders[s],key=lambda a:scores[s][a]) for s in range(S)]
   learns=policy.startswith(('online_','gated_')) or (policy=='frozen_after4' and j<4)
   r=execute(' UNION ALL '.join(frag(s,*q[s],actions[s]) for s in range(S)),'productive',(1000 if policy.startswith('gated_') else learns))
   assert r['rows']==expected,(policy,j,r['rows'],expected)
   if learns:
    for scan in r['scans']:
     name=scan['name']
     if name.startswith('b') and name.endswith(('_aid','_cid')):
      s=int(name[1]);side='left' if name.endswith('_aid') else 'right';pos=0 if side=='left' else 1
      cache[s,side,q[s][pos]]=scan['visits'];updates+=1
  trace.append({'job':j,'query':q,'actions':actions})
 elapsed=time.perf_counter()-start;db.close()
 return {'seed':seed,'length':length,'regime':regime,'policy':policy,'elapsed_seconds':elapsed,'vm_steps':sum(r['vm_steps'] for r in costs),'calibration_seconds':sum(r['seconds'] for r in costs if r['role']=='calibration'),'calibration_calls':sum(r['role']=='calibration' for r in costs),'successes':length,'native_directions':native,'model_updates':updates,'trace':trace,'costs':costs}

if __name__=='__main__':
 rr=[];setup=[];rng=random.Random(900173)
 for seed in [15401,27611,39209]:
  path,truth,catalog,edgesizes,seconds=build(seed);setup.append({'seed':seed,'seconds':seconds,'edgesizes':edgesizes,'bytes':path.stat().st_size})
  print('Built join data',seed,edgesizes,flush=True)
  for regime,length in [('sparse_outliers',48),('sparse_outliers',192),('balanced',48)]:
   for rep in range(3):
    order=POLICIES.copy();rng.shuffle(order)
    for p in order:
     r=run(path,truth,catalog,edgesizes,seed,length,regime,p);r['repetition']=rep;rr.append(r)
    print('Completed',seed,regime,length,rep,flush=True)
  (ROOT/'join-results.json').write_text(json.dumps({'setup':setup,'results':rr,'engine':lib.sqlite3_libversion().decode()},indent=2))
 summary={}
 for regime,length in [('sparse_outliers',48),('sparse_outliers',192),('balanced',48)]:
  summary[f'{regime}_{length}']={}
  for p in POLICIES:
   r=[r for r in rr if r['regime']==regime and r['length']==length and r['policy']==p]
   summary[f'{regime}_{length}'][p]={'median_seconds':statistics.median(x['elapsed_seconds'] for x in r),'median_vm_steps':statistics.median(x['vm_steps'] for x in r),'median_calibration_seconds':statistics.median(x['calibration_seconds'] for x in r),'all_correct':all(x['successes']==length for x in r)}
 (ROOT/'join-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
