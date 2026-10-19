"""Minimal read-only ctypes wrapper for pinned SQLite with native scan-status profiling."""
import ctypes as C,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXT='dylib' if sys.platform=='darwin' else 'so'   # built by build_sqlite.sh
lib=C.CDLL(str(ROOT/f'vendor/libsqlite_profile.{EXT}'))
P=C.c_void_p
for name,args,ret in [('sqlite3_open_v2',[C.c_char_p,C.POINTER(P),C.c_int,C.c_char_p],C.c_int),('sqlite3_close',[P],C.c_int),('sqlite3_prepare_v2',[P,C.c_char_p,C.c_int,C.POINTER(P),C.POINTER(C.c_char_p)],C.c_int),('sqlite3_step',[P],C.c_int),('sqlite3_finalize',[P],C.c_int),('sqlite3_column_count',[P],C.c_int),('sqlite3_column_type',[P,C.c_int],C.c_int),('sqlite3_column_int64',[P,C.c_int],C.c_int64),('sqlite3_column_double',[P,C.c_int],C.c_double),('sqlite3_column_text',[P,C.c_int],C.c_char_p),('sqlite3_stmt_scanstatus',[P,C.c_int,C.c_int,P],C.c_int),('sqlite3_stmt_status',[P,C.c_int,C.c_int],C.c_int),('sqlite3_errmsg',[P],C.c_char_p),('sqlite3_libversion',[],C.c_char_p)]:
 fn=getattr(lib,name);fn.argtypes=args;fn.restype=ret
class DB:
 def __init__(self,path):
  self.db=P();assert lib.sqlite3_open_v2(str(path).encode(),C.byref(self.db),1,None)==0
 def close(self):
  if self.db:assert lib.sqlite3_close(self.db)==0;self.db=None
 def run(self,sql,profile=True):
  start=time.perf_counter();stmt=P();rc=lib.sqlite3_prepare_v2(self.db,sql.encode(),-1,C.byref(stmt),None)
  if rc:raise RuntimeError(lib.sqlite3_errmsg(self.db).decode())
  rows=[];scans=[]
  try:
   while True:
    rc=lib.sqlite3_step(stmt)
    if rc==101:break
    if rc!=100:raise RuntimeError(lib.sqlite3_errmsg(self.db).decode())
    row=[]
    for k in range(lib.sqlite3_column_count(stmt)):
     t=lib.sqlite3_column_type(stmt,k)
     row.append(lib.sqlite3_column_int64(stmt,k) if t==1 else lib.sqlite3_column_double(stmt,k) if t==2 else lib.sqlite3_column_text(stmt,k).decode() if t==3 else None)
    rows.append(row)
   i=0;steps=lib.sqlite3_stmt_status(stmt,4,0)
   collect=bool(profile) and (isinstance(profile,bool) or steps>profile)
   if collect:
    while True:
     visits=C.c_int64()
     if lib.sqlite3_stmt_scanstatus(stmt,i,1,C.byref(visits)):break
     loops=C.c_int64();name=C.c_char_p();est=C.c_double();description=C.c_char_p()
     for op,x in [(0,loops),(2,est),(3,name),(4,description)]:lib.sqlite3_stmt_scanstatus(stmt,i,op,C.byref(x))
     scans.append({'name':name.value.decode() if name.value else '', 'visits':visits.value,'loops':loops.value,'estimate':est.value,'description':description.value.decode() if description.value else ''});i+=1
   steps=lib.sqlite3_stmt_status(stmt,4,0)
  finally:lib.sqlite3_finalize(stmt)
  return {'rows':rows,'scans':scans,'vm_steps':steps,'seconds':time.perf_counter()-start}
