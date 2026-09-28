import glob,os,time
from inspect_ai.log import read_eval_log
lanes={"a":"inspect-logs","b":"inspect-logs-b","a2":"inspect-logs-a2","b2":"inspect-logs-b2"}
prev={}
def denom(task): return 86 if "level2" in str(task) else 53
while True:
    for k,L in lanes.items():
        fs=sorted(glob.glob("_state/%s/2026-09-*T*gaia-level*.eval"%L),key=os.path.getmtime)
        if not fs: continue
        f=fs[-1]
        try:
            e=read_eval_log(f, header_only=True); st=e.status; N=denom(e.eval.task)
        except Exception:
            continue
        n=0;c=0
        try:
            if e.reductions:
                for red in e.reductions:
                    for s in (red.samples or []):
                        n+=1
                        if s.value in ("C",1,1.0,"1",True): c+=1
            else:
                e2=read_eval_log(f)
                for s in (e2.samples or []):
                    n+=1
                    v=None
                    for _,sc in (s.scores or {}).items(): v=sc.value
                    if v=="C": c+=1
        except Exception:
            continue
        lvl="l2" if N==86 else "l1"
        msg=None
        if st in ("success","error","cancelled"):
            msg="lane_%s_%s DONE status=%s solve=%d/%d solve/%d=%.3f"%(k,lvl,st,c,n,N,c/N)
        elif n>=N-2:
            msg="lane_%s_%s TAIL samples=%d/%d"%(k,lvl,n,N)
        if msg and prev.get(k)!=msg:
            print(msg,flush=True); prev[k]=msg
    time.sleep(150)
