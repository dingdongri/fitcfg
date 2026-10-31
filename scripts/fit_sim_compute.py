"""sim_3d 계산 시간 항을 실제 GPU rank 0 측정(gpu_parallel.csv)으로 맞추기: eff(full·lora), t_half, TP 벌점 격자 탐색.
    python scripts/fit_sim_compute.py data/gpu_parallel.csv
"""
import csv, sys, math, itertools, statistics as st
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import fitcfg.parallel as S
from fitcfg.models import get_model
G = S.GPU("mig3g40", 39.2, 133.7, 900)
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['ok']=='1']
def pred(r, eff, T0, tp_pen):
    m = get_model(r['model']); S.SEQ=int(r['seq']); ck=r['ckpt']=='1'; meth=r['method']; tok=S.SEQ
    tp = int(r['ws']) if r['kind']=='tp' else 1
    h = m.hidden/tp
    util = tok/(tok + T0*(4096/h))
    speed = G.tflops*1e12*eff[meth]*util*(1/(1+tp_pen*(tp-1)))
    nl = int(r['n_layers']) if r['kind']=='pp' else m.n_layers
    last = r['kind']!='pp' or r['stage']=='last'
    f = nl*S.layer_flops_per_token(m, meth, ck)*tok/tp
    if last: f += S.head_flops_per_token(m, meth)*tok
    return f/speed
best=None
for e_full in [x/100 for x in range(30,95,2)]:
  for e_lora in [x/100 for x in range(30,95,2)]:
    for T0 in [0,64,128,256,512,1024]:
      for pen in [0,0.05,0.1,0.15,0.2,0.3]:
        errs=[math.log(pred(r,{'full':e_full,'lora':e_lora},T0,pen)/(float(r['step_ms'])/1000)) for r in rows]
        sse=sum(e*e for e in errs)
        if best is None or sse<best[0]: best=(sse,e_full,e_lora,T0,pen,errs)
sse,ef,el,T0,pen,errs=best
print('eff_full',ef,'eff_lora',el,'T0',T0,'tp_pen',pen,'MAPE',round(st.mean(abs(math.exp(e)-1) for e in errs)*100,1),'max',round(max(abs(math.exp(e)-1) for e in errs)*100,1))
for r,e in zip(rows,errs): print(r['kind'],r['model'],r['method'],r.get('ws'),r.get('stage'),r['seq'],r['ckpt'],f'{(math.exp(e)-1)*100:+.0f}%')
