"""Model v1 (combined form, FORM_combined_v1.md): T = max(T_callback measured, T_chain) per launch (TrueKNN per round). CPU only, reads existing raw files.
Usage: python3 -I model/priorworks/score_model_v1.py > model/priorworks/results_model_v1.md"""
import re, glob, json, numpy as np
from scipy.spatial import cKDTree
E='/home/min/a/cashman3/eoss/'; R33=E+'notes/q33/raw/'; R34=E+'notes/q34/raw/'; M=E+'model/priorworks/'
old=json.load(open(M+'pred.json')); wp=json.load(open(M+'pred_warp.json')); cp=json.load(open(M+'pred_cp.json')); SM,CLK=142,2.5e9
med=lambda x: float(np.median(x)); lg=lambda m,p: abs(np.log(m/p))
def rtt(pat): return med([float(re.search(r'time search compute: (\S+)',open(f).read())[1]) for f in sorted(glob.glob(pat))])
rows=[]  # (program, label, measured, callback form, chain, v1)
u=np.loadtxt('/dev/shm/pw/data/uni3_ref.csv',delimiter=','); Cref=cKDTree(u).query_ball_point(u,0.0232,p=np.inf,return_length=True,workers=-1).astype(float)
cbref=(rtt(R34+'rtnn_uni3_ref_ref_full_*')-rtt(R34+'rtnn_uni3_ref_ref_stub_*'))*1e-3*SM*CLK/(1.10*Cref.sum())
for k,v in old.items():
    if not k.startswith('rtnn'): continue
    _,ds,r=k.split('_'); m=rtt(R33+'rtnn_%s_%s_*'%(ds,r)); full=rtt(R34+'rtnn_%s_%s_full_*'%(ds,r)); stub=rtt(R34+'rtnn_%s_%s_stub_*'%(ds,r)); N=v['N']
    cb=(full-stub)*1e-3*SM*CLK/(N*1.10*v['C'])
    qm=1.10*(50*(v['C']+1)/(v['S']+1) if v['S']>50 else v['C'])+1
    pb=(110+(4.9+cb-cbref)*qm)*N/(SM*CLK)*1e3; ch=cp[k]['chain']
    rows.append(('RTNN',k[5:],m,pb,ch,max(pb,ch)))
for k,v in old.items():
    if not k.startswith('trueknn'): continue
    _,ds,kk=k.split('_'); Ts=[];Bs=[]
    for f in sorted(glob.glob(R33+'tk_%s_%s_fixed_*.txt'%(ds,kk))):
        rr=re.findall(r'GT span_ms (\S+) CBK warps \d+ total_warp_cycles (\S+) f0_cycles (\S+)',open(f).read())
        if not rr: continue
        Ts.append(sum(float(a) for a,b,c in rr)); Bs.append([float(c)*1.485/(SM*12.8)/CLK*1e3 for a,b,c in rr])
    n=min(len(b) for b in Bs); B=np.median([b[:n] for b in Bs],axis=0); ch=np.array(cp[k]['chain'][:n]+[0]*(n-len(cp[k]['chain'])))
    rows.append(('TrueKNN','%s %s (%d probe rounds, %d chain rounds)'%(ds,kk,n,len(cp[k]['chain'])),med(Ts),float(B.sum()),float(ch.sum()),float(np.maximum(B,ch).sum())))
def ark(ds,var):
    s=open(sorted(glob.glob(R34+'ark_%s_%s_*.txt'%(ds,var)))[0]).read(); t=float(re.search(r'KERNEL queries \d+ ms (\S+)',s)[1]); c=re.search(r'total_warp_cycles (\S+) trace_warp_cycles (\S+) f0_cycles (\S+)',s); return t,float(c[3])
for k,v in old.items():
    if not k.startswith('arkade'): continue
    ds=k.split('_')[1]
    for var in ('fixed','orig'):
        t,F=ark(ds,var); tr,Fr=ark('ref',var); pb=tr*F/Fr; B=cp[k]['B']*(0.09 if var=='fixed' else 1)
        rows.append(('Arkade','%s %s'%(ds,var),t,pb,B,max(pb,B)))
print('# Model v1 scored on the 14 real rows (2026-11-17, triton, no run, no refit)\n\nForm: model/priorworks/FORM_combined_v1.md (committed eb22f98 before this script). Ratio = measured / predicted, closes when 0.5 to 2. NOT blind (see the form file). Units ms.\n')
print('| program | row | measured | callback form | chain | v1 = max | ratio v1 | ratio callback alone |\n|---|---|---|---|---|---|---|---|')
for p,l,m,b,c,v in rows: print('| %s | %s | %.4g | %.4g | %.4g | %.4g | %.2f | %.2f |'%(p,l,m,b,c,v,m/v,m/b))
print()
for p in ('RTNN','TrueKNN','Arkade','ALL'):
    s=[r for r in rows if p=='ALL' or r[0]==p]; e=[lg(r[2],r[5]) for r in s]; ins=sum(0.5<=r[2]/r[5]<=2 for r in s)
    print('- %s: median abs log error v1 %.2f (callback alone %.2f), inside factor 2: %d of %d, worst ratio %.1f'%(p,med(e),med([lg(r[2],r[3]) for r in s]),ins,len(s),max(max(r[2]/r[5],r[5]/r[2]) for r in s)))
