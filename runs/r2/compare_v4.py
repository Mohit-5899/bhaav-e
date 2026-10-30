"""Paired r1 vs r2 comparison on the same test clips, effect checker v4 (OUR script).
Usage: python runs/r2/compare_v4.py meta.jsonl
meta.jsonl: one line per clip with the generator's labels (id, category, subcategory, motion, effect, background_name, accent).
Prints per-split error rates, the exact two-sided sign test on clips where only one run is right, and v3 vs v4 agreement."""
import json, os, sys, collections, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../model'))
R=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + '/'
meta={}
for l in open(sys.argv[1]):
    m=json.loads(l); meta[m['id']]=m
def labels(m):
    return dict(subject=f"{m['category']}/{m['subcategory']}", motion=None if m["category"]=="ui" else m["motion"],
                effect=m["effect"], background=m["background_name"], accent=m.get("accent"))
A=['subject','motion','effect','background','accent']
def ok(pred,truth):
    o={a:[None if t[a] is None else pred[a][i]==t[a] for i,t in enumerate(truth)] for a in A}
    o['all']=[all(o[a][i] is not False for a in A) for i in range(len(truth))]; return o
def binom_p(b,c):
    n=b+c; k=min(b,c)
    return min(1, 2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1
a=json.load(open(R+'r1/eval_v4/report.json')); b=json.load(open(R+'r2/eval_v4/report.json'))
for sp in ('test_seen','test_unseen'):
    ids=a[sp]['clips']; truth=[labels(meta[c]) for c in ids]
    real=ok(a[sp]['real'],truth)
    for g in ('cfg:3','cfg:5'):
        o1=ok(a[sp][g]['pred'],truth); o2=ok(b[sp][g]['pred'],truth)
        for att in ('all','effect'):
            idx=[i for i in range(len(ids)) if real[att][i]]
            fixed=sum(1 for i in idx if not o1[att][i] and o2[att][i]); broke=sum(1 for i in idx if o1[att][i] and not o2[att][i])
            e1=sum(1 for i in idx if not o1[att][i]); e2=sum(1 for i in idx if not o2[att][i])
            print(f"{sp} {g} {att}: n={len(idx)} r1_err={e1} ({e1/len(idx):.1%}) r2_err={e2} ({e2/len(idx):.1%}) fixed={fixed} broke={broke} p={binom_p(fixed,broke):.3f}")
    for name,rep in (('r1',a),('r2',b)):
        conf=collections.Counter()
        for i,c in enumerate(ids):
            if real['effect'][i] and rep[sp]['cfg:3']['pred']['effect'][i]!=truth[i]['effect']:
                conf[(truth[i]['subject'].split('/')[1],truth[i]['effect'],rep[sp]['cfg:3']['pred']['effect'][i])]+=1
        print(f"  {name} {sp} cfg:3 effect mistakes (subject, asked, got):", dict(conf.most_common(12)))
print('---omissions (asked an effect, got none), cfg:3, real-readable effect clips')
for sp in ('test_seen','test_unseen'):
    ids=a[sp]['clips']; truth=[labels(meta[c]) for c in ids]; real=ok(a[sp]['real'],truth)
    for name,rep in (('r1',a),('r2',b)):
        p=rep[sp]['cfg:3']['pred']['effect']; idx=[i for i in range(len(ids)) if real['effect'][i] and p[i]!=truth[i]['effect']]
        om=sum(1 for i in idx if p[i]=='none'); print(sp,name,'effect errors',len(idx),'omissions',om,'additions',sum(1 for i in idx if truth[i]['effect']=='none'))
print('---v3 vs v4 on the same generated clips (effect), cfg:3, real-readable by v4')
for sp in ('test_seen','test_unseen'):
    ids=a[sp]['clips']; truth=[labels(meta[c]) for c in ids]; real=ok(a[sp]['real'],truth)
    for name,rep in (('r1',a),('r2',b)):
        p=rep[sp]['cfg:3']['pred']; v4=p['effect']; v3=p.get('effect_v3')
        if v3 is None: print('no effect_v3 key', list(p)); continue
        idx=[i for i in range(len(ids)) if real['effect'][i]]
        e3=sum(v3[i]!=truth[i]['effect'] for i in idx); e4=sum(v4[i]!=truth[i]['effect'] for i in idx)
        agree=sum(v3[i]==v4[i] for i in idx)
        v3none=sum(v3[i]=='none' and truth[i]['effect']!='none' and v4[i]==truth[i]['effect'] for i in idx)
        print(f"{sp} {name}: n={len(idx)} v3_err={e3} v4_err={e4} agree={agree} v3_said_none_v4_says_correct={v3none}")
