"""Terminal-only five-arm accuracy and physical cost audit; labels stay unchanged."""
import math


def paired(left,right):
    ids=sorted(set(left)&set(right))
    gains=[q for q in ids if not left[q] and right[q]]
    losses=[q for q in ids if left[q] and not right[q]]
    n=len(gains)+len(losses)
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(len(gains),len(losses))+1))/2**n) if n else 1.0
    return {'n':len(ids),'gains':gains,'losses':losses,'net':len(gains)-len(losses),'exact_two_sided_p':p}


def holm(items):
    ordered = sorted(items.items(), key=lambda kv: kv[1]["exact_two_sided_p"])
    last = 0
    for i,(_,v) in enumerate(ordered):
        last=max(last,min(1,(len(ordered)-i)*v['exact_two_sided_p']))
        v['holm_p']=last
