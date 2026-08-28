#!/usr/bin/env python3
"""
Compare CartiMorph v16-final outputs with the supplied POMA/Chondrometrics table.

Inputs
------
--root registrations
  registrations/oaizib_357/
      MorphQuant.csv
      MorphQuant_Compartments.csv
      FCL_Areas.csv
      MorphQuant_meta.json

--chondrometrics Chondrometrics_AllMetrics_with_CMT-ID.csv

Strict comparable analyses
--------------------------
A) Regional COMMON16
   FCL (%) and Mean Thickness (mm), 16 direct ROI counterparts:
     ecMFC ccMFC icMFC
     ecLFC ccLFC icLFC
     aMTC eMTC pMTC iMTC cMTC
     aLTC eLTC pLTC iLTC cLTC

   Chondrometrics columns:
     <ROI>_dABp
     <ROI>_ThCtAB_aMe

B) Four direct aggregate compartments
   MTC, cMFC, LTC, cLFC:
     FCL (%)            <comp>_dABp
     Mean Thickness mm  <comp>_ThCtAB_aMe
     Surface Area mm2   <comp>_cAB * 100 (source cAB is cm2)
     Volume mm3         <comp>_VC

Important
---------
Chondrometrics/POMA is an independent analysis source, not assumed to use the
same OAIZIB cartilage masks. Numerical equality is NOT expected. This script
quantifies agreement/bias without applying calibration multipliers.

Outputs
-------
comparison_summary.txt
region_pairs.csv
compartment_pairs.csv
summary_overall.csv
summary_by_region.csv
summary_by_compartment.csv
excluded_cases.csv
"""

from __future__ import annotations
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

ROI_NAMES = [
    "aMFC","ecMFC","ccMFC","icMFC","pMFC",
    "aLFC","ecLFC","ccLFC","icLFC","pLFC",
    "aMTC","eMTC","pMTC","iMTC","cMTC",
    "aLTC","eLTC","pLTC","iLTC","cLTC",
]
COMMON16 = [
    "ecMFC","ccMFC","icMFC",
    "ecLFC","ccLFC","icLFC",
    "aMTC","eMTC","pMTC","iMTC","cMTC",
    "aLTC","eLTC","pLTC","iLTC","cLTC",
]
COMPS = ["MTC","cMFC","LTC","cLFC"]
REGION_METRICS = ["FCL","Mean Thickness"]
COMP_METRICS = ["FCL","Mean Thickness","Surface Area","Volume"]

def num(x):
    try:
        s=str(x).strip()
        if not s or s.lower() in {"nan","na","n/a","none","null"}:
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")

def finite(x):
    try: return math.isfinite(float(x))
    except Exception: return False

def case_cmt_id(case: str):
    m=re.search(r"(\d+)$", case)
    return int(m.group(1)) if m else None

def read_matrix(path: Path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.reader(f))
    if not rows: raise ValueError(f"empty CSV: {path}")
    cols=[str(x).strip() for x in rows[0][1:]]
    out={}
    for row in rows[1:]:
        if not row: continue
        name=str(row[0]).strip()
        out[name]={c:num(v) for c,v in zip(cols,row[1:])}
    return out

def read_chondro(path: Path):
    out={}
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        rd=csv.DictReader(f)
        fields=rd.fieldnames or []
        if "CMT-ID" not in fields:
            raise ValueError("Chondrometrics CSV missing CMT-ID")
        required=[]
        for roi in COMMON16:
            required += [f"{roi}_dABp", f"{roi}_ThCtAB_aMe"]
        for c in COMPS:
            required += [f"{c}_dABp", f"{c}_ThCtAB_aMe", f"{c}_cAB", f"{c}_VC"]
        miss=[x for x in required if x not in fields]
        if miss:
            raise ValueError(f"Chondrometrics CSV missing columns: {miss}")
        for r in rd:
            raw=r.get("CMT-ID","")
            if not str(raw).strip(): continue
            cid=int(round(float(raw)))
            if cid in out: raise ValueError(f"duplicate CMT-ID {cid}")
            out[cid]=r
    return out

def pearson(x,y):
    if len(x)<2 or np.std(x,ddof=1)==0 or np.std(y,ddof=1)==0: return float("nan")
    return float(np.corrcoef(x,y)[0,1])

def rank_average(a):
    a=np.asarray(a,float); order=np.argsort(a,kind="mergesort"); r=np.empty(len(a),float)
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]: j+=1
        rr=(i+1+j)/2.0
        r[order[i:j]]=rr
        i=j
    return r

def spearman(x,y):
    if len(x)<2: return float("nan")
    return pearson(rank_average(x),rank_average(y))

def ccc(ref,pred):
    if len(ref)<2: return float("nan")
    mx,my=float(np.mean(ref)),float(np.mean(pred))
    vx,vy=float(np.var(ref,ddof=1)),float(np.var(pred,ddof=1))
    cov=float(np.cov(ref,pred,ddof=1)[0,1])
    den=vx+vy+(mx-my)**2
    return float(2*cov/den) if den else (1.0 if mx==my else float("nan"))

def summarize(rows):
    if not rows: return {}
    ref=np.asarray([r["chondrometrics"] for r in rows],float)
    py=np.asarray([r["python"] for r in rows],float)
    ok=np.isfinite(ref)&np.isfinite(py); ref=ref[ok]; py=py[ok]
    if len(ref)==0: return {}
    d=py-ref
    dsd=float(np.std(d,ddof=1)) if len(d)>1 else float("nan")
    if len(ref)>1 and np.std(ref,ddof=1)>0:
        slope,intercept=np.polyfit(ref,py,1)
    else:
        slope=intercept=float("nan")
    rm=float(np.mean(ref))
    return {
        "n":len(ref),
        "python_mean":float(np.mean(py)),
        "python_sd":float(np.std(py,ddof=1)) if len(py)>1 else 0.0,
        "chondro_mean":rm,
        "chondro_sd":float(np.std(ref,ddof=1)) if len(ref)>1 else 0.0,
        "bias_python_minus_chondro":float(np.mean(d)),
        "bias_sd":dsd,
        "loa_lower":float(np.mean(d)-1.96*dsd) if finite(dsd) else float("nan"),
        "loa_upper":float(np.mean(d)+1.96*dsd) if finite(dsd) else float("nan"),
        "mae":float(np.mean(np.abs(d))),
        "rmse":float(np.sqrt(np.mean(d*d))),
        "pearson_r":pearson(ref,py),
        "spearman_rho":spearman(ref,py),
        "ccc":ccc(ref,py),
        "ols_slope_python_on_chondro":float(slope),
        "ols_intercept":float(intercept),
        "relative_mean_bias_pct":100.0*float(np.mean(d))/rm if rm!=0 else float("nan"),
    }

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=list(rows[0].keys())
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def group_summary(rows, keys):
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rr in sorted(groups.items(), key=lambda x:tuple(map(str,x[0]))):
        rec={k:v for k,v in zip(keys,key)}
        rec.update(summarize(rr))
        out.append(rec)
    return out

def fmt(x,n=4):
    return "NaN" if not finite(x) else f"{float(x):.{n}f}"

def main():
    ap=argparse.ArgumentParser(description="Compare CartiMorph v16-final with Chondrometrics/POMA")
    ap.add_argument("--root",type=Path,default=Path("registrations"))
    ap.add_argument("--chondrometrics",type=Path,default=Path("Chondrometrics_AllMetrics_with_CMT-ID.csv"))
    ap.add_argument("--morph-name",default="MorphQuant.csv")
    ap.add_argument("--compartment-name",default="MorphQuant_Compartments.csv")
    ap.add_argument("--out",type=Path,default=Path("comparison_chondrometrics_v16"))
    args=ap.parse_args()

    if not args.root.is_dir(): raise FileNotFoundError(args.root)
    if not args.chondrometrics.is_file(): raise FileNotFoundError(args.chondrometrics)
    ref=read_chondro(args.chondrometrics)

    region_rows=[]; comp_rows=[]; excluded=[]
    for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir() and p.name.lower().startswith("oaizib_")):
        cid=case_cmt_id(case_dir.name)
        if cid is None or cid not in ref:
            excluded.append({"case":case_dir.name,"reason":"no matching CMT-ID in Chondrometrics CSV"}); continue
        py_path=case_dir/args.morph_name
        cp_path=case_dir/args.compartment_name
        missing=[str(x.name) for x in (py_path,cp_path) if not x.is_file()]
        if missing:
            excluded.append({"case":case_dir.name,"reason":"missing "+", ".join(missing)}); continue
        try:
            py=read_matrix(py_path); pc=read_matrix(cp_path)
        except Exception as e:
            excluded.append({"case":case_dir.name,"reason":f"read error: {e}"}); continue
        rr=ref[cid]

        for metric in REGION_METRICS:
            if metric not in py: continue
            suffix="dABp" if metric=="FCL" else "ThCtAB_aMe"
            for roi in COMMON16:
                pv=py[metric].get(roi,float("nan")); rv=num(rr.get(f"{roi}_{suffix}"))
                if finite(pv) and finite(rv):
                    region_rows.append({
                        "case":case_dir.name,"CMT_ID":cid,"metric":metric,"region":roi,
                        "python":float(pv),"chondrometrics":float(rv),
                        "difference":float(pv)-float(rv),
                    })

        for metric in COMP_METRICS:
            if metric not in pc: continue
            for comp in COMPS:
                pv=pc[metric].get(comp,float("nan"))
                if metric=="FCL": rv=num(rr.get(f"{comp}_dABp"))
                elif metric=="Mean Thickness": rv=num(rr.get(f"{comp}_ThCtAB_aMe"))
                elif metric=="Surface Area":
                    x=num(rr.get(f"{comp}_cAB")); rv=x*100.0 if finite(x) else float("nan")
                else: rv=num(rr.get(f"{comp}_VC"))
                if finite(pv) and finite(rv):
                    comp_rows.append({
                        "case":case_dir.name,"CMT_ID":cid,"metric":metric,"compartment":comp,
                        "python":float(pv),"chondrometrics":float(rv),
                        "difference":float(pv)-float(rv),
                    })

    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/"region_pairs.csv",region_rows)
    write_csv(args.out/"compartment_pairs.csv",comp_rows)
    write_csv(args.out/"excluded_cases.csv",excluded)

    overall=[]
    for metric in REGION_METRICS:
        rr=[r for r in region_rows if r["metric"]==metric]
        if rr:
            rec={"scope":"COMMON16_REGION","metric":metric}; rec.update(summarize(rr)); overall.append(rec)
    for metric in COMP_METRICS:
        rr=[r for r in comp_rows if r["metric"]==metric]
        if rr:
            rec={"scope":"FOUR_COMPARTMENTS","metric":metric}; rec.update(summarize(rr)); overall.append(rec)

    by_region=group_summary(region_rows,["metric","region"])
    by_comp=group_summary(comp_rows,["metric","compartment"])
    write_csv(args.out/"summary_overall.csv",overall)
    write_csv(args.out/"summary_by_region.csv",by_region)
    write_csv(args.out/"summary_by_compartment.csv",by_comp)

    lines=[
        "CARTIMORPH V16-FINAL vs CHONDROMETRICS/POMA",
        "============================================",
        f"root: {args.root}",
        f"Chondrometrics source: {args.chondrometrics}",
        f"regional pairs: {len(region_rows)}",
        f"compartment pairs: {len(comp_rows)}",
        f"excluded cases: {len(excluded)}",
        "",
        "STRICT COMPARISON DESIGN",
        "  COMMON16 regional: FCL, Mean Thickness",
        "  Four compartments MTC/cMFC/LTC/cLFC: FCL, Mean Thickness, Surface Area, Volume",
        "  Chondrometrics cAB converted cm^2 -> mm^2 by x100.",
        "  No calibration/scaling is applied to Python results.",
        "",
        "OVERALL RESULTS",
    ]
    for r in overall:
        lines += [
            f"{r['scope']} | {r['metric']} | n={r['n']}",
            f"  Python mean={fmt(r['python_mean'])}  Chondro mean={fmt(r['chondro_mean'])}",
            f"  bias={fmt(r['bias_python_minus_chondro'])}  MAE={fmt(r['mae'])}  RMSE={fmt(r['rmse'])}",
            f"  Pearson={fmt(r['pearson_r'])}  Spearman={fmt(r['spearman_rho'])}  CCC={fmt(r['ccc'])}",
            f"  relative mean bias={fmt(r['relative_mean_bias_pct'],2)}%",
        ]
    lines += [
        "",
        "INTERPRETATION NOTE",
        "  The POMA/Chondrometrics measurements are an independent analysis source",
        "  and are not assumed to use the same OAIZIB manual voxel masks.",
        "  Agreement should therefore be reported as external-method comparison,",
        "  not as a requirement for numerical identity.",
    ]
    (args.out/"comparison_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved under: {args.out}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
