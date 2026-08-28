#!/usr/bin/env python3
"""
Reproduce the CartiMorph paper-style pseudo hit rate (pHR) analysis for FCL.

Paper:
  Yao Y, Zhong J, Zhang L, Khan S, Chen W.
  CartiMorph: A framework for automated knee articular cartilage morphometrics.
  Medical Image Analysis. 2024;91:103035.
  DOI: 10.1016/j.media.2023.103035

Paper pHR definition:
  - Three human raters are analysed independently.
  - Manual FCL grades 0..10 are converted to continuous FCL by grade * 10,
    so grade 1 -> 10%, ..., grade 10 -> 100%.
  - For tolerance R (percentage points), a prediction P is a "hit" when:
        |P - G| <= R
    where G is the human continuous ground truth.
  - pHR = hits / number of ground-truth observations.
  - The paper evaluates tolerances from 5% to 100%.
  - It reports approximately 0.85 pHR at 10% tolerance for CartiMorph.

IMPORTANT ABOUT ROI SCOPE
-------------------------
The supplied POMA/Chondrometrics CSV has direct regional dABp counterparts for
16 CartiMorph ROIs:
    ecMFC ccMFC icMFC
    ecLFC ccLFC icLFC
    aMTC eMTC pMTC iMTC cMTC
    aLTC eLTC pLTC iLTC cLTC

It does NOT contain direct aMFC/pMFC/aLFC/pLFC regional dABp values.

Therefore this script reports:
  1) COMMON16_PAIRED
     Primary apples-to-apples comparison. Only observations where BOTH Python
     and Chondrometrics have finite FCL are included, so both methods use the
     exact same denominator for every rater/tolerance.

  2) ALL20_PYTHON
     Secondary analysis using all 20 CartiMorph ROIs for Python only.

No Chondrometrics values are invented or distributed into missing femoral ROIs.

Inputs
------
--root:
    registrations/
      oaizib_224/MorphQuant.csv
      ...

--mapping:
    Chondrometrics_AllMetrics_with_CMT-ID.csv
    This serves BOTH as:
      - CMT-ID -> SubjectID mapping
      - Chondrometrics regional dABp source

--rater1/2/3:
    FCLgrading_rater*.xlsx
    Expected columns include Subject and the 20 CartiMorph ROI names.

Outputs
-------
out/
  phr_curve.csv
  phr_key_tolerances.csv
  phr_rater_summary.csv
  distribution_by_grade.csv
  records_long.csv
  excluded_cases.csv
  paper_phr_summary.txt

No pandas/openpyxl dependency: XLSX is read directly from its XML.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROI_NAMES = [
    "aMFC", "ecMFC", "ccMFC", "icMFC", "pMFC",
    "aLFC", "ecLFC", "ccLFC", "icLFC", "pLFC",
    "aMTC", "eMTC", "pMTC", "iMTC", "cMTC",
    "aLTC", "eLTC", "pLTC", "iLTC", "cLTC",
]

COMMON16 = [
    "ecMFC", "ccMFC", "icMFC",
    "ecLFC", "ccLFC", "icLFC",
    "aMTC", "eMTC", "pMTC", "iMTC", "cMTC",
    "aLTC", "eLTC", "pLTC", "iLTC", "cLTC",
]

CHONDRO_FCL_COLUMN = {roi: f"{roi}_dABp" for roi in COMMON16}

PAPER_APPROX_PHR_AT_10 = 0.85


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def num(x) -> float:
    try:
        s = str(x).strip()
        if not s or s.lower() in {"nan", "na", "n/a", "none", "null"}:
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def fmt(x, digits=4) -> str:
    if not finite(x):
        return "NaN"
    return f"{float(x):.{digits}f}"


def percentile(values: Sequence[float], q: float) -> float:
    a = np.asarray([float(x) for x in values if finite(x)], dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.percentile(a, q))


# ---------------------------------------------------------------------------
# XLSX reader copied in spirit from the already-used compare_fcl_human.py:
# direct XLSX XML parsing, first worksheet, no openpyxl dependency.
# ---------------------------------------------------------------------------

def _xlsx_rows(path: Path) -> List[List[object]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append(
                    "".join(
                        t.text or ""
                        for t in si.iter("{%s}t" % ns["m"])
                    )
                )

        sheet_path = "xl/worksheets/sheet1.xml"
        try:
            wb = ET.fromstring(z.read("xl/workbook.xml"))
            relns = {
                "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                "m": ns["m"],
            }
            sh = wb.find(".//m:sheets/m:sheet", relns)
            rid = (
                sh.attrib.get("{%s}id" % relns["r"])
                if sh is not None else None
            )
            if rid and "xl/_rels/workbook.xml.rels" in z.namelist():
                relroot = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
                for rel in relroot:
                    if rel.attrib.get("Id") == rid:
                        target = rel.attrib.get(
                            "Target", "worksheets/sheet1.xml"
                        ).lstrip("/")
                        sheet_path = (
                            target if target.startswith("xl/")
                            else "xl/" + target
                        )
                        break
        except Exception:
            pass

        root = ET.fromstring(z.read(sheet_path))
        rows: List[List[object]] = []
        for row in root.findall(".//m:sheetData/m:row", ns):
            vals: List[object] = []
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "A1")
                letters = "".join(ch for ch in ref if ch.isalpha())
                idx = 0
                for ch in letters:
                    idx = idx * 26 + ord(ch.upper()) - 64
                while len(vals) < idx - 1:
                    vals.append(None)

                typ = cell.attrib.get("t")
                v = cell.find("m:v", ns)
                if typ == "inlineStr":
                    t = cell.find(".//m:t", ns)
                    value: object = t.text if t is not None else ""
                elif v is None:
                    value = None
                elif typ == "s":
                    value = shared[int(v.text)]
                else:
                    raw = v.text
                    try:
                        x = float(raw)
                        value = int(x) if x.is_integer() else x
                    except Exception:
                        value = raw
                vals.append(value)
            rows.append(vals)
        return rows


def read_rater(path: Path) -> Dict[int, Dict[str, int]]:
    rows = _xlsx_rows(path)
    if not rows:
        raise ValueError(f"Empty rater workbook: {path}")

    hdr = [str(x).strip() if x is not None else "" for x in rows[0]]
    try:
        subj_col = hdr.index("Subject")
    except ValueError:
        # Backward-compatible fallback used in the earlier validation script.
        subj_col = 0

    roi_cols = {roi: hdr.index(roi) for roi in ROI_NAMES if roi in hdr}
    missing = [roi for roi in ROI_NAMES if roi not in roi_cols]
    if missing:
        raise ValueError(
            f"{path}: missing ROI columns: {missing}; header={hdr[:30]}"
        )

    out: Dict[int, Dict[str, int]] = {}
    for row in rows[1:]:
        if len(row) <= subj_col or row[subj_col] in (None, ""):
            continue
        sid = int(round(float(row[subj_col])))
        vals: Dict[str, int] = {}
        for roi, j in roi_cols.items():
            if j >= len(row) or row[j] in (None, ""):
                continue
            g = int(round(float(row[j])))
            if not 0 <= g <= 10:
                raise ValueError(
                    f"{path}: Subject {sid}, {roi}: grade {g} outside 0..10"
                )
            vals[roi] = g
        out[sid] = vals
    return out


# ---------------------------------------------------------------------------
# Mapping, Python MorphQuant and Chondrometrics
# ---------------------------------------------------------------------------

def normalize_case_from_cmt(raw) -> str:
    return f"oaizib_{int(round(float(raw))):03d}"


def read_mapping_and_chondro(path: Path):
    case_to_subject: Dict[str, int] = {}
    chondro: Dict[Tuple[str, str], float] = {}
    rows_by_case: Dict[str, dict] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        for required in ["CMT-ID", "SubjectID"]:
            if required not in fields:
                raise ValueError(
                    f"{path}: missing required column {required}; first fields={fields[:20]}"
                )
        missing_ch = [
            c for c in CHONDRO_FCL_COLUMN.values()
            if c not in fields
        ]
        if missing_ch:
            raise ValueError(
                f"{path}: missing Chondrometrics regional dABp columns: {missing_ch}"
            )

        for row in rd:
            if not str(row.get("CMT-ID", "")).strip():
                continue
            case = normalize_case_from_cmt(row["CMT-ID"])
            if case in rows_by_case:
                raise ValueError(f"{path}: duplicate case/CMT-ID: {case}")
            rows_by_case[case] = row

            sid_raw = row.get("SubjectID", "")
            if str(sid_raw).strip():
                case_to_subject[case] = int(round(float(sid_raw)))

            for roi, col in CHONDRO_FCL_COLUMN.items():
                v = num(row.get(col))
                # POMA dABp is already in percent (e.g. 60.256, 100.0).
                if finite(v):
                    chondro[(case, roi)] = float(v)

    return case_to_subject, chondro, rows_by_case


def read_python_fcl(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty MorphQuant CSV: {path}")

    hdr = [str(x).strip() for x in rows[0][1:]]
    for row in rows[1:]:
        if row and str(row[0]).strip().lower() == "fcl":
            vals: Dict[str, float] = {}
            for roi, raw in zip(hdr, row[1:]):
                v = num(raw)
                if roi in ROI_NAMES and finite(v):
                    vals[roi] = float(v)
            return vals
    raise ValueError(f"No FCL row in {path}")


def find_morph_file(case_dir: Path, morph_name: str) -> Optional[Path]:
    p = case_dir / morph_name
    if p.is_file():
        return p
    return None


# ---------------------------------------------------------------------------
# pHR and descriptive stats
# ---------------------------------------------------------------------------

def hit(pred_percent: float, gt_percent: float, tolerance: float) -> bool:
    # Eq. 26: G in [P-R, P+R] == |P-G| <= R
    return abs(float(pred_percent) - float(gt_percent)) <= float(tolerance) + 1e-12


def phr(pred: Sequence[float], gt: Sequence[float], tolerance: float):
    if len(pred) != len(gt):
        raise ValueError("pred/gt length mismatch")
    n = len(pred)
    hits = sum(hit(p, g, tolerance) for p, g in zip(pred, gt))
    return hits, n, hits / n if n else float("nan")


def describe_predictions(values: Sequence[float], gt_value: float):
    a = np.asarray([x for x in values if finite(x)], dtype=float)
    if len(a) == 0:
        return {
            "n": 0, "mean": float("nan"), "sd": float("nan"),
            "median": float("nan"), "p10": float("nan"),
            "p25": float("nan"), "p75": float("nan"),
            "p90": float("nan"), "mae_to_grade_target": float("nan"),
            "rmse_to_grade_target": float("nan"),
        }
    d = a - float(gt_value)
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "mae_to_grade_target": float(np.mean(np.abs(d))),
        "rmse_to_grade_target": float(np.sqrt(np.mean(d * d))),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CartiMorph paper-style FCL pseudo hit rate (pHR) analysis."
    )
    ap.add_argument("--root", type=Path, default=Path("registrations"))
    ap.add_argument(
        "--mapping", type=Path,
        default=Path("Chondrometrics_AllMetrics_with_CMT-ID.csv"),
        help="POMA/Chondrometrics CSV; also provides CMT-ID -> SubjectID mapping",
    )
    ap.add_argument("--rater1", type=Path, default=Path("FCLgrading_rater1.xlsx"))
    ap.add_argument("--rater2", type=Path, default=Path("FCLgrading_rater2.xlsx"))
    ap.add_argument("--rater3", type=Path, default=Path("FCLgrading_rater3.xlsx"))
    ap.add_argument(
        "--morph-name", default="MorphQuant.csv",
        help="Python MorphQuant filename inside each case directory",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("human_fcl_v16/paper_phr")
    )
    ap.add_argument(
        "--tolerances",
        nargs="*",
        type=float,
        default=None,
        help="Optional explicit tolerance list in percentage points. "
             "Default is every integer from 5 through 100.",
    )
    args = ap.parse_args()

    for p in [args.mapping, args.rater1, args.rater2, args.rater3]:
        if not p.is_file():
            raise FileNotFoundError(p)
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)

    tolerances = (
        sorted(set(float(x) for x in args.tolerances))
        if args.tolerances
        else [float(x) for x in range(5, 101)]
    )
    if any(t < 0 for t in tolerances):
        ap.error("tolerances must be >=0")

    args.out.mkdir(parents=True, exist_ok=True)

    case_to_subject, chondro, source_rows = read_mapping_and_chondro(args.mapping)
    raters = [
        ("RATER1", read_rater(args.rater1)),
        ("RATER2", read_rater(args.rater2)),
        ("RATER3", read_rater(args.rater3)),
    ]

    # Build one case/ROI base table.
    base = []
    excluded_cases = []
    n_python_cases = 0

    for case in sorted(case_to_subject):
        sid = case_to_subject[case]
        case_dir = args.root / case
        morph = find_morph_file(case_dir, args.morph_name)
        if morph is None:
            excluded_cases.append({
                "case": case,
                "subject_id": sid,
                "reason": f"missing {args.morph_name}",
                "expected_path": str(case_dir / args.morph_name),
            })
            continue
        try:
            pyfcl = read_python_fcl(morph)
        except Exception as e:
            excluded_cases.append({
                "case": case,
                "subject_id": sid,
                "reason": f"MorphQuant read error: {e}",
                "expected_path": str(morph),
            })
            continue

        n_python_cases += 1
        for roi in ROI_NAMES:
            p = pyfcl.get(roi, float("nan"))
            c = chondro.get((case, roi), float("nan"))
            base.append({
                "case": case,
                "subject_id": sid,
                "roi": roi,
                "python_fcl_percent": p,
                "chondrometrics_fcl_percent": c,
                "is_common16": int(roi in COMMON16),
            })

    write_csv(args.out / "excluded_cases.csv", excluded_cases)

    # Long observation table: one row per case/ROI/rater.
    records_long = []
    for b in base:
        case = b["case"]
        sid = int(b["subject_id"])
        roi = b["roi"]
        for rname, rmap in raters:
            grade = rmap.get(sid, {}).get(roi)
            if grade is None:
                continue
            gt = 10.0 * int(grade)
            records_long.append({
                **b,
                "rater": rname,
                "human_grade": int(grade),
                "human_target_percent": gt,
                "python_abs_error_pp": (
                    abs(float(b["python_fcl_percent"]) - gt)
                    if finite(b["python_fcl_percent"]) else float("nan")
                ),
                "chondrometrics_abs_error_pp": (
                    abs(float(b["chondrometrics_fcl_percent"]) - gt)
                    if finite(b["chondrometrics_fcl_percent"]) else float("nan")
                ),
            })

    write_csv(args.out / "records_long.csv", records_long)

    # Index records by rater.
    by_rater = defaultdict(list)
    for r in records_long:
        by_rater[r["rater"]].append(r)

    curve_rows = []
    key_rows = []
    distribution_rows = []

    # These are the two scientifically valid scopes.
    scopes = ["COMMON16_PAIRED", "ALL20_PYTHON"]

    # Keep per-rater curve values so we can emit arithmetic mean + SD.
    per_curve = defaultdict(list)  # (scope, method, tol) -> [pHR rater1..3]

    for rname, _ in raters:
        rr = by_rater[rname]

        common_paired = [
            r for r in rr
            if r["roi"] in COMMON16
            and finite(r["python_fcl_percent"])
            and finite(r["chondrometrics_fcl_percent"])
        ]
        all20_py = [
            r for r in rr
            if finite(r["python_fcl_percent"])
        ]

        scope_method_records = [
            ("COMMON16_PAIRED", "PYTHON_V16", common_paired, "python_fcl_percent"),
            ("COMMON16_PAIRED", "CHONDROMETRICS", common_paired, "chondrometrics_fcl_percent"),
            ("ALL20_PYTHON", "PYTHON_V16", all20_py, "python_fcl_percent"),
        ]

        for scope, method, rows, pred_col in scope_method_records:
            pred = [float(r[pred_col]) for r in rows]
            gt = [float(r["human_target_percent"]) for r in rows]

            for t in tolerances:
                hits, n, val = phr(pred, gt, t)
                row = {
                    "scope": scope,
                    "method": method,
                    "rater": rname,
                    "tolerance_pp": t,
                    "hits": hits,
                    "n": n,
                    "pHR": val,
                }
                curve_rows.append(row)
                per_curve[(scope, method, t)].append(val)
                if t in {5.0, 10.0, 15.0, 20.0, 25.0, 30.0}:
                    key_rows.append(row.copy())

            # Fig-16a-like prediction distributions by manual grade.
            for g in range(11):
                g_rows = [r for r in rows if int(r["human_grade"]) == g]
                vals = [float(r[pred_col]) for r in g_rows if finite(r[pred_col])]
                stats = describe_predictions(vals, 10.0 * g)
                distribution_rows.append({
                    "scope": scope,
                    "method": method,
                    "rater": rname,
                    "human_grade": g,
                    "human_target_percent": 10.0 * g,
                    **stats,
                })

    # Mean and SD of the 3 independent rater analyses.
    aggregate_curve_rows = []
    aggregate_key_rows = []
    for (scope, method, t), vals in sorted(per_curve.items()):
        a = [float(x) for x in vals if finite(x)]
        row = {
            "scope": scope,
            "method": method,
            "rater": "RATER_MEAN",
            "tolerance_pp": t,
            "hits": "",
            "n": "",
            "pHR": float(np.mean(a)) if a else float("nan"),
            "pHR_rater_sd": (
                float(np.std(a, ddof=1)) if len(a) > 1
                else (0.0 if len(a) == 1 else float("nan"))
            ),
            "n_raters": len(a),
        }
        aggregate_curve_rows.append(row)
        if t in {5.0, 10.0, 15.0, 20.0, 25.0, 30.0}:
            aggregate_key_rows.append(row.copy())

    # Write curve with compatible union of columns.
    all_curve = []
    for r in curve_rows:
        all_curve.append({
            **r,
            "pHR_rater_sd": "",
            "n_raters": "",
        })
    all_curve.extend(aggregate_curve_rows)

    all_key = []
    for r in key_rows:
        all_key.append({
            **r,
            "pHR_rater_sd": "",
            "n_raters": "",
        })
    all_key.extend(aggregate_key_rows)

    write_csv(args.out / "phr_curve.csv", all_curve)
    write_csv(args.out / "phr_key_tolerances.csv", all_key)
    write_csv(args.out / "distribution_by_grade.csv", distribution_rows)

    # Compact one-row-per-rater summary at 10%.
    rater_summary = []
    for rname, _ in raters:
        common_py = next(
            r for r in curve_rows
            if r["scope"] == "COMMON16_PAIRED"
            and r["method"] == "PYTHON_V16"
            and r["rater"] == rname
            and r["tolerance_pp"] == 10.0
        )
        common_ch = next(
            r for r in curve_rows
            if r["scope"] == "COMMON16_PAIRED"
            and r["method"] == "CHONDROMETRICS"
            and r["rater"] == rname
            and r["tolerance_pp"] == 10.0
        )
        all20 = next(
            r for r in curve_rows
            if r["scope"] == "ALL20_PYTHON"
            and r["method"] == "PYTHON_V16"
            and r["rater"] == rname
            and r["tolerance_pp"] == 10.0
        )
        rater_summary.append({
            "rater": rname,
            "common16_n": common_py["n"],
            "python_common16_pHR_at10": common_py["pHR"],
            "chondrometrics_common16_pHR_at10": common_ch["pHR"],
            "python_minus_chondro_pHR_at10": (
                float(common_py["pHR"]) - float(common_ch["pHR"])
            ),
            "all20_python_n": all20["n"],
            "python_all20_pHR_at10": all20["pHR"],
        })

    write_csv(args.out / "phr_rater_summary.csv", rater_summary)

    # Human-readable summary.
    mean_common_py = float(np.mean([
        r["python_common16_pHR_at10"] for r in rater_summary
    ]))
    mean_common_ch = float(np.mean([
        r["chondrometrics_common16_pHR_at10"] for r in rater_summary
    ]))
    mean_all20_py = float(np.mean([
        r["python_all20_pHR_at10"] for r in rater_summary
    ]))

    lines = [
        "CARTIMORPH PAPER-STYLE FCL PSEUDO HIT RATE (pHR)",
        "================================================",
        "",
        "Paper method reproduced:",
        "  human continuous target = manual grade * 10 percentage points",
        "  hit at tolerance R iff |prediction - target| <= R",
        "  each of 3 raters analysed independently",
        "",
        f"Python MorphQuant filename: {args.morph_name}",
        f"mapped cases in source CSV: {len(case_to_subject)}",
        f"Python cases successfully read: {n_python_cases}",
        f"excluded Python cases: {len(excluded_cases)}",
        f"long human observations: {len(records_long)}",
        "",
        "PRIMARY FAIR COMPARISON: COMMON16_PAIRED",
        "  Uses only the 16 ROIs with direct Chondrometrics dABp counterparts.",
        "  Python and Chondrometrics use exactly the same case/ROI/rater observations.",
        "",
    ]

    for r in rater_summary:
        lines += [
            f"{r['rater']}: n={r['common16_n']}",
            f"  Python v16 pHR@10%        = {fmt(r['python_common16_pHR_at10'],4)}",
            f"  Chondrometrics pHR@10%    = {fmt(r['chondrometrics_common16_pHR_at10'],4)}",
            f"  Python - Chondrometrics   = {fmt(r['python_minus_chondro_pHR_at10'],4)}",
        ]

    lines += [
        "",
        "3-RATER ARITHMETIC MEAN @10%",
        f"  Python v16 COMMON16       = {fmt(mean_common_py,4)}",
        f"  Chondrometrics COMMON16   = {fmt(mean_common_ch,4)}",
        f"  Difference                = {fmt(mean_common_py - mean_common_ch,4)}",
        "",
        "SECONDARY PYTHON-ONLY ALL20",
    ]

    for r in rater_summary:
        lines.append(
            f"  {r['rater']}: n={r['all20_python_n']} "
            f"pHR@10%={fmt(r['python_all20_pHR_at10'],4)}"
        )

    lines += [
        f"  3-rater mean ALL20        = {fmt(mean_all20_py,4)}",
        "",
        "PAPER REFERENCE",
        "  CartiMorph paper reports approximately pHR = 0.85 at 10% tolerance.",
        "  This is an approximate visual/reported reference, not a hard pass/fail cutoff.",
        f"  Python COMMON16 mean minus 0.85 = {fmt(mean_common_py - PAPER_APPROX_PHR_AT_10,4)}",
        f"  Python ALL20 mean minus 0.85    = {fmt(mean_all20_py - PAPER_APPROX_PHR_AT_10,4)}",
        "",
        "IMPORTANT SCOPE NOTE",
        "  The supplied POMA table has no direct regional dABp for",
        "  aMFC/pMFC/aLFC/pLFC. No Chondrometrics values were invented for them.",
        "  Use COMMON16_PAIRED for the Python-vs-Chondrometrics comparison.",
        "",
        "FILES",
        "  phr_curve.csv              full tolerance curve (default R=5..100)",
        "  phr_key_tolerances.csv     5/10/15/20/25/30% subset",
        "  phr_rater_summary.csv      compact @10% comparison",
        "  distribution_by_grade.csv Fig-16a-like distributions",
        "  records_long.csv           observation-level audit table",
        "  excluded_cases.csv         missing/bad MorphQuant cases",
    ]

    summary_path = args.out / "paper_phr_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved under: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
