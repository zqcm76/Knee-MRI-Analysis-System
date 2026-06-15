"""
compare_pred_vs_gt.py — 预测 mask vs 金标准 形态学指标对比 (多进程版)

对每个测试集 case 分别测量 predictions_2d 和 labelsTs 的形态学指标：
  - 体积 (mm³)
  - 表面积 (mm²)
  - 厚度 (mean±std, mm)
  - dABp (%, 需骨配准结果)

计算 pred 与 GT 的差值和相对误差，按 KL 分级汇总。

用法:
    python compare_pred_vs_gt.py                   # 默认10进程
    python compare_pred_vs_gt.py --workers 8       # 指定进程数
    python compare_pred_vs_gt.py --no-thickness    # 跳过厚度(更快)
"""

import os
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))

from parcellation import (
    volume_parcellation_fc,
    volume_parcellation_tc,
    ALL_REGION_NAMES,
)
from mesh_utils import (
    mask_to_mesh,
    calculate_tri_mesh_area,
    bwareaopen_3d,
)
from thickness import (
    estimate_surface_normals,
    smooth_surface_normals,
    thickness_map_sn_fast,
)
from dabp import compute_dabp_from_pseudo_healthy

# ============================================================
# Paths
# ============================================================

BASE_DIR = r"E:\OAIZIB-CM\nnUNet_raw\Dataset001_KneeSeg"
PRED_DIR = os.path.join(BASE_DIR, "Dataset001_MRI_results")
GT_DIR = os.path.join(BASE_DIR, "labelsTs")
REG_DIR = r"E:\OAIZIB-CM\unigradicon_pipeline\results\bone_single_batch"
TEST_INFO = r"E:\OAIZIB-CM\info\subInfo_test.xlsx"
OUTPUT_DIR = r"E:\OAIZIB-CM\measure\results_3d_vs_gt"


# ============================================================
# Morphological measurement
# ============================================================

def measure_single_region(mask_3d, voxel_size, region_name='Cartilage',
                          sn_n_neighbors=10, thickness_depth=20.0,
                          do_thickness=True):
    """Measure volume, surface area, thickness for one region."""
    n_voxels = int(np.sum(mask_3d))
    voxel_vol = float(np.prod(voxel_size))

    result = {
        'region': region_name,
        'n_voxels': n_voxels,
        'volume_mm3': n_voxels * voxel_vol,
        'surface_area_mm2': 0.0,
        'thickness_mean_mm': 0.0,
        'thickness_std_mm': 0.0,
        'thickness_min_mm': 0.0,
        'thickness_max_mm': 0.0,
        'thickness_median_mm': 0.0,
        'thickness_n_measured': 0,
    }

    if n_voxels == 0:
        return result

    mask_clean = bwareaopen_3d(mask_3d, min_size=10, connectivity=26)
    if np.sum(mask_clean) == 0:
        return result

    fv = mask_to_mesh(mask_clean)
    verts = fv['vertices'].astype(np.float64)
    faces = fv['faces']

    if len(faces) == 0:
        return result

    face_areas = calculate_tri_mesh_area(verts, faces)
    result['surface_area_mm2'] = float(np.sum(face_areas))

    if do_thickness and len(verts) > 0:
        sn = estimate_surface_normals(verts, n_neighbors=sn_n_neighbors)
        sn_smooth = smooth_surface_normals(sn, verts, n_neighbors=sn_n_neighbors)
        tm = thickness_map_sn_fast(verts, sn_smooth, fv, depth=thickness_depth)
        thick_vals = tm[:, 3]
        thick_pos = thick_vals[thick_vals > 0]

        if len(thick_pos) > 0:
            result['thickness_mean_mm'] = float(np.mean(thick_pos))
            result['thickness_std_mm'] = float(np.std(thick_pos))
            result['thickness_min_mm'] = float(np.min(thick_pos))
            result['thickness_max_mm'] = float(np.max(thick_pos))
            result['thickness_median_mm'] = float(np.median(thick_pos))
            result['thickness_n_measured'] = int(len(thick_pos))

    return result


def measure_case(seg_path, do_thickness=True):
    """Measure one case: morphology + dABp, return region rows."""
    seg_path = Path(seg_path)
    case_name = seg_path.stem.replace('.nii', '')
    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata().astype(np.uint8)
    voxel_size = np.array(seg_img.header.get_zooms()[:3], dtype=np.float64)
    img_size = seg.shape

    mask_fc = (seg == 3).astype(np.uint8)
    mask_mtc = (seg == 4).astype(np.uint8)
    mask_ltc = (seg == 5).astype(np.uint8)
    mask_cart = ((seg == 3) | (seg == 4) | (seg == 5)).astype(np.uint8)
    mask_bone = ((seg == 1) | (seg == 2)).astype(np.uint8)

    atlas_fc = volume_parcellation_fc(mask_fc, knee_side='right',
                                       cc_percentage=0.6, img_size=img_size)
    atlas_tc = volume_parcellation_tc(mask_mtc, mask_ltc, knee_side='right',
                                       voxel_size=voxel_size, img_size=img_size)
    atlas = np.where(atlas_tc > 0, atlas_tc, atlas_fc)

    rows = []

    whole = measure_single_region(mask_cart, voxel_size, 'whole', do_thickness=do_thickness)
    rows.append({'roi_code': 0, 'region_name': 'whole', **whole})

    fc_whole = measure_single_region(mask_fc, voxel_size, 'FC_whole', do_thickness=do_thickness)
    rows.append({'roi_code': 0, 'region_name': 'FC_whole', **fc_whole})

    mask_tc = (mask_mtc | mask_ltc).astype(np.uint8)
    tc_whole = measure_single_region(mask_tc, voxel_size, 'TC_whole', do_thickness=do_thickness)
    rows.append({'roi_code': 0, 'region_name': 'TC_whole', **tc_whole})

    for roi_code in sorted(ALL_REGION_NAMES.keys()):
        if roi_code == 0:
            continue
        region_name = ALL_REGION_NAMES[roi_code]
        region_mask = (atlas == roi_code).astype(np.uint8)
        if np.sum(region_mask) == 0:
            continue
        reg_result = measure_single_region(region_mask, voxel_size, region_name, do_thickness=do_thickness)
        rows.append({'roi_code': roi_code, 'region_name': region_name, **reg_result})

    # dABp
    fused_path = Path(REG_DIR) / case_name / "fused_mask_weighted.nii.gz"
    if fused_path.exists():
        fused = nib.load(str(fused_path)).get_fdata()
        pseudo = (fused > 0).astype(np.uint8)
        region_codes = [c for c in sorted(ALL_REGION_NAMES.keys()) if c > 0]
        dabp_dict, _ = compute_dabp_from_pseudo_healthy(
            pseudo, mask_cart, mask_bone, atlas, voxel_size, img_size, region_codes
        )
        for row in rows:
            name = row['region_name']
            if name == 'whole':
                row['dabp_percent'] = dabp_dict.get(0, 0.0)
            elif name in ('FC_whole', 'TC_whole'):
                row['dabp_percent'] = 0.0
            else:
                row['dabp_percent'] = dabp_dict.get(row['roi_code'], 0.0)
    else:
        for row in rows:
            row['dabp_percent'] = None

    return rows


# ============================================================
# Worker: process one case (pred + GT)
# ============================================================

def process_one_case(args):
    """Worker function for multiprocessing."""
    pid, kl, pred_path, gt_path, do_thickness = args
    t0 = time.time()
    try:
        pred_rows = measure_case(pred_path, do_thickness=do_thickness)
        gt_rows = measure_case(gt_path, do_thickness=do_thickness)

        pred_dict = {r['region_name']: r for r in pred_rows}
        gt_dict = {r['region_name']: r for r in gt_rows}

        case_rows = []
        all_regions = set(list(pred_dict.keys()) + list(gt_dict.keys()))
        for region_name in sorted(all_regions):
            p = pred_dict.get(region_name, {})
            g = gt_dict.get(region_name, {})

            vol_p = p.get('volume_mm3', 0)
            vol_g = g.get('volume_mm3', 0)
            area_p = p.get('surface_area_mm2', 0)
            area_g = g.get('surface_area_mm2', 0)
            thick_p = p.get('thickness_mean_mm', 0)
            thick_g = g.get('thickness_mean_mm', 0)
            dabp_p = p.get('dabp_percent')
            dabp_g = g.get('dabp_percent')

            row = {
                'case_name': pid,
                'kl_grade': kl,
                'region_name': region_name,
                'pred_volume_mm3': vol_p,
                'pred_surface_area_mm2': area_p,
                'pred_thickness_mean_mm': thick_p,
                'pred_thickness_std_mm': p.get('thickness_std_mm', 0),
                'pred_dabp_percent': dabp_p,
                'gt_volume_mm3': vol_g,
                'gt_surface_area_mm2': area_g,
                'gt_thickness_mean_mm': thick_g,
                'gt_thickness_std_mm': g.get('thickness_std_mm', 0),
                'gt_dabp_percent': dabp_g,
                'diff_volume_mm3': vol_p - vol_g,
                'diff_surface_area_mm2': area_p - area_g,
                'diff_thickness_mean_mm': thick_p - thick_g,
                'diff_dabp_percent': (dabp_p - dabp_g) if (dabp_p is not None and dabp_g is not None) else None,
                'rel_err_volume_pct': abs(vol_p - vol_g) / max(vol_g, 1e-6) * 100,
                'rel_err_area_pct': abs(area_p - area_g) / max(area_g, 1e-6) * 100,
                'rel_err_thickness_pct': abs(thick_p - thick_g) / max(thick_g, 1e-6) * 100,
            }
            case_rows.append(row)

        elapsed = time.time() - t0
        return case_rows, pid, kl, elapsed, None

    except Exception as e:
        elapsed = time.time() - t0
        return [], pid, kl, elapsed, str(e)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pred vs GT morphological comparison (parallel)")
    parser.add_argument("--no-thickness", action="store_true",
                        help="Skip thickness measurement (much faster)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel processes (default: 10)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test patient list
    df = pd.read_excel(TEST_INFO)
    test_cases = []
    for _, row in df.iterrows():
        cid = int(row['CMT-ID'])
        pid = f"oaizib_{cid:03d}"
        kl = row['KLGrade']
        if pd.isna(kl):
            continue

        pred_path = Path(PRED_DIR) / f"{pid}.nii.gz"
        gt_path = Path(GT_DIR) / f"{pid}.nii.gz"
        if pred_path.exists() and gt_path.exists():
            test_cases.append((pid, int(kl), pred_path, gt_path, not args.no_thickness))

    print(f"Test cases: {len(test_cases)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print(f"Thickness: {'ON' if not args.no_thickness else 'OFF'}\n")

    # Run parallel
    t_total_start = time.time()
    all_rows = []
    done = 0

    with Pool(processes=args.workers) as pool:
        for case_rows, pid, kl, elapsed, error in pool.imap_unordered(process_one_case, test_cases):
            done += 1
            if error:
                print(f"[{done}/{len(test_cases)}] {pid} (KL={kl}) ERROR: {error} ({elapsed:.0f}s)")
            else:
                all_rows.extend(case_rows)
                print(f"[{done}/{len(test_cases)}] {pid} (KL={kl}) done ({elapsed:.0f}s)")

    # Save results
    t_total = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        csv_path = output_dir / "batch_comparison_summary.csv"
        df_all.to_csv(csv_path, index=False)
        print(f"\nFull comparison saved: {csv_path}")

        # Summary by KL
        print(f"\n{'='*80}")
        print("WHOLE CARTILAGE: Pred vs GT by KL Grade")
        print(f"{'='*80}")

        whole_rows = [r for r in all_rows if r['region_name'] == 'whole']
        df_whole = pd.DataFrame(whole_rows)

        for kl in sorted(df_whole['kl_grade'].unique()):
            kl_data = df_whole[df_whole['kl_grade'] == kl]
            n = len(kl_data)
            print(f"\nKL={kl} (n={n}):")
            print(f"  Volume:  Pred={kl_data['pred_volume_mm3'].mean():.0f}±{kl_data['pred_volume_mm3'].std():.0f}  "
                  f"GT={kl_data['gt_volume_mm3'].mean():.0f}±{kl_data['gt_volume_mm3'].std():.0f}  "
                  f"Diff={kl_data['diff_volume_mm3'].mean():.0f}±{kl_data['diff_volume_mm3'].std():.0f} mm³")
            print(f"  Area:    Pred={kl_data['pred_surface_area_mm2'].mean():.0f}±{kl_data['pred_surface_area_mm2'].std():.0f}  "
                  f"GT={kl_data['gt_surface_area_mm2'].mean():.0f}±{kl_data['gt_surface_area_mm2'].std():.0f}  "
                  f"Diff={kl_data['diff_surface_area_mm2'].mean():.0f}±{kl_data['diff_surface_area_mm2'].std():.0f} mm²")
            print(f"  Thick:   Pred={kl_data['pred_thickness_mean_mm'].mean():.2f}±{kl_data['pred_thickness_mean_mm'].std():.2f}  "
                  f"GT={kl_data['gt_thickness_mean_mm'].mean():.2f}±{kl_data['gt_thickness_mean_mm'].std():.2f}  "
                  f"Diff={kl_data['diff_thickness_mean_mm'].mean():.2f}±{kl_data['diff_thickness_mean_mm'].std():.2f} mm")
            print(f"  RelErr:  Vol={kl_data['rel_err_volume_pct'].mean():.1f}%  "
                  f"Area={kl_data['rel_err_area_pct'].mean():.1f}%  "
                  f"Thick={kl_data['rel_err_thickness_pct'].mean():.1f}%")

            dabp_valid = kl_data.dropna(subset=['pred_dabp_percent', 'gt_dabp_percent'])
            if len(dabp_valid) > 0:
                print(f"  dABp:    Pred={dabp_valid['pred_dabp_percent'].mean():.2f}±{dabp_valid['pred_dabp_percent'].std():.2f}  "
                      f"GT={dabp_valid['gt_dabp_percent'].mean():.2f}±{dabp_valid['gt_dabp_percent'].std():.2f}  "
                      f"Diff={dabp_valid['diff_dabp_percent'].mean():.2f}±{dabp_valid['diff_dabp_percent'].std():.2f}%")

        # Per-region summary
        print(f"\n{'='*80}")
        print("PER-REGION AVERAGE DIFFERENCE (Pred - GT)")
        print(f"{'='*80}")
        print(f"{'Region':<10} {'ΔVol(mm³)':>12} {'ΔArea(mm²)':>12} {'ΔThick(mm)':>12} {'RelErrVol%':>12}")
        print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for region_name in ['whole', 'FC_whole', 'TC_whole']:
            reg_data = [r for r in all_rows if r['region_name'] == region_name]
            if reg_data:
                df_r = pd.DataFrame(reg_data)
                print(f"{region_name:<10} "
                      f"{df_r['diff_volume_mm3'].mean():>12.1f} "
                      f"{df_r['diff_surface_area_mm2'].mean():>12.1f} "
                      f"{df_r['diff_thickness_mean_mm'].mean():>12.2f} "
                      f"{df_r['rel_err_volume_pct'].mean():>12.1f}")

    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    main()
