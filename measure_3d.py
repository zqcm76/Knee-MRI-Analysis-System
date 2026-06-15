"""
measure_3d.py — 测量 3D 预测 mask 的形态学指标 (多进程版)

指标：体积、表面积、厚度、dABp

用法:
    python measure_3d.py
    python measure_3d.py --workers 14
"""

import os
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

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
REG_DIR = r"E:\OAIZIB-CM\unigradicon_pipeline\results\bone_single_batch"
TEST_INFO = r"E:\OAIZIB-CM\info\subInfo_test.xlsx"
OUTPUT_DIR = r"E:\OAIZIB-CM\measure\results_3d"


# ============================================================
# Measurement
# ============================================================

def measure_single_region(mask_3d, voxel_size, region_name='Cartilage',
                          sn_n_neighbors=10, thickness_depth=20.0,
                          do_thickness=True):
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

    result['surface_area_mm2'] = float(np.sum(calculate_tri_mesh_area(verts, faces)))

    if do_thickness and len(verts) > 0:
        sn = estimate_surface_normals(verts, n_neighbors=sn_n_neighbors)
        sn_smooth = smooth_surface_normals(sn, verts, n_neighbors=sn_n_neighbors)
        tm = thickness_map_sn_fast(verts, sn_smooth, fv, depth=thickness_depth)
        thick_pos = tm[:, 3][tm[:, 3] > 0]
        if len(thick_pos) > 0:
            result['thickness_mean_mm'] = float(np.mean(thick_pos))
            result['thickness_std_mm'] = float(np.std(thick_pos))
            result['thickness_min_mm'] = float(np.min(thick_pos))
            result['thickness_max_mm'] = float(np.max(thick_pos))
            result['thickness_median_mm'] = float(np.median(thick_pos))
            result['thickness_n_measured'] = int(len(thick_pos))

    return result


def process_one_case(args):
    pid, kl, seg_path, do_thickness = args
    t0 = time.time()
    try:
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

        for row in rows:
            row['case_name'] = pid
            row['kl_grade'] = kl

        elapsed = time.time() - t0
        return rows, pid, kl, elapsed, None

    except Exception as e:
        elapsed = time.time() - t0
        return [], pid, kl, elapsed, str(e)


def main():
    parser = argparse.ArgumentParser(description="Measure 3D predictions")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--no-thickness", action="store_true")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(TEST_INFO)
    test_cases = []
    for _, row in df.iterrows():
        cid = int(row['CMT-ID'])
        pid = f"oaizib_{cid:03d}"
        kl = row['KLGrade']
        if pd.isna(kl):
            continue
        seg_path = Path(PRED_DIR) / f"{pid}.nii.gz"
        if seg_path.exists():
            test_cases.append((pid, int(kl), seg_path, not args.no_thickness))

    print(f"Cases: {len(test_cases)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print(f"Thickness: {'ON' if not args.no_thickness else 'OFF'}\n")

    t_start = time.time()
    all_rows = []
    done = 0

    with Pool(processes=args.workers) as pool:
        for rows, pid, kl, elapsed, error in pool.imap_unordered(process_one_case, test_cases):
            done += 1
            if error:
                print(f"[{done}/{len(test_cases)}] {pid} ERROR: {error} ({elapsed:.0f}s)")
            else:
                all_rows.extend(rows)
                print(f"[{done}/{len(test_cases)}] {pid} (KL={kl}) done ({elapsed:.0f}s)")

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total: {t_total:.0f}s ({t_total/60:.1f}min)")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        csv_path = output_dir / "batch_summary.csv"
        df_all.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Summary by KL
        print(f"\n{'='*60}")
        print("WHOLE CARTILAGE by KL Grade")
        print(f"{'='*60}")
        whole = df_all[df_all['region_name'] == 'whole']
        for kl in sorted(whole['kl_grade'].unique()):
            d = whole[whole['kl_grade'] == kl]
            print(f"KL={kl} (n={len(d)}): vol={d['volume_mm3'].mean():.0f}±{d['volume_mm3'].std():.0f}  "
                  f"area={d['surface_area_mm2'].mean():.0f}±{d['surface_area_mm2'].std():.0f}  "
                  f"thick={d['thickness_mean_mm'].mean():.2f}±{d['thickness_mean_mm'].std():.2f}")
            dabp = d.dropna(subset=['dabp_percent'])
            if len(dabp) > 0:
                print(f"  dABp={dabp['dabp_percent'].mean():.2f}±{dabp['dabp_percent'].std():.2f}%")


if __name__ == '__main__':
    main()
