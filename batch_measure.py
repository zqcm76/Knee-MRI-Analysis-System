"""
batch_measure.py — 批量测量 103 例测试集

对每个 case 测量：
- 20 分区 + 整体
- 体积 (mm³)
- 表面积 (mm²)
- 厚度 (mean±std, mm)
- dABp (来自骨骼配准结果, %)

用法:
    python batch_measure.py
    python batch_measure.py --no-thickness
"""

import os
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

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
REG_DIR = r"E:\OAIZIB-CM\unigradicon_pipeline\results\bone_single_batch"
TEST_INFO = r"E:\OAIZIB-CM\info\subInfo_test.xlsx"
OUTPUT_DIR = r"E:\OAIZIB-CM\measure\results"


# ============================================================
# Single region measurement
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

    # Clean mask
    mask_clean = bwareaopen_3d(mask_3d, min_size=10, connectivity=26)
    if np.sum(mask_clean) == 0:
        return result

    # Mesh
    fv = mask_to_mesh(mask_clean)
    verts = fv['vertices'].astype(np.float64)
    faces = fv['faces']

    if len(faces) == 0:
        return result

    # Surface area
    face_areas = calculate_tri_mesh_area(verts, faces)
    result['surface_area_mm2'] = float(np.sum(face_areas))

    # Thickness
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


# ============================================================
# Single case measurement
# ============================================================

def measure_case(seg_path, output_dir, do_thickness=True, verbose=True):
    """Measure one case: parcellation + volume + area + thickness + dABp."""
    seg_path = Path(seg_path)
    case_name = seg_path.stem.replace('.nii', '')

    if verbose:
        print(f"\n{'='*60}")
        print(f"Case: {case_name}")
        print(f"{'='*60}")

    # Load segmentation
    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata().astype(np.uint8)
    voxel_size = np.array(seg_img.header.get_zooms()[:3], dtype=np.float64)
    img_size = seg.shape

    # Extract masks
    mask_fc = (seg == 3).astype(np.uint8)
    mask_mtc = (seg == 4).astype(np.uint8)
    mask_ltc = (seg == 5).astype(np.uint8)
    mask_cart = ((seg == 3) | (seg == 4) | (seg == 5)).astype(np.uint8)
    mask_bone = ((seg == 1) | (seg == 2)).astype(np.uint8)

    if verbose:
        print(f"FC: {np.sum(mask_fc)}, MTC: {np.sum(mask_mtc)}, LTC: {np.sum(mask_ltc)}")

    # --- Parcellation ---
    if verbose:
        print("Parcellation...")

    atlas_fc = volume_parcellation_fc(mask_fc, knee_side='right',
                                       cc_percentage=0.6, img_size=img_size)
    atlas_tc = volume_parcellation_tc(mask_mtc, mask_ltc, knee_side='right',
                                       voxel_size=voxel_size, img_size=img_size)
    atlas = np.where(atlas_tc > 0, atlas_tc, atlas_fc)

    # Save atlas
    if output_dir:
        atlas_path = output_dir / f"{case_name}_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas, seg_img.affine), str(atlas_path))

    # --- Measure each region ---
    if verbose:
        print("Measuring regions...")

    rows = []

    # Whole cartilage
    whole = measure_single_region(mask_cart, voxel_size, 'whole',
                                   do_thickness=do_thickness)
    rows.append({'roi_code': 0, 'region_name': 'whole', **whole})

    # FC whole
    fc_whole = measure_single_region(mask_fc, voxel_size, 'FC_whole',
                                      do_thickness=do_thickness)
    rows.append({'roi_code': 0, 'region_name': 'FC_whole', **fc_whole})

    # TC whole
    mask_tc = (mask_mtc | mask_ltc).astype(np.uint8)
    tc_whole = measure_single_region(mask_tc, voxel_size, 'TC_whole',
                                      do_thickness=do_thickness)
    rows.append({'roi_code': 0, 'region_name': 'TC_whole', **tc_whole})

    # 20 subregions
    for roi_code in sorted(ALL_REGION_NAMES.keys()):
        if roi_code == 0:
            continue
        region_name = ALL_REGION_NAMES[roi_code]
        region_mask = (atlas == roi_code).astype(np.uint8)
        if np.sum(region_mask) == 0:
            continue

        reg_result = measure_single_region(region_mask, voxel_size, region_name,
                                            do_thickness=do_thickness)
        rows.append({'roi_code': roi_code, 'region_name': region_name, **reg_result})

    # --- dABp from bone-only registration ---
    if verbose:
        print("Computing dABp...")

    fused_path = Path(REG_DIR) / case_name / "fused_mask_weighted.nii.gz"
    if fused_path.exists():
        fused = nib.load(str(fused_path)).get_fdata()
        pseudo = (fused > 0).astype(np.uint8)

        region_codes = [c for c in sorted(ALL_REGION_NAMES.keys()) if c > 0]
        dabp_dict, _ = compute_dabp_from_pseudo_healthy(
            pseudo, mask_cart, mask_bone, atlas, voxel_size, img_size, region_codes
        )

        # Add dABp to rows
        for row in rows:
            name = row['region_name']
            if name == 'whole':
                row['dabp_percent'] = dabp_dict.get(0, 0.0)
            elif name == 'FC_whole' or name == 'TC_whole':
                row['dabp_percent'] = 0.0  # skip
            else:
                roi = row['roi_code']
                row['dabp_percent'] = dabp_dict.get(roi, 0.0)

        if verbose:
            print(f"  Whole dABp: {dabp_dict.get(0, 0):.2f}%")
    else:
        if verbose:
            print(f"  No registration result for {case_name}, skipping dABp")
        for row in rows:
            row['dabp_percent'] = None

    return rows


# ============================================================
# Batch processing
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Batch cartilage measurement")
    parser.add_argument("--no-thickness", action="store_true",
                        help="Skip thickness measurement (much faster)")
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

        # Find image
        img_path = Path(BASE_DIR) / "labelsTs" / f"{pid}.nii.gz"
        if not img_path.exists():
            img_path = Path(BASE_DIR) / "labelsTr" / f"{pid}.nii.gz"
        if img_path.exists():
            test_cases.append((pid, int(kl), img_path))

    print(f"Test cases: {len(test_cases)}")
    print(f"Output: {output_dir}")
    print(f"Thickness: {'ON' if not args.no_thickness else 'OFF'}")
    print()

    # Process each case
    all_rows = []
    t_total_start = time.time()

    for i, (pid, kl, seg_path) in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {pid} (KL={kl})")
        t0 = time.time()

        try:
            rows = measure_case(
                seg_path, output_dir,
                do_thickness=not args.no_thickness,
                verbose=False,
            )

            # Add case info to each row
            for row in rows:
                row['case_name'] = pid
                row['kl_grade'] = kl

            all_rows.extend(rows)

            # One-line summary
            whole = next((r for r in rows if r['region_name'] == 'whole'), None)
            if whole:
                print(f"  vol={whole['volume_mm3']:.1f}mm3, "
                      f"area={whole['surface_area_mm2']:.1f}mm2, "
                      f"thick={whole['thickness_mean_mm']:.2f}mm, "
                      f"dABp={whole.get('dabp_percent', 0):.1f}%")

            # Save per-case CSV
            case_df = pd.DataFrame(rows)
            case_csv = output_dir / f"{pid}_report.csv"
            case_df.to_csv(case_csv, index=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        t1 = time.time()
        print(f"  Time: {t1-t0:.1f}s")

    # Save batch summary
    t_total = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")

    if all_rows:
        batch_df = pd.DataFrame(all_rows)
        batch_csv = output_dir / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False)
        print(f"Batch summary: {batch_csv}")

        # Print summary table
        print(f"\n{'='*60}")
        print("SUMMARY BY KL GRADE")
        print(f"{'='*60}")

        whole_rows = batch_df[batch_df['region_name'] == 'whole']
        summary = whole_rows.groupby('kl_grade').agg({
            'volume_mm3': ['mean', 'std'],
            'surface_area_mm2': ['mean', 'std'],
            'thickness_mean_mm': ['mean', 'std'],
            'dabp_percent': ['mean', 'std'],
        }).round(2)
        print(summary.to_string())


if __name__ == '__main__':
    main()
