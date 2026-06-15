"""
batch_measure_pred.py — 用预测 mask 跑形态学测量

与 batch_measure.py 相同流程，但使用 predictions_2d 的预测 mask。
结果保存到新文件夹，方便与金标准对比。
"""
import os
import sys
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
PRED_DIR = os.path.join(BASE_DIR, "predictions_2d")
REG_DIR = r"E:\OAIZIB-CM\unigradicon_pipeline\results\bone_single_batch"
TEST_INFO = r"E:\OAIZIB-CM\info\subInfo_test.xlsx"
OUTPUT_DIR = r"E:\OAIZIB-CM\measure\results_pred"


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

    # Surface area
    try:
        mesh = mask_to_mesh(mask_3d.astype(np.uint8))
        if mesh is not None and len(mesh.get('faces', [])) > 0:
            areas = calculate_tri_mesh_area(
                mesh['vertices'].astype(np.float64), mesh['faces']
            )
            result['surface_area_mm2'] = float(np.sum(areas))
    except Exception:
        pass

    # Thickness
    if do_thickness and n_voxels > 100:
        try:
            mesh = mask_to_mesh(mask_3d.astype(np.uint8))
            if mesh is not None and len(mesh.get('vertices', [])) > 10:
                verts = mesh['vertices'].astype(np.float64)
                faces = mesh['faces']
                sn = estimate_surface_normals(verts, n_neighbors=sn_n_neighbors)
                sn_smooth = smooth_surface_normals(verts, sn, faces)
                tmap = thickness_map_sn_fast(verts, sn_smooth, depth=thickness_depth)
                tmap = tmap[tmap > 0]
                if len(tmap) > 0:
                    result['thickness_mean_mm'] = float(np.mean(tmap))
                    result['thickness_std_mm'] = float(np.std(tmap))
                    result['thickness_min_mm'] = float(np.min(tmap))
                    result['thickness_max_mm'] = float(np.max(tmap))
                    result['thickness_median_mm'] = float(np.median(tmap))
                    result['thickness_n_measured'] = len(tmap)
        except Exception:
            pass

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
    parser = argparse.ArgumentParser(description="Batch measurement on predicted masks")
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

        # 使用 predictions_2d 的预测 mask
        pred_path = Path(PRED_DIR) / f"{pid}.nii.gz"
        if pred_path.exists():
            test_cases.append((pid, int(kl), pred_path))
        else:
            print(f"[WARN] 预测 mask 不存在: {pred_path}")

    print(f"Test cases: {len(test_cases)}")
    print(f"Output: {output_dir}")
    print(f"Thickness: {'ON' if not args.no_thickness else 'OFF'}\n")

    # Process each case
    all_rows = []
    for i, (pid, kl, seg_path) in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {pid} (KL={kl})")

        try:
            rows = measure_case(
                seg_path, output_dir,
                do_thickness=not args.no_thickness,
                verbose=True,
            )
            for row in rows:
                row['case_name'] = pid
                row['kl_grade'] = kl
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Save per-case reports
    print(f"\n{'='*60}")
    print("Saving reports...")
    for row in all_rows:
        case_name = row['case_name']
        report_path = output_dir / f"{case_name}_report.csv"
        case_rows = [r for r in all_rows if r['case_name'] == case_name]
        if case_rows:
            df_case = pd.DataFrame(case_rows)
            df_case.to_csv(report_path, index=False)

    # Save batch summary
    summary_path = output_dir / "batch_summary.csv"
    df_summary = pd.DataFrame(all_rows)
    df_summary.to_csv(summary_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY BY KL GRADE")
    print(f"{'='*60}")

    whole_rows = [r for r in all_rows if r['region_name'] == 'whole']
    for kl in sorted(set(r['kl_grade'] for r in whole_rows)):
        kl_rows = [r for r in whole_rows if r['kl_grade'] == kl]
        if kl_rows:
            vol_vals = [r['volume_mm3'] for r in kl_rows]
            dabp_vals = [r.get('dabp_percent', 0) for r in kl_rows if r.get('dabp_percent') is not None]
            print(f"KL={kl}: n={len(kl_rows)}, "
                  f"vol={np.mean(vol_vals):.0f}mm³, "
                  f"dABp={np.mean(dabp_vals):.2f}%" if dabp_vals else f"KL={kl}: n={len(kl_rows)}, vol={np.mean(vol_vals):.0f}mm³")

    print(f"\n结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
