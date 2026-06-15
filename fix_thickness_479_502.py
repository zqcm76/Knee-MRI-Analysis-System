"""
fix_thickness_479_502.py — 补跑 479-502 的厚度测量

只更新 thickness 相关列，保留原有的 volume/area/dABp。
"""

import sys
import time
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mesh_utils import (
    mask_to_mesh,
    bwareaopen_3d,
)
from thickness import (
    estimate_surface_normals,
    smooth_surface_normals,
    thickness_map_sn_fast,
)


BASE_DIR = r"E:\OAIZIB-CM\nnUNet_raw\Dataset001_KneeSeg"
TEST_INFO = r"E:\OAIZIB-CM\info\subInfo_test.xlsx"
OUTPUT_DIR = r"E:\OAIZIB-CM\measure\results"


def measure_thickness_only(mask_3d, voxel_size, sn_n_neighbors=10, thickness_depth=20.0):
    """Measure thickness for one region. Returns dict with thickness fields."""
    result = {
        'thickness_mean_mm': 0.0,
        'thickness_std_mm': 0.0,
        'thickness_min_mm': 0.0,
        'thickness_max_mm': 0.0,
        'thickness_median_mm': 0.0,
        'thickness_n_measured': 0,
    }

    n_voxels = int(np.sum(mask_3d))
    if n_voxels == 0:
        return result

    mask_clean = bwareaopen_3d(mask_3d, min_size=10, connectivity=26)
    if np.sum(mask_clean) == 0:
        return result

    fv = mask_to_mesh(mask_clean)
    verts = fv['vertices'].astype(np.float64)
    faces = fv['faces']

    if len(faces) == 0 or len(verts) == 0:
        return result

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


def fix_case(case_name, seg_path):
    """Recompute thickness for one case and update its CSV."""
    csv_path = Path(OUTPUT_DIR) / f"{case_name}_report.csv"
    if not csv_path.exists():
        print(f"  CSV not found, skipping")
        return

    # Load existing atlas (saved by batch_measure.py)
    atlas_path = Path(OUTPUT_DIR) / f"{case_name}_atlas.nii.gz"
    if not atlas_path.exists():
        print(f"  Atlas not found, skipping")
        return
    atlas = nib.load(str(atlas_path)).get_fdata().astype(np.int32)

    # Load segmentation
    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata().astype(np.uint8)
    voxel_size = np.array(seg_img.header.get_zooms()[:3], dtype=np.float64)

    # Extract masks
    mask_fc = (seg == 3).astype(np.uint8)
    mask_mtc = (seg == 4).astype(np.uint8)
    mask_ltc = (seg == 5).astype(np.uint8)
    mask_cart = ((seg == 3) | (seg == 4) | (seg == 5)).astype(np.uint8)

    # Read existing CSV
    df = pd.read_csv(csv_path)

    # Update thickness for each row
    for idx, row in df.iterrows():
        region_name = row['region_name']

        if region_name == 'whole':
            mask = mask_cart
        elif region_name == 'FC_whole':
            mask = mask_fc
        elif region_name == 'TC_whole':
            mask = (mask_mtc | mask_ltc).astype(np.uint8)
        else:
            roi_code = row['roi_code']
            mask = (atlas == roi_code).astype(np.uint8)

        thick = measure_thickness_only(mask, voxel_size)
        for k, v in thick.items():
            df.at[idx, k] = v

    # Save updated CSV
    df.to_csv(csv_path, index=False)

    # Print summary
    whole = df[df['region_name'] == 'whole'].iloc[0]
    print(f"  thickness_mean={whole['thickness_mean_mm']:.2f}mm, "
          f"n_measured={int(whole['thickness_n_measured'])}")


def main():
    df = pd.read_excel(TEST_INFO)

    # Build case list for 479-502
    target_ids = list(range(479, 503))
    cases = []
    for _, row in df.iterrows():
        cid = int(row['CMT-ID'])
        if cid in target_ids:
            pid = f"oaizib_{cid:03d}"
            img_path = Path(BASE_DIR) / "labelsTs" / f"{pid}.nii.gz"
            if not img_path.exists():
                img_path = Path(BASE_DIR) / "labelsTr" / f"{pid}.nii.gz"
            if img_path.exists():
                cases.append((pid, img_path))

    print(f"Fixing thickness for {len(cases)} cases (479-502)")
    print()

    t_start = time.time()
    for i, (pid, seg_path) in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {pid}")
        t0 = time.time()
        try:
            fix_case(pid, seg_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        print(f"  Time: {time.time()-t0:.1f}s")

    print(f"\nDone! Total: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
