"""
CartiMorph 20-Region Cartilage Morphological Measurement

Main entry point for automated cartilage morphological analysis.
Performs rule-based 20-region parcellation and measures:
    - Volume (mm³)
    - Surface area (mm²)
    - Thickness (mean ± std, mm)
    - dABp (denuded area bone percentage, %)

for both whole cartilage and 20 subregions.

Input:  NIfTI segmentation mask
Output: CSV report + optional NIfTI atlas

Usage:
    python cartimorph_measure.py <segmentation.nii.gz> [options]

    # Single case
    python cartimorph_measure.py case001.nii.gz --output results/

    # Batch
    python cartimorph_measure.py labelsTr/*.nii.gz --output results/

Label convention (OAIZIB dataset):
    1 = Femoral bone
    2 = Tibial bone
    3 = Femoral cartilage (FC)
    4 = Medial tibial cartilage (MTC)
    5 = Lateral tibial cartilage (LTC)

Author: Python implementation based on CartiMorph by Yongcheng YAO (CUHK)
"""

import argparse
import csv
import json
import sys
import time
import traceback

import nibabel as nib
import numpy as np
from collections import OrderedDict
from pathlib import Path

# Add current directory to path for local imports
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
from dabp import compute_dabp_simple, compute_dabp_from_pseudo_healthy


# ============================================================
# Helpers
# ============================================================

_ZERO_THICKNESS = {
    'thickness_mean_mm': 0.0,
    'thickness_std_mm': 0.0,
    'thickness_min_mm': 0.0,
    'thickness_max_mm': 0.0,
    'thickness_median_mm': 0.0,
    'thickness_n_measured': 0,
}

_ZERO_AREA_AND_THICKNESS = {
    'surface_area_mm2': 0.0,
    **_ZERO_THICKNESS,
}


def _empty_thickness(results):
    """Populate results dict with zero-valued thickness and area fields."""
    results.update(_ZERO_AREA_AND_THICKNESS)
    return results


# ============================================================
# Single Region Measurement
# ============================================================

def measure_single_region(mask_3d, voxel_size, region_name='Cartilage',
                          sn_n_neighbors=10, sn_smooth_n_neighbors=10,
                          thickness_depth=20.0, do_thickness=True,
                          verbose=False):
    """
    Measure volume, surface area, and thickness for a single cartilage region.

    Parameters
    ----------
    mask_3d : np.ndarray, binary
        3D mask for the region
    voxel_size : np.ndarray, shape (3,)
    region_name : str
    sn_n_neighbors : int
    sn_smooth_n_neighbors : int
    thickness_depth : float
    do_thickness : bool
    verbose : bool

    Returns
    -------
    dict : measurement results
    """
    results = OrderedDict()
    results['region'] = region_name
    n_voxels = int(np.sum(mask_3d))
    results['n_voxels'] = n_voxels

    if n_voxels == 0:
        results['volume_mm3'] = 0.0
        return _empty_thickness(results)

    voxel_volume = float(np.prod(voxel_size))
    results['volume_mm3'] = float(n_voxels * voxel_volume)

    # Clean mask
    mask_clean = bwareaopen_3d(mask_3d, min_size=10, connectivity=26)
    if np.sum(mask_clean) == 0:
        return _empty_thickness(results)

    # Mesh reconstruction (outer surface)
    fv = mask_to_mesh(mask_clean)
    vertices = fv['vertices'].astype(np.float64)
    faces = fv['faces']

    if len(faces) == 0:
        return _empty_thickness(results)

    results['n_vertices'] = len(vertices)
    results['n_faces'] = len(faces)

    # Surface area
    results['surface_area_mm2'] = float(np.sum(calculate_tri_mesh_area(vertices, faces)))

    # Thickness
    if not do_thickness or len(vertices) == 0:
        results.update(_ZERO_THICKNESS)
        return results

    if verbose:
        print(f"    Estimating surface normals ({len(vertices)} verts)...")

    sn = estimate_surface_normals(vertices, n_neighbors=sn_n_neighbors)
    sn_smooth = smooth_surface_normals(sn, vertices, n_neighbors=sn_smooth_n_neighbors)

    if verbose:
        print(f"    Measuring thickness (depth={thickness_depth})...")

    thickness_map = thickness_map_sn_fast(vertices, sn_smooth, fv, thickness_depth)
    thickness_positive = thickness_map[:, 3][thickness_map[:, 3] > 0]

    if len(thickness_positive) > 0:
        results['thickness_mean_mm'] = float(np.mean(thickness_positive))
        results['thickness_std_mm'] = float(np.std(thickness_positive))
        results['thickness_min_mm'] = float(np.min(thickness_positive))
        results['thickness_max_mm'] = float(np.max(thickness_positive))
        results['thickness_median_mm'] = float(np.median(thickness_positive))
        results['thickness_n_measured'] = int(len(thickness_positive))
    else:
        results.update(_ZERO_THICKNESS)

    return results


# ============================================================
# Main Measurement Pipeline
# ============================================================

def measure_cartilage_20regions(
    seg_path,
    output_dir=None,
    label_fc=3,
    label_mtc=4,
    label_ltc=5,
    label_femur_bone=1,
    label_tibia_bone=2,
    knee_side='right',
    cc_percentage=0.6,
    sn_n_neighbors=10,
    sn_smooth_n_neighbors=10,
    thickness_depth=20.0,
    do_thickness=True,
    do_dabp=True,
    pseudo_healthy_path=None,
    save_atlas=True,
    verbose=True,
):
    """
    Main pipeline: 20-region parcellation + morphological measurement.

    Parameters
    ----------
    seg_path : str or Path
        Path to segmentation NIfTI file
    output_dir : str or Path, optional
        Directory to save results
    label_fc, label_mtc, label_ltc : int
        Segmentation labels for cartilage
    label_femur_bone, label_tibia_bone : int
        Segmentation labels for bone
    knee_side : str
        'right' or 'left'
    cc_percentage : float
        Central region parameter (default 0.6)
    sn_n_neighbors : int
    sn_smooth_n_neighbors : int
    thickness_depth : float
    do_thickness : bool
    do_dabp : bool
    save_atlas : bool
    verbose : bool

    Returns
    -------
    dict : complete measurement results
    """
    seg_path = Path(seg_path)
    t_start = time.time()

    if verbose:
        print(f"{'='*60}")
        print(f"CartiMorph 20-Region Measurement")
        print(f"{'='*60}")
        print(f"Input: {seg_path}")

    # --- Load segmentation ---
    seg_img = nib.load(str(seg_path))
    seg_data = seg_img.get_fdata().astype(np.uint8)
    affine = seg_img.affine
    header = seg_img.header
    voxel_size = np.array(header.get_zooms()[:3], dtype=np.float64)
    img_size = seg_data.shape

    if verbose:
        print(f"Image size: {img_size}")
        print(f"Voxel size: {voxel_size} mm")
        print(f"Labels: {np.unique(seg_data)}")

    # --- Extract masks ---
    mask_fc = (seg_data == label_fc).astype(np.uint8)
    mask_mtc = (seg_data == label_mtc).astype(np.uint8)
    mask_ltc = (seg_data == label_ltc).astype(np.uint8)
    mask_cartilage = ((seg_data == label_fc) |
                      (seg_data == label_mtc) |
                      (seg_data == label_ltc)).astype(np.uint8)
    mask_femur_bone = (seg_data == label_femur_bone).astype(np.uint8)
    mask_tibia_bone = (seg_data == label_tibia_bone).astype(np.uint8)
    mask_bone = (mask_femur_bone | mask_tibia_bone).astype(np.uint8)

    if verbose:
        print(f"FC voxels: {np.sum(mask_fc)}")
        print(f"MTC voxels: {np.sum(mask_mtc)}")
        print(f"LTC voxels: {np.sum(mask_ltc)}")

    # --- Parcellation ---
    if verbose:
        print(f"\n{'='*60}")
        print("PARCELLATION")
        print(f"{'='*60}")

    # FC parcellation
    if np.sum(mask_fc) > 0:
        if verbose:
            print("Running FC parcellation...")
        atlas_fc = volume_parcellation_fc(mask_fc, knee_side, cc_percentage, img_size)
    else:
        atlas_fc = np.zeros(img_size, dtype=np.uint8)

    # TC parcellation
    if np.sum(mask_mtc) > 0 and np.sum(mask_ltc) > 0:
        if verbose:
            print("Running TC parcellation...")
        atlas_tc = volume_parcellation_tc(
            mask_mtc, mask_ltc, knee_side, voxel_size, img_size
        )
    else:
        atlas_tc = np.zeros(img_size, dtype=np.uint8)

    # Combined atlas (FC codes 1-10, TC codes 11-20; use where to avoid overlap corruption)
    atlas = np.where(atlas_tc > 0, atlas_tc, atlas_fc)

    # Print parcellation summary
    if verbose:
        print(f"\nParcellation results:")
        for roi_code in sorted(ALL_REGION_NAMES.keys()):
            if roi_code == 0:
                continue
            count = np.sum(atlas == roi_code)
            if count > 0:
                print(f"  ROI {roi_code:2d} ({ALL_REGION_NAMES[roi_code]:5s}): {count} voxels")

    # --- Save atlas ---
    if save_atlas and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        case_name = seg_path.stem.replace('.nii', '')
        atlas_path = output_dir / f"{case_name}_atlas.nii.gz"
        atlas_img = nib.Nifti1Image(atlas, affine=affine)
        nib.save(atlas_img, str(atlas_path))
        if verbose:
            print(f"Atlas saved: {atlas_path}")

    # --- Measure each region ---
    if verbose:
        print(f"\n{'='*60}")
        print("MEASUREMENTS")
        print(f"{'='*60}")

    results = OrderedDict()
    results['case_name'] = seg_path.stem.replace('.nii', '')
    results['input_file'] = str(seg_path)
    results['image_size'] = list(img_size)
    results['voxel_size_mm'] = voxel_size.tolist()
    results['knee_side'] = knee_side
    results['regions'] = OrderedDict()

    # Whole cartilage measurements
    if verbose:
        print(f"\n--- Whole Cartilage ---")
    results['regions']['whole'] = measure_single_region(
        mask_cartilage, voxel_size, 'Whole',
        sn_n_neighbors, sn_smooth_n_neighbors, thickness_depth,
        do_thickness, verbose
    )

    # FC whole
    if np.sum(mask_fc) > 0:
        if verbose:
            print(f"\n--- Whole FC ---")
        results['regions']['FC_whole'] = measure_single_region(
            mask_fc, voxel_size, 'FC_whole',
            sn_n_neighbors, sn_smooth_n_neighbors, thickness_depth,
            do_thickness, verbose
        )

    # TC whole
    mask_tc = (mask_mtc | mask_ltc).astype(np.uint8)
    if np.sum(mask_tc) > 0:
        if verbose:
            print(f"\n--- Whole TC ---")
        results['regions']['TC_whole'] = measure_single_region(
            mask_tc, voxel_size, 'TC_whole',
            sn_n_neighbors, sn_smooth_n_neighbors, thickness_depth,
            do_thickness, verbose
        )

    # 20 subregions
    for roi_code in sorted(ALL_REGION_NAMES.keys()):
        if roi_code == 0:
            continue
        region_name = ALL_REGION_NAMES[roi_code]
        region_mask = (atlas == roi_code).astype(np.uint8)

        if np.sum(region_mask) == 0:
            continue

        if verbose:
            print(f"\n--- ROI {roi_code}: {region_name} ---")

        results['regions'][region_name] = measure_single_region(
            region_mask, voxel_size, region_name,
            sn_n_neighbors, sn_smooth_n_neighbors, thickness_depth,
            do_thickness, verbose
        )

    # --- dABp ---
    if do_dabp:
        if verbose:
            print(f"\n{'='*60}")
            print("dABp (Denuded Area Bone Percentage)")
            print(f"{'='*60}")

        region_codes = [c for c in sorted(ALL_REGION_NAMES.keys()) if c > 0]

        # Use registration-based dABp if pseudo-healthy mask is provided
        if pseudo_healthy_path is not None and Path(pseudo_healthy_path).exists():
            if verbose:
                print(f"Using registration-based dABp from: {pseudo_healthy_path}")
            pseudo_img = nib.load(str(pseudo_healthy_path))
            pseudo_healthy_mask = (pseudo_img.get_fdata() > 0).astype(np.uint8)
            dabp_dict, bone_area_dict = compute_dabp_from_pseudo_healthy(
                pseudo_healthy_mask, mask_cartilage, mask_bone,
                atlas, voxel_size, img_size, region_codes
            )
        else:
            if verbose:
                print("Using rule-based dABp (no pseudo-healthy mask provided)")
            dabp_dict, bone_area_dict = compute_dabp_simple(
                mask_cartilage, mask_bone, atlas, voxel_size, img_size, region_codes
            )

        # Add dABp to results
        for roi_code, dabp_val in dabp_dict.items():
            if roi_code == 0:
                # roi_code 0 = whole cartilage
                if 'whole' in results['regions']:
                    results['regions']['whole']['dabp_percent'] = dabp_val
                continue
            region_name = ALL_REGION_NAMES[roi_code]
            if region_name in results['regions']:
                results['regions'][region_name]['dabp_percent'] = dabp_val

        if verbose:
            print(f"\ndABp results:")
            if 0 in dabp_dict:
                print(f"  {'whole':5s}: {dabp_dict[0]:.1f}%")
            for roi_code in sorted(dabp_dict.keys()):
                if roi_code == 0:
                    continue
                name = ALL_REGION_NAMES[roi_code]
                print(f"  {name:5s}: {dabp_dict[roi_code]:.1f}%")

    # --- Processing time ---
    t_elapsed = time.time() - t_start
    results['processing_time_sec'] = round(t_elapsed, 2)

    if verbose:
        print(f"\nProcessing time: {t_elapsed:.1f} seconds")

    return results


# ============================================================
# Save Results
# ============================================================

def save_results_csv(results, output_path):
    """
    Save measurement results to CSV file.

    Parameters
    ----------
    results : dict
        Output from measure_cartilage_20regions
    output_path : str or Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Header
    headers = [
        'roi_code', 'region_name',
        'volume_mm3', 'surface_area_mm2',
        'thickness_mean_mm', 'thickness_std_mm',
        'thickness_min_mm', 'thickness_max_mm', 'thickness_median_mm',
        'thickness_n_measured',
        'dabp_percent',
    ]

    rows = []
    for region_name, region_data in results['regions'].items():
        # Determine ROI code
        roi_code = 0
        for code, name in ALL_REGION_NAMES.items():
            if name == region_name:
                roi_code = code
                break

        row = {
            'roi_code': roi_code,
            'region_name': region_name,
            'volume_mm3': round(region_data.get('volume_mm3', 0), 2),
            'surface_area_mm2': round(region_data.get('surface_area_mm2', 0), 2),
            'thickness_mean_mm': round(region_data.get('thickness_mean_mm', 0), 3),
            'thickness_std_mm': round(region_data.get('thickness_std_mm', 0), 3),
            'thickness_min_mm': round(region_data.get('thickness_min_mm', 0), 3),
            'thickness_max_mm': round(region_data.get('thickness_max_mm', 0), 3),
            'thickness_median_mm': round(region_data.get('thickness_median_mm', 0), 3),
            'thickness_n_measured': region_data.get('thickness_n_measured', 0),
            'dabp_percent': round(region_data.get('dabp_percent', 0), 2),
        }
        rows.append(row)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def save_results_json(results, output_path):
    """Save results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def print_summary(results):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"MEASUREMENT SUMMARY - {results['case_name']}")
    print(f"{'='*80}")
    print(f"{'Region':<10} {'Vol(mm3)':>10} {'Area(mm2)':>10} "
          f"{'Thick(mm)':>10} {'Thick±std':>12} {'dABp(%)':>8}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    for region_name, data in results['regions'].items():
        vol = data.get('volume_mm3', 0)
        area = data.get('surface_area_mm2', 0)
        thick = data.get('thickness_mean_mm', 0)
        thick_std = data.get('thickness_std_mm', 0)
        dabp = data.get('dabp_percent', 0)

        thick_str = f"{thick:.2f}±{thick_std:.2f}" if thick > 0 else "N/A"
        print(f"{region_name:<10} {vol:>10.1f} {area:>10.1f} "
              f"{thick:>10.2f} {thick_str:>12} {dabp:>8.1f}")

    print(f"\nProcessing time: {results.get('processing_time_sec', 0):.1f} seconds")


# ============================================================
# Batch Processing
# ============================================================

def measure_batch(seg_paths, output_dir, **kwargs):
    """
    Process multiple segmentation files.

    Parameters
    ----------
    seg_paths : list of str
    output_dir : str or Path
    **kwargs : passed to measure_cartilage_20regions

    Returns
    -------
    list of dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_files = len(seg_paths)

    print(f"Processing {n_files} files...")

    for i, seg_path in enumerate(seg_paths):
        print(f"\n[{i+1}/{n_files}] {Path(seg_path).name}")

        try:
            result = measure_cartilage_20regions(
                seg_path, output_dir=output_dir, verbose=False, **kwargs
            )
            all_results.append(result)

            # Save per-case CSV
            case_name = result['case_name']
            csv_path = output_dir / f"{case_name}_report.csv"
            save_results_csv(result, csv_path)

            # One-line summary
            whole = result['regions'].get('whole', {})
            vol = whole.get('volume_mm3', 0)
            area = whole.get('surface_area_mm2', 0)
            thick = whole.get('thickness_mean_mm', 0)
            print(f"  Whole: vol={vol:.1f}mm3, area={area:.1f}mm2, thick={thick:.2f}mm")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                'case_name': Path(seg_path).stem.replace('.nii', ''),
                'error': str(e)
            })

    # Save batch summary
    if all_results:
        batch_path = output_dir / "batch_summary.json"
        save_results_json(all_results, batch_path)
        print(f"\nBatch summary saved: {batch_path}")

    return all_results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='CartiMorph 20-Region Cartilage Morphological Measurement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single case
  python cartimorph_measure.py case001.nii.gz --output results/

  # Custom labels
  python cartimorph_measure.py case001.nii.gz --label_fc 3 --label_mtc 4 --label_ltc 5

  # Batch processing
  python cartimorph_measure.py labelsTr/*.nii.gz --output results/ --batch

  # Skip thickness (faster)
  python cartimorph_measure.py case001.nii.gz --no-thickness --output results/

  # Save atlas NIfTI
  python cartimorph_measure.py case001.nii.gz --save-atlas --output results/

Label convention (OAIZIB):
  1=Femur bone, 2=Tibia bone, 3=FC, 4=MTC, 5=LTC
        """
    )

    parser.add_argument('seg_files', nargs='+', type=str,
                        help='Segmentation NIfTI file(s)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--label_fc', type=int, default=3,
                        help='Label for FC (default: 3)')
    parser.add_argument('--label_mtc', type=int, default=4,
                        help='Label for MTC (default: 4)')
    parser.add_argument('--label_ltc', type=int, default=5,
                        help='Label for LTC (default: 5)')
    parser.add_argument('--label_femur_bone', type=int, default=1,
                        help='Label for femur bone (default: 1)')
    parser.add_argument('--label_tibia_bone', type=int, default=2,
                        help='Label for tibia bone (default: 2)')
    parser.add_argument('--knee_side', type=str, default='right',
                        choices=['right', 'left'],
                        help='Knee side (default: right)')
    parser.add_argument('--cc_percentage', type=float, default=0.6,
                        help='Central region parameter (default: 0.6)')
    parser.add_argument('--no-thickness', action='store_true',
                        help='Skip thickness measurement (much faster)')
    parser.add_argument('--no-dabp', action='store_true',
                        help='Skip dABp calculation')
    parser.add_argument('--pseudo_healthy', type=str, default=None,
                        help='Path to pseudo-healthy cartilage mask (registration-based dABp)')
    parser.add_argument('--save-atlas', action='store_true',
                        help='Save parcellation atlas as NIfTI')
    parser.add_argument('--sn_neighbors', type=int, default=10,
                        help='Neighbors for surface normal estimation')
    parser.add_argument('--thickness_depth', type=float, default=20.0,
                        help='Maximum thickness depth in voxels')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: process all files')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    verbose = not args.quiet
    output_dir = args.output or str(Path.cwd() / 'results')

    kwargs = dict(
        label_fc=args.label_fc,
        label_mtc=args.label_mtc,
        label_ltc=args.label_ltc,
        label_femur_bone=args.label_femur_bone,
        label_tibia_bone=args.label_tibia_bone,
        knee_side=args.knee_side,
        cc_percentage=args.cc_percentage,
        sn_n_neighbors=args.sn_neighbors,
        sn_smooth_n_neighbors=args.sn_neighbors,
        thickness_depth=args.thickness_depth,
        do_thickness=not args.no_thickness,
        do_dabp=not args.no_dabp,
        pseudo_healthy_path=args.pseudo_healthy,
        save_atlas=args.save_atlas,
    )

    if args.batch or len(args.seg_files) > 1:
        # Batch mode
        measure_batch(args.seg_files, output_dir, **kwargs)
    else:
        # Single file
        result = measure_cartilage_20regions(
            args.seg_files[0], output_dir=output_dir,
            verbose=verbose, **kwargs
        )

        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        case_name = result['case_name']

        csv_path = output_dir / f"{case_name}_report.csv"
        save_results_csv(result, csv_path)
        print(f"\nCSV report saved: {csv_path}")

        json_path = output_dir / f"{case_name}_report.json"
        save_results_json(result, json_path)
        print(f"JSON report saved: {json_path}")

        # Print summary
        print_summary(result)


if __name__ == '__main__':
    main()
