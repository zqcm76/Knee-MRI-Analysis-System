"""
dABp (denuded Area Bone percentage) Calculation

Computes the percentage of full-thickness cartilage loss relative to
the total subchondral bone surface area for each cartilage subregion.

Method:
    1. Extract subchondral bone surface (bone voxels adjacent to cartilage)
    2. Reconstruct pseudo-healthy surface via sagittal-slice hole filling
    3. Compare actual cartilage coverage vs reconstructed coverage
    4. dABp = denuded_area / total_bone_area * 100%

Reference:
    YAO et al., CartiMorph (formula for dABp measurement)
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

from mesh_utils import (
    mask_to_mesh, calculate_tri_mesh_area,
    convert_voxel_indices_to_mask_3d,
)


def _voxel_subsurface_area(subs, img_size):
    """Compute mesh surface area for a set of voxel subscripts."""
    if len(subs) == 0:
        return 0.0
    mask = convert_voxel_indices_to_mask_3d(subs, img_size)
    mesh = mask_to_mesh(mask)
    return float(np.sum(calculate_tri_mesh_area(
        mesh['vertices'].astype(np.float64), mesh['faces']
    )))


def compute_dabp(mask_cartilage, mask_bone, atlas, voxel_size, img_size,
                 region_codes):
    """
    Compute dABp for each cartilage subregion.

    For each region:
    1. Find the bone-cartilage interface surface
    2. Reconstruct pseudo-healthy interface via hole filling
    3. dABp = (reconstructed_area - actual_cartilage_area) / reconstructed_area * 100

    Parameters
    ----------
    mask_cartilage : np.ndarray, binary
        Combined cartilage mask (FC + MTC + LTC)
    mask_bone : np.ndarray, binary
        Combined bone mask (femur + tibia)
    atlas : np.ndarray, dtype uint8
        Parcellation atlas with ROI codes 1-20
    voxel_size : np.ndarray, shape (3,)
    img_size : tuple of 3 ints
    region_codes : list of int
        ROI codes to compute dABp for
    knee_side : str

    Returns
    -------
    dabp_dict : dict
        {roi_code: dABp_percentage}
    """
    dabp_dict = {}

    for roi_code in region_codes:
        # Get the region mask
        region_mask = (atlas == roi_code).astype(np.uint8)

        if np.sum(region_mask) == 0:
            dabp_dict[roi_code] = 0.0
            continue

        # Compute dABp for this region
        dabp = _compute_region_dabp(
            region_mask, mask_bone, voxel_size, img_size
        )
        dabp_dict[roi_code] = dabp

    return dabp_dict


def _compute_region_dabp(region_mask, mask_bone, voxel_size, img_size):
    """
    Compute dABp for a single cartilage region.

    Uses sagittal-slice analysis to detect and fill gaps in cartilage coverage.

    Parameters
    ----------
    region_mask : np.ndarray, binary
        Single region cartilage mask
    mask_bone : np.ndarray, binary
        Bone mask
    voxel_size : np.ndarray
    img_size : tuple

    Returns
    -------
    dabp : float
        Percentage of denuded bone area (0-100)
    """
    # Get cartilage voxels in this region
    subs_cart = np.column_stack(np.where(region_mask))

    if len(subs_cart) == 0:
        return 0.0

    # Find bone-cartilage interface
    # Bone voxels adjacent to cartilage (dilate cartilage by 1, intersect with bone)
    cart_dilated = ndimage.binary_dilation(region_mask, iterations=1)
    interface_mask = mask_bone & cart_dilated

    if np.sum(interface_mask) == 0:
        return 0.0

    # Get interface voxels
    subs_interface = np.column_stack(np.where(interface_mask))

    # Sagittal slice analysis for hole filling
    # For each sagittal slice (dim0), check if there are gaps
    sag_slices = np.unique(subs_interface[:, 0])

    # Reconstruct pseudo-healthy interface
    subs_filled = subs_interface.copy()

    for sag_idx in sag_slices:
        # Get interface points in this sagittal slice
        mask_sag = subs_interface[:, 0] == sag_idx
        subs_sag = subs_interface[mask_sag]

        if len(subs_sag) < 5:
            continue

        # Check for gaps using connected components in 2D
        sag_2d = subs_sag[:, [1, 2]]  # AP, SI
        sag_2d_mask = np.zeros((img_size[1], img_size[2]), dtype=np.uint8)
        sag_2d_mask[sag_2d[:, 0], sag_2d[:, 1]] = 1

        # Remove small components (use 2D labeling directly)
        labeled_raw, num_raw = ndimage.label(sag_2d_mask)
        if num_raw > 0:
            sizes = np.bincount(labeled_raw.ravel())
            keep = np.where(sizes >= 3)[0]
            keep = keep[keep > 0]
            sag_2d_mask = np.isin(labeled_raw, keep).astype(np.uint8)

        # Find connected components
        labeled, num_cc = ndimage.label(sag_2d_mask)

        if num_cc <= 1:
            # No gap in this slice
            continue

        # There are gaps - try to fill them via curve fitting
        filled_points = _fill_sagittal_gaps(
            sag_2d, sag_2d_mask, img_size[1], img_size[2]
        )

        if len(filled_points) > 0:
            new_subs = np.column_stack([
                np.full(len(filled_points), sag_idx),
                filled_points
            ])
            subs_filled = np.vstack([subs_filled, new_subs])

    # Compute areas
    area_actual = _voxel_subsurface_area(subs_interface, img_size)

    # Reconstructed (pseudo-healthy) interface area
    if len(subs_filled) > 0:
        subs_filled_unique = np.unique(subs_filled, axis=0)
        area_filled = _voxel_subsurface_area(subs_filled_unique, img_size)
    else:
        area_filled = area_actual

    # dABp = (reconstructed - actual) / reconstructed * 100
    if area_filled > 0:
        dabp = (area_filled - area_actual) / area_filled * 100.0
        dabp = np.clip(dabp, 0.0, 100.0)
    else:
        dabp = 0.0

    return float(dabp)


def _fill_sagittal_gaps(subs_2d, mask_2d, dim_ap, dim_si):
    """
    Fill gaps in a sagittal slice using curve fitting.

    Detects multiple connected components and tries to bridge gaps
    by fitting a curve through the boundary points.

    Parameters
    ----------
    subs_2d : np.ndarray, shape (n, 2)
        2D coordinates (AP, SI) of interface points
    mask_2d : np.ndarray, shape (dim_ap, dim_si)
        2D binary mask
    dim_ap, dim_si : int
        Dimensions

    Returns
    -------
    filled_points : np.ndarray, shape (m, 2)
        Points added to fill gaps (AP, SI)
    """
    labeled, num_cc = ndimage.label(mask_2d)

    if num_cc <= 1:
        return np.empty((0, 2), dtype=int)

    # Find the largest component (main interface)
    cc_sizes = []
    for cc_id in range(1, num_cc + 1):
        cc_sizes.append(np.sum(labeled == cc_id))

    largest_cc = np.argmax(cc_sizes) + 1

    # For each smaller component, try to connect to the largest
    filled = []

    for cc_id in range(1, num_cc + 1):
        if cc_id == largest_cc:
            continue

        cc_points = np.column_stack(np.where(labeled == cc_id))
        main_points = np.column_stack(np.where(labeled == largest_cc))

        if len(cc_points) < 3 or len(main_points) < 3:
            continue

        # Find the closest points between the two components
        dists = cdist(cc_points.astype(float), main_points.astype(float))
        min_dist = np.min(dists)
        idx_cc, idx_main = np.unravel_index(np.argmin(dists), dists.shape)

        # If gap is small enough, fill it
        if min_dist < 20:  # max gap of 20 voxels
            # Simple linear interpolation between closest points
            p1 = cc_points[idx_cc].astype(float)
            p2 = main_points[idx_main].astype(float)

            n_steps = int(np.ceil(min_dist))
            for step in range(1, n_steps):
                t = step / n_steps
                p = p1 * (1 - t) + p2 * t
                p_int = np.round(p).astype(int)
                # Check bounds
                if (0 <= p_int[0] < dim_ap and 0 <= p_int[1] < dim_si):
                    filled.append(p_int)

    if filled:
        return np.array(filled)
    return np.empty((0, 2), dtype=int)


def compute_dabp_simple(mask_cartilage, mask_bone, atlas, voxel_size,
                        img_size, region_codes):
    """
    Simplified dABp computation using voxel-based coverage estimation.

    For each region, estimates the bone surface area and the fraction
    that lacks cartilage coverage directly above it.

    This is faster than the hole-filling method but may slightly
    underestimate dABp in areas with thin cartilage remnants.

    Parameters
    ----------
    mask_cartilage : np.ndarray, binary
    mask_bone : np.ndarray, binary
    atlas : np.ndarray, uint8
    voxel_size : np.ndarray
    img_size : tuple
    region_codes : list of int

    Returns
    -------
    dabp_dict : dict
        {roi_code: dABp_percentage}
    bone_area_dict : dict
        {roi_code: bone_surface_voxel_count}
    """
    dabp_dict = {}
    bone_area_dict = {}

    for roi_code in region_codes:
        region_mask = (atlas == roi_code).astype(np.uint8)

        if np.sum(region_mask) == 0:
            dabp_dict[roi_code] = 0.0
            bone_area_dict[roi_code] = 0
            continue

        # Find bone surface for this region:
        # Bone voxels that are adjacent to this cartilage region
        region_dilated = ndimage.binary_dilation(region_mask, iterations=1)
        bone_surface = mask_bone & region_dilated

        total_count = int(np.sum(bone_surface))
        bone_area_dict[roi_code] = total_count

        if total_count == 0:
            dabp_dict[roi_code] = 0.0
            continue

        # Find denuded bone: bone surface voxels NOT covered by cartilage
        # A bone voxel is "covered" if the original cartilage mask has a
        # cartilage voxel within 1 voxel (using dilation of the ORIGINAL
        # cartilage mask, not the region mask)
        cart_dilated = ndimage.binary_dilation(mask_cartilage, iterations=1)
        denuded_bone = bone_surface & ~cart_dilated
        denuded_count = int(np.sum(denuded_bone))

        dabp = denuded_count / total_count * 100.0
        dabp_dict[roi_code] = float(np.clip(dabp, 0.0, 100.0))

    return dabp_dict, bone_area_dict


def compute_dabp_from_pseudo_healthy(
    pseudo_healthy_mask, actual_mask, mask_bone, atlas, voxel_size, img_size, region_codes
):
    """
    Compute dABp using volumetric set operations (no interface extraction).

    Per CartiMorph paper (Eckstein et al. 2006a):
        dABp = count(pseudo & ~actual) / count(pseudo) * 100%

    Robust version with:
    - Minimum voxel threshold (filter unreliable boundary regions)
    - Actual mask dilation to absorb 1-2 voxel registration boundary errors

    Parameters
    ----------
    pseudo_healthy_mask : np.ndarray, binary
        Pseudo-healthy cartilage mask (from template registration)
    actual_mask : np.ndarray, binary
        Actual patient cartilage mask
    mask_bone : np.ndarray, binary
        Bone mask (unused in computation, kept for API compatibility)
    atlas : np.ndarray, uint8
        Warped parcellation atlas with ROI codes 1-20
    voxel_size : np.ndarray, shape (3,)
    img_size : tuple of 3 ints
    region_codes : list of int

    Returns
    -------
    dabp_dict : dict
        {roi_code: dABp_percentage} (roi_code 0 = whole cartilage)
    bone_area_dict : dict
        {roi_code: pseudo_healthy_voxel_count}
    """
    struct = ndimage.generate_binary_structure(3, 1)
    pseudo = pseudo_healthy_mask.astype(bool)
    actual = actual_mask.astype(bool)

    # ROI restriction: only compute near pseudo-healthy region
    roi = ndimage.binary_dilation(pseudo, structure=struct, iterations=3)

    # Bidirectional buffer: dilate actual by 2 to absorb boundary noise
    actual_dilated = ndimage.binary_dilation(actual, structure=struct, iterations=2)

    # FCL = pseudo region where actual is absent (after buffer), within ROI
    fcl = pseudo & ~actual_dilated & roi

    # Denominator: pseudo within ROI
    pseudo_in_roi = pseudo & roi
    total_pseudo = int(pseudo_in_roi.sum())

    dabp_dict = {}
    bone_area_dict = {}

    # Whole cartilage dABp
    fcl_voxels = int(fcl.sum())
    dabp_dict[0] = fcl_voxels / total_pseudo * 100.0 if total_pseudo > 0 else 0.0
    bone_area_dict[0] = total_pseudo

    # Per-region dABp using warped atlas
    # Filter on pseudo volume (not actual) — keep regions where pseudo >= threshold
    min_pseudo_voxels = 200

    for roi_code in region_codes:
        region_pseudo = (atlas == roi_code) & pseudo_in_roi
        region_pseudo_total = int(region_pseudo.sum())
        bone_area_dict[roi_code] = region_pseudo_total

        if region_pseudo_total < min_pseudo_voxels:
            dabp_dict[roi_code] = None
            continue

        region_fcl = int((fcl & (atlas == roi_code)).sum())
        dabp_dict[roi_code] = region_fcl / region_pseudo_total * 100.0

    return dabp_dict, bone_area_dict
