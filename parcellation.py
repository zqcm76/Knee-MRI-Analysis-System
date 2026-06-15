"""
Rule-Based Cartilage Parcellation (20 Regions)

Implements the rule-based parcellation scheme from CartiMorph:
- FC (Femoral Cartilage): 10 subregions
- TC (Tibial Cartilage): 10 subregions

This is a RULE-BASED approach (not atlas-based), following the MATLAB
implementation by Yongcheng YAO (CUHK).

FC ROI codes:
    1: aMFC   (anterior medial FC)
    2: ecMFC  (exterior central medial FC)
    3: ccMFC  (central central medial FC)
    4: icMFC  (interior central medial FC)
    5: pMFC   (posterior medial FC)
    6: aLFC   (anterior lateral FC)
    7: ecLFC  (exterior central lateral FC)
    8: ccLFC  (central central lateral FC)
    9: icLFC  (interior central lateral FC)
    10: pLFC  (posterior lateral FC)

TC ROI codes:
    11: aMTC  (anterior medial TC)
    12: eMTC  (exterior medial TC)
    13: pMTC  (posterior medial TC)
    14: iMTC  (interior medial TC)
    15: cMTC  (central medial TC)
    16: aLTC  (anterior lateral TC)
    17: eLTC  (exterior lateral TC)
    18: pLTC  (posterior lateral TC)
    19: iLTC  (interior lateral TC)
    20: cLTC  (central lateral TC)

Reference:
    YAO et al., CartiMorph: Automated cartilage morphological analysis
"""

import numpy as np


# ============================================================
# ROI name mapping
# ============================================================

FC_REGION_NAMES = {
    1: 'aMFC', 2: 'ecMFC', 3: 'ccMFC', 4: 'icMFC', 5: 'pMFC',
    6: 'aLFC', 7: 'ecLFC', 8: 'ccLFC', 9: 'icLFC', 10: 'pLFC',
}

TC_REGION_NAMES = {
    11: 'aMTC', 12: 'eMTC', 13: 'pMTC', 14: 'iMTC', 15: 'cMTC',
    16: 'aLTC', 17: 'eLTC', 18: 'pLTC', 19: 'iLTC', 20: 'cLTC',
}

ALL_REGION_NAMES = {**FC_REGION_NAMES, **TC_REGION_NAMES}
ALL_REGION_NAMES[0] = 'background'


def get_region_name(roi_code):
    """Get region name from ROI code."""
    return ALL_REGION_NAMES.get(roi_code, f'unknown_{roi_code}')


def _set_atlas(atlas, subs, value, img_size):
    """Assign value to atlas at given voxel subscripts, with bounds checking."""
    if len(subs) == 0:
        return
    valid = (
        (subs[:, 0] >= 0) & (subs[:, 0] < img_size[0]) &
        (subs[:, 1] >= 0) & (subs[:, 1] < img_size[1]) &
        (subs[:, 2] >= 0) & (subs[:, 2] < img_size[2])
    )
    s = subs[valid]
    atlas[s[:, 0], s[:, 1], s[:, 2]] = value


# ============================================================
# FC Parcellation (Volume-based, rule-based)
# ============================================================

def volume_parcellation_fc(mask_fc, knee_side, cc_percentage, img_size):
    """
    Rule-based femoral cartilage volume parcellation.

    Algorithm:
    1. Locate intercondylar notch (center 50% sagittal slices)
    2. Split FC into MFC (medial) and LFC (lateral) by notch LR position
    3. For each: define central region by AP percentage (cc_percentage)
    4. Central region: split into ec/cc/ic by LR thirds
    5. Anterior and posterior: remaining parts

    Parameters
    ----------
    mask_fc : np.ndarray, binary
        Femoral cartilage segmentation mask
    knee_side : str
        'right' or 'left'
    cc_percentage : float
        Parameter for central region definition (e.g., 0.6)
    img_size : tuple of 3 ints

    Returns
    -------
    atlas : np.ndarray, dtype uint8
        Atlas with ROI codes 1-10
    """
    # Get voxel subscripts (row, col, slice)
    subs_fc = np.column_stack(np.where(mask_fc))

    if len(subs_fc) == 0:
        return np.zeros(img_size, dtype=np.uint8)

    # --- Locate intercondylar notch ---
    # Assuming dim0 = L-R, dim1 = P-A (y+ points to anterior)
    subs_lr = np.unique(subs_fc[:, 0])
    num_sag = len(subs_lr)

    # Center 50% of sagittal slices
    idx_start = int(num_sag * 0.25)
    idx_end = max(int(num_sag * 0.75), idx_start + 1)
    subs_lr_center = subs_lr[idx_start:idx_end]

    # For each center sagittal slice, find the min AP (deepest point = notch)
    min_sub_ap_center = np.zeros(len(subs_lr_center))
    for i, s in enumerate(subs_lr_center):
        mask_s = subs_fc[:, 0] == s
        min_sub_ap_center[i] = np.min(subs_fc[mask_s, 1])

    # Notch position: the sagittal slice where the deepest point is maximal
    sub_notch_ap_center = np.max(min_sub_ap_center)
    candidates = np.where(min_sub_ap_center == sub_notch_ap_center)[0]
    idx = candidates[len(candidates) // 2]
    sub_notch_lr = subs_lr_center[idx]

    # --- Split into LFC and MFC ---
    # Right knee: LFC has larger LR (lateral), MFC has smaller LR (medial)
    subs_lfc = subs_fc[subs_fc[:, 0] >= sub_notch_lr, :]
    subs_mfc = subs_fc[subs_fc[:, 0] < sub_notch_lr, :]

    # --- Locate central LFC (cLFC) ---
    min_sub_ap_lfc = np.min(subs_lfc[:, 1])
    min_sub_ap_clfc = sub_notch_ap_center - int(
        (sub_notch_ap_center - min_sub_ap_lfc) * cc_percentage
    )
    idx_ap_clfc = (subs_lfc[:, 1] < sub_notch_ap_center) & \
                  (subs_lfc[:, 1] > min_sub_ap_clfc)
    subs_clfc = subs_lfc[idx_ap_clfc, :]

    # --- Locate central MFC (cMFC) ---
    min_sub_ap_mfc = np.min(subs_mfc[:, 1])
    min_sub_ap_cmfc = sub_notch_ap_center - int(
        (sub_notch_ap_center - min_sub_ap_mfc) * cc_percentage
    )
    idx_ap_cmfc = (subs_mfc[:, 1] < sub_notch_ap_center) & \
                  (subs_mfc[:, 1] > min_sub_ap_cmfc)
    subs_cmfc = subs_mfc[idx_ap_cmfc, :]

    # --- Anterior and posterior LFC ---
    max_sub_ap_lfc = np.max(subs_lfc[:, 1])
    subs_alfc = subs_lfc[
        (subs_lfc[:, 1] <= max_sub_ap_lfc) & (subs_lfc[:, 1] >= sub_notch_ap_center)
    ]
    subs_plfc = subs_lfc[
        (subs_lfc[:, 1] <= min_sub_ap_clfc) & (subs_lfc[:, 1] >= min_sub_ap_lfc)
    ]

    # --- Anterior and posterior MFC ---
    max_sub_ap_mfc = np.max(subs_mfc[:, 1])
    subs_amfc = subs_mfc[
        (subs_mfc[:, 1] <= max_sub_ap_mfc) & (subs_mfc[:, 1] >= sub_notch_ap_center)
    ]
    subs_pmfc = subs_mfc[
        (subs_mfc[:, 1] <= min_sub_ap_cmfc) & (subs_mfc[:, 1] >= min_sub_ap_mfc)
    ]

    # --- Divide central regions into LR thirds ---
    def divide_central_lr(subs_central):
        """Split central region into exterior/central/interior by LR thirds."""
        if len(subs_central) == 0:
            return (np.empty((0, 3), dtype=int),
                    np.empty((0, 3), dtype=int),
                    np.empty((0, 3), dtype=int))

        subs_ap = np.unique(subs_central[:, 1])
        subs_ec_list, subs_cc_list, subs_ic_list = [], [], []

        for ap in subs_ap:
            i_subs = subs_central[subs_central[:, 1] == ap, :]
            max_lr = np.max(i_subs[:, 0])
            min_lr = np.min(i_subs[:, 0])

            cut_f1 = min_lr + (max_lr - min_lr) / 3.0
            cut_f2 = min_lr + 2.0 * (max_lr - min_lr) / 3.0

            idx_ec = i_subs[:, 0] > cut_f2
            idx_cc = (i_subs[:, 0] >= cut_f1) & (i_subs[:, 0] <= cut_f2)
            idx_ic = i_subs[:, 0] < cut_f1

            subs_ec_list.append(i_subs[idx_ec])
            subs_cc_list.append(i_subs[idx_cc])
            subs_ic_list.append(i_subs[idx_ic])

        subs_ec = np.vstack(subs_ec_list) if subs_ec_list else np.empty((0, 3), dtype=int)
        subs_cc = np.vstack(subs_cc_list) if subs_cc_list else np.empty((0, 3), dtype=int)
        subs_ic = np.vstack(subs_ic_list) if subs_ic_list else np.empty((0, 3), dtype=int)

        return subs_ec, subs_cc, subs_ic

    subs_eclfc, subs_cclfc, subs_iclfc = divide_central_lr(subs_clfc)
    subs_ecmfc, subs_ccmfc, subs_icmfc = divide_central_lr(subs_cmfc)

    # --- Create atlas ---
    atlas = np.zeros(img_size, dtype=np.uint8)

    # Right knee: MFC=1-5, LFC=6-10; Left knee: swap medial and lateral
    if knee_side.lower() == 'right':
        mfc_labels = (1, 2, 3, 4, 5)
        lfc_labels = (6, 7, 8, 9, 10)
    else:
        mfc_labels = (6, 7, 8, 9, 10)
        lfc_labels = (1, 2, 3, 4, 5)

    for subs, label in [
        (subs_amfc, mfc_labels[0]), (subs_ecmfc, mfc_labels[1]),
        (subs_ccmfc, mfc_labels[2]), (subs_icmfc, mfc_labels[3]),
        (subs_pmfc, mfc_labels[4]),
        (subs_alfc, lfc_labels[0]), (subs_eclfc, lfc_labels[1]),
        (subs_cclfc, lfc_labels[2]), (subs_iclfc, lfc_labels[3]),
        (subs_plfc, lfc_labels[4]),
    ]:
        _set_atlas(atlas, subs, label, img_size)

    return atlas


# ============================================================
# TC Parcellation (Volume-based, rule-based)
# ============================================================

def volume_parcellation_tc(mask_mtc, mask_ltc, knee_side, voxel_size, img_size):
    """
    Rule-based tibial cartilage volume parcellation.

    Algorithm:
    1. For each TC (MTC, LTC):
       a. SVD to find principal axes
       b. Ellipse-based central region (20% volume)
       c. Remaining: split into 4 quadrants using ±45° cutting lines
    2. Assign labels based on quadrant centers (anterior/posterior/interior/exterior)

    Parameters
    ----------
    mask_mtc : np.ndarray, binary
        Medial tibial cartilage mask
    mask_ltc : np.ndarray, binary
        Lateral tibial cartilage mask
    knee_side : str
        'right' or 'left'
    voxel_size : np.ndarray, shape (3,)
    img_size : tuple of 3 ints

    Returns
    -------
    atlas : np.ndarray, dtype uint8
        Atlas with ROI codes 11-20
    """
    atlas = np.zeros(img_size, dtype=np.uint8)

    # Parcellate MTC and LTC
    subs_cmtc, vers_noncmtc, trans_mat_mtc, center_mtc, vers_mtc = \
        _parcellate_single_tc(mask_mtc, voxel_size)
    subs_cltc, vers_noncltc, trans_mat_ltc, center_ltc, vers_ltc = \
        _parcellate_single_tc(mask_ltc, voxel_size)

    # Define quadrants for non-central regions
    subs_amtc, subs_pmtc, subs_imtc, subs_emtc = _define_quadrants(
        vers_noncmtc, trans_mat_mtc, center_mtc, center_ltc, voxel_size, img_size
    )
    subs_altc, subs_pltc, subs_iltc, subs_eltc = _define_quadrants(
        vers_noncltc, trans_mat_ltc, center_ltc, center_mtc, voxel_size, img_size
    )

    # Right knee: a/e/p/i/c order; Left knee: swap interior and exterior
    if knee_side.lower() == 'right':
        mtc_labels = (11, 12, 13, 14, 15)
        ltc_labels = (16, 17, 18, 19, 20)
        mtc_subs = (subs_amtc, subs_emtc, subs_pmtc, subs_imtc, subs_cmtc)
        ltc_subs = (subs_altc, subs_eltc, subs_pltc, subs_iltc, subs_cltc)
    else:
        mtc_labels = (11, 12, 13, 14, 15)
        ltc_labels = (16, 17, 18, 19, 20)
        mtc_subs = (subs_amtc, subs_imtc, subs_pmtc, subs_emtc, subs_cmtc)
        ltc_subs = (subs_altc, subs_iltc, subs_pltc, subs_eltc, subs_cltc)

    for subs, label in zip(mtc_subs + ltc_subs, mtc_labels + ltc_labels):
        _set_atlas(atlas, subs, label, img_size)

    return atlas


def _parcellate_single_tc(mask_tc, voxel_size):
    """
    Parcellate a single tibial cartilage into central + non-central.

    Uses SVD to find principal axes, then ellipse-based central region
    covering 20% of total volume.

    Parameters
    ----------
    mask_tc : np.ndarray, binary
    voxel_size : np.ndarray, shape (3,)

    Returns
    -------
    subs_central : np.ndarray, shape (n, 3)
        Voxel subscripts of central region
    vers_noncentral : np.ndarray, shape (m, 3)
        Coordinates of non-central voxels (in mm, centered)
    trans_mat : np.ndarray, shape (3, 3)
        Transformation matrix (columns = principal axes)
    center : np.ndarray, shape (3,)
        Centroid of the TC
    vers_tc : np.ndarray, shape (k, 3)
        All TC voxel coordinates (in mm)
    """
    subs_tc = np.column_stack(np.where(mask_tc))
    if len(subs_tc) == 0:
        return (np.empty((0, 3), dtype=int),
                np.empty((0, 3)),
                np.eye(3), np.zeros(3), np.empty((0, 3)))

    vers_tc = subs_tc * voxel_size  # Convert to mm

    # Center the data
    center = np.mean(vers_tc, axis=0)
    vers_tc_c = vers_tc - center

    # SVD
    _, s, v = np.linalg.svd(vers_tc_c, full_matrices=False)

    # Find nearest anatomical axes
    direction_vectors = _find_nearest_axes(v)

    # Build transformation matrix
    trans_mat = np.zeros((3, 3))
    trans_mat[:, direction_vectors[0]] = v[:, 0]
    trans_mat[:, direction_vectors[1]] = v[:, 1]
    trans_mat[:, direction_vectors[2]] = v[:, 2]

    # Ensure positive diagonal
    for i in range(3):
        if trans_mat[i, i] < 0:
            trans_mat[:, i] *= -1

    # Transform to principal axis space
    vers_tc_ct = (np.linalg.inv(trans_mat) @ vers_tc_c.T).T

    # Initial ellipse for central region
    tmp_a = 5.0
    ratio = s[direction_vectors[1]] / max(s[direction_vectors[0]], 1e-10)
    tmp_b = tmp_a * np.sqrt(max(ratio, 0.01))

    x = vers_tc_ct[:, 0]
    y = vers_tc_ct[:, 1]

    # Adjust ellipse to cover 20% of volume
    volume_voxel = np.prod(voxel_size)
    volume_tc = len(np.unique(vers_tc, axis=0)) * volume_voxel

    target_pct = 0.20
    max_iter = 2000
    oscillation_count = 0
    last_direction = None

    for iteration in range(max_iter):
        idx_central = (x**2 / tmp_a**2 + y**2 / tmp_b**2 - 1) < 0
        vers_central_ct = vers_tc_ct[idx_central, :]
        vers_central_c = (trans_mat @ vers_central_ct.T).T
        vers_central = vers_central_c + center

        volume_central = len(np.unique(vers_central, axis=0)) * volume_voxel
        ratio_current = volume_central / max(volume_tc, 1e-10)

        if abs(ratio_current - target_pct) <= 0.005:
            break

        if ratio_current > target_pct:
            tmp_a -= 0.5
            direction = -1
        else:
            tmp_a += 0.5
            direction = 1

        # Guard against tmp_a going to zero or negative
        if tmp_a <= 0.5:
            tmp_a = 0.5

        tmp_b = tmp_a * np.sqrt(max(ratio, 0.01))

        if last_direction is not None and direction != last_direction:
            oscillation_count += 1
            if oscillation_count >= 1000:
                break
        last_direction = direction

    # Convert central region back to voxel subscripts
    subs_central = np.round(vers_central / voxel_size).astype(int)

    # Non-central region
    vers_noncentral = vers_tc_ct[~idx_central, :]

    return subs_central, vers_noncentral, trans_mat, center, vers_tc


def _define_quadrants(vers_noncentral, trans_mat, center_source, center_target,
                      voxel_size, img_size):
    """
    Split non-central tibial cartilage into 4 quadrants.

    Uses ±45° cutting lines relative to the direction from source to target
    (i.e., from MTC center to LTC center or vice versa).

    Returns
    -------
    subs_a, subs_p, subs_i, subs_e : np.ndarray
        Voxel subscripts for anterior, posterior, interior, exterior
    """
    if len(vers_noncentral) == 0:
        empty = np.empty((0, 3), dtype=int)
        return empty, empty, empty, empty

    # Transform target center relative to source
    center_c2t = center_target - center_source
    center_t_c = np.linalg.inv(trans_mat) @ center_c2t

    # Rotation matrices for ±45°
    theta1 = np.radians(45)
    theta2 = np.radians(-45)

    rot1 = np.array([
        [np.cos(theta1), -np.sin(theta1), 0],
        [np.sin(theta1),  np.cos(theta1), 0],
        [0, 0, 1]
    ])
    rot2 = np.array([
        [np.cos(theta2), -np.sin(theta2), 0],
        [np.sin(theta2),  np.cos(theta2), 0],
        [0, 0, 1]
    ])

    vect1 = rot1 @ center_t_c
    k1 = vect1[1] / vect1[0] if abs(vect1[0]) > 1e-10 else np.inf

    vect2 = rot2 @ center_t_c
    k2 = vect2[1] / vect2[0] if abs(vect2[0]) > 1e-10 else np.inf

    # Non-central voxels in transformed space (transpose for column operations)
    v = vers_noncentral.T  # shape (3, m)

    # Cut into 4 clusters
    c1 = (v[1, :] > k1 * v[0, :]) & (v[1, :] > k2 * v[0, :])
    c2 = (v[1, :] < k1 * v[0, :]) & (v[1, :] < k2 * v[0, :])
    c3 = (v[1, :] <= k1 * v[0, :]) & (v[1, :] >= k2 * v[0, :])
    c4 = (v[1, :] >= k1 * v[0, :]) & (v[1, :] <= k2 * v[0, :])

    clusters_trans = [v[:, c1], v[:, c2], v[:, c3], v[:, c4]]

    # Transform back to voxel space
    clusters_voxel = []
    for cluster in clusters_trans:
        if cluster.shape[1] == 0:
            clusters_voxel.append(np.empty((0, 3), dtype=int))
            continue
        cluster_back = (trans_mat @ cluster).T + center_source
        subs = np.round(cluster_back / voxel_size).astype(int)
        clusters_voxel.append(subs)

    # Identify quadrants by cluster centers
    centers = []
    for subs in clusters_voxel:
        if len(subs) > 0:
            centers.append(np.mean(subs, axis=0))
        else:
            centers.append(np.array([0.0, 0.0, 0.0]))

    # Anterior = max dim1 (AP direction), Posterior = min dim1
    # Interior = max dim0 (LR direction), Exterior = min dim0
    idx_a = np.argmax([c[1] for c in centers])
    idx_p = np.argmin([c[1] for c in centers])
    idx_i = np.argmax([c[0] for c in centers])
    idx_e = np.argmin([c[0] for c in centers])

    return clusters_voxel[idx_a], clusters_voxel[idx_p], \
           clusters_voxel[idx_i], clusters_voxel[idx_e]


def _find_nearest_axes(v):
    """
    Find nearest anatomical axes (R, A, S) to singular vectors.

    Uses greedy matching: each singular vector is assigned to the closest
    anatomical axis that hasn't been taken yet.

    Parameters
    ----------
    v : np.ndarray, shape (3, 3)
        Singular vectors as columns

    Returns
    -------
    directions : list of 3 ints
        Indices (0, 1, 2) mapping each SV to nearest axis (R=0, A=1, S=2)
    """
    identity = np.eye(3)

    # Compute angle matrix: angle_matrix[i, j] = angle between SV i and axis j
    angle_matrix = np.full((3, 3), 90.0)
    for i in range(3):
        vec = v[:, i]
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            continue
        for j in range(3):
            cos_angle = np.clip(np.dot(vec, identity[:, j]) / norm, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            angle_matrix[i, j] = angle if angle <= 90 else 180 - angle

    # Greedy assignment: pick smallest angle, mark that axis as used
    directions = []
    used_axes = set()
    for _ in range(3):
        # Mask already-used axes
        masked = angle_matrix.copy()
        for used in used_axes:
            masked[:, used] = 90
        # Find the SV-axis pair with the smallest angle
        sv_idx = np.argmin(masked.max(axis=1))
        axis_idx = int(np.argmin(masked[sv_idx]))
        directions.append(axis_idx)
        used_axes.add(axis_idx)

    return directions
