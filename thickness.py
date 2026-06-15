"""
Cartilage Thickness Measurement

Surface normal estimation and ray-triangle intersection based thickness
measurement, translated from CartiMorph MATLAB code.

Functions:
    estimate_surface_normals() - SVD-based surface normal estimation
    smooth_surface_normals() - Neighbor-weighted smoothing
    thickness_map_sn() - SN-based thickness via ray-triangle intersection
    triangle_ray_intersection() - Möller-Trumbore algorithm
"""

import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# Surface Normal Estimation
# ============================================================

def estimate_surface_normals(vertices, n_neighbors=10):
    """
    Estimate surface normals using local SVD.

    For each vertex, find its k nearest neighbors, center them, compute SVD,
    and take the smallest singular vector as the surface normal.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, 3)
        Vertex coordinates
    n_neighbors : int
        Number of neighbors for local SVD

    Returns
    -------
    sn : np.ndarray, shape (nv, 3)
        Unit surface normals
    """
    nv = len(vertices)
    sn = np.zeros((nv, 3))

    tree = cKDTree(vertices)

    for i in range(nv):
        # Find k nearest neighbors (including the point itself)
        _, idx_neigh = tree.query(vertices[i], k=n_neighbors)

        # Get neighbor coordinates and center
        pool = vertices[idx_neigh]
        pool_centered = pool - np.mean(pool, axis=0)

        # SVD - the last column of V is the normal direction
        _, _, vt = np.linalg.svd(pool_centered, full_matrices=False)
        normal = vt[-1, :]

        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            sn[i] = normal / norm

    return sn


def smooth_surface_normals(sn, vertices, n_neighbors=10):
    """
    Smooth surface normals by averaging with neighbors.

    Parameters
    ----------
    sn : np.ndarray, shape (nv, 3)
        Input surface normals
    vertices : np.ndarray, shape (nv, 3)
        Vertex coordinates
    n_neighbors : int
        Number of neighbors for averaging

    Returns
    -------
    sn_smooth : np.ndarray, shape (nv, 3)
        Smoothed unit surface normals
    """
    nv = len(vertices)
    sn_smooth = np.zeros_like(sn)

    tree = cKDTree(vertices)

    for i in range(nv):
        _, idx_neigh = tree.query(vertices[i], k=n_neighbors)

        # Average normals of neighbors
        avg_normal = np.mean(sn[idx_neigh], axis=0)

        # Normalize
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-10:
            sn_smooth[i] = avg_normal / norm
        else:
            sn_smooth[i] = sn[i]

    return sn_smooth


# ============================================================
# Ray-Triangle Intersection (Möller-Trumbore)
# ============================================================

def triangle_ray_intersection(origin, direction, p0, p1, p2,
                               border='inclusive'):
    """
    Möller-Trumbore ray-triangle intersection algorithm.

    Tests a single ray against multiple triangles.

    Parameters
    ----------
    origin : np.ndarray, shape (3,)
        Ray origin
    direction : np.ndarray, shape (3,)
        Ray direction (need not be normalized)
    p0, p1, p2 : np.ndarray, shape (nt, 3)
        Triangle vertices
    border : str
        'inclusive' includes edge/vertex intersections

    Returns
    -------
    intersect : np.ndarray, shape (nt, bool)
        Whether each triangle is intersected
    t : np.ndarray, shape (nt,)
        Ray parameter at intersection
    xcoor : np.ndarray, shape (nt, 3)
        Intersection coordinates
    """
    nt = len(p0)
    eps = 1e-10

    # Edge vectors
    e1 = p1 - p0  # (nt, 3)
    e2 = p2 - p0

    # Begin calculating determinant - also used to calculate u parameter
    pvec = np.cross(direction, e2)  # (nt, 3)
    det = np.sum(e1 * pvec, axis=1)  # (nt,)

    valid = np.abs(det) > eps

    # Initialize outputs
    intersect = np.zeros(nt, dtype=bool)
    t = np.full(nt, np.inf)
    xcoor = np.full((nt, 3), np.nan)

    if not np.any(valid):
        return intersect, t, xcoor

    # Only process valid triangles
    det_v = det[valid]
    e1_v = e1[valid]
    e2_v = e2[valid]
    p0_v = p0[valid]

    inv_det = 1.0 / det_v

    # Calculate distance from p0 to ray origin
    tvec = origin - p0_v  # (nv, 3)

    # Calculate u parameter and test bounds
    u = np.sum(tvec * pvec[valid], axis=1) * inv_det

    if border == 'inclusive':
        mask_u = (u >= -eps) & (u <= 1.0 + eps)
    else:
        mask_u = (u > eps) & (u < 1.0 - eps)

    if not np.any(mask_u):
        return intersect, t, xcoor

    # Prepare to test v parameter
    qvec = np.cross(tvec[mask_u], e1_v[mask_u])
    det_uv = det_v[mask_u]
    inv_det_uv = 1.0 / det_uv
    e2_uv = e2_v[mask_u]

    # Calculate v parameter and test bounds
    v = np.sum(direction * qvec, axis=1) * inv_det_uv

    if border == 'inclusive':
        mask_v = (v >= -eps) & (u[mask_u] + v <= 1.0 + eps)
    else:
        mask_v = (v > eps) & (u[mask_u] + v < 1.0 - eps)

    if not np.any(mask_v):
        return intersect, t, xcoor

    # Calculate t - ray parameter
    t_val = np.sum(e2_uv[mask_v] * qvec[mask_v], axis=1) * inv_det_uv[mask_v]

    # Only positive t (intersection in ray direction)
    pos = t_val > eps

    # Map back to original indices
    idx_valid = np.where(valid)[0]
    idx_u = idx_valid[mask_u]
    idx_v = idx_u[mask_v]
    idx_final = idx_v[pos]

    intersect[idx_final] = True
    t[idx_final] = t_val[pos]

    # Intersection coordinates
    xcoor[idx_final] = origin + np.outer(t_val[pos], direction)

    return intersect, t, xcoor


# ============================================================
# Thickness Measurement
# ============================================================

def _cast_ray_thickness(origin, direction, p0, p1, p2, depth):
    """Cast a single ray and return the thickness (nearest intersection distance), or 0."""
    if np.linalg.norm(direction) < 1e-10:
        return 0.0

    intersect, _, xcoor = triangle_ray_intersection(
        origin, direction, p0, p1, p2, border='inclusive'
    )

    if not np.any(intersect):
        return 0.0

    intersections = xcoor[intersect]
    distance = np.min(np.linalg.norm(intersections - origin, axis=1))

    return distance if distance < depth else 0.0


def thickness_map_sn(vertices, sn, mesh_outer, depth=20.0):
    """
    SN-based cartilage thickness measurement.

    For each vertex on the inner (bone-cartilage) surface, cast a ray
    along its surface normal and find intersection with the outer
    (cartilage-synovial) surface. The distance is the local thickness.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, 3)
        Inner surface vertex coordinates (voxel space)
    sn : np.ndarray, shape (nv, 3)
        Surface normals at each vertex
    mesh_outer : dict
        'vertices': (nvo, 3), 'faces': (nfo, 3) - outer surface mesh
    depth : float
        Maximum measuring depth (voxels)

    Returns
    -------
    thickness_map : np.ndarray, shape (nv, 4)
        Columns: x, y, z, thickness
    """
    nv = len(vertices)
    thickness_map = np.column_stack([vertices, np.zeros(nv)])

    faces_outer = mesh_outer['faces']
    vers_outer = mesh_outer['vertices'].astype(np.float64)

    if len(faces_outer) == 0:
        return thickness_map

    # Get triangle vertices for all outer faces
    p0 = vers_outer[faces_outer[:, 0], :]
    p1 = vers_outer[faces_outer[:, 1], :]
    p2 = vers_outer[faces_outer[:, 2], :]

    for i in range(nv):
        thickness_map[i, 3] = _cast_ray_thickness(
            vertices[i], sn[i], p0, p1, p2, depth
        )

    return thickness_map


def thickness_map_sn_fast(vertices, sn, mesh_outer, depth=20.0,
                          max_vertices=50000):
    """
    Faster thickness measurement with precomputed vertex-to-face mapping.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, 3)
    sn : np.ndarray, shape (nv, 3)
    mesh_outer : dict
    depth : float
    max_vertices : int
        Subsample if more vertices (for speed)

    Returns
    -------
    thickness_map : np.ndarray, shape (nv, 4)
    """
    nv = len(vertices)
    thickness_map = np.column_stack([vertices, np.zeros(nv)])

    faces_outer = mesh_outer['faces']
    vers_outer = mesh_outer['vertices'].astype(np.float64)

    if len(faces_outer) == 0:
        return thickness_map

    # Precompute vertex-to-face mapping for O(1) lookup
    vertex_to_faces = {}
    for fi in range(len(faces_outer)):
        for vi in faces_outer[fi]:
            if vi not in vertex_to_faces:
                vertex_to_faces[vi] = []
            vertex_to_faces[vi].append(fi)

    # Precompute triangle vertex arrays
    p0 = vers_outer[faces_outer[:, 0], :]
    p1 = vers_outer[faces_outer[:, 1], :]
    p2 = vers_outer[faces_outer[:, 2], :]

    # Build KD-tree for outer vertices
    tree = cKDTree(vers_outer)

    # Subsample vertices if too many
    if nv > max_vertices:
        indices = np.random.choice(nv, max_vertices, replace=False)
    else:
        indices = np.arange(nv)

    for idx in indices:
        origin = vertices[idx]
        direction = sn[idx]

        if np.linalg.norm(direction) < 1e-10:
            continue

        # Find nearby outer vertices via KD-tree
        nearby_verts = tree.query_ball_point(origin, depth * 1.5)
        if len(nearby_verts) == 0:
            continue

        # Collect faces containing nearby vertices
        nearby_faces = set()
        for vi in nearby_verts:
            nearby_faces.update(vertex_to_faces.get(vi, []))

        if not nearby_faces:
            continue

        face_indices = np.array(list(nearby_faces))
        thickness_map[idx, 3] = _cast_ray_thickness(
            origin, direction,
            p0[face_indices], p1[face_indices], p2[face_indices],
            depth
        )

    return thickness_map
