"""
Mesh Utilities for CartiMorph Measurement

Functions for converting 3D masks to triangular meshes, calculating mesh
properties, boundary detection, and surface extraction.

Translated from MATLAB CartiMorph code (CM_cal_mask2mesh, CM_quant_triMeshArea,
CM_cal_detectSurfaceBoundary, CM_cal_extractFaces_OR, etc.)
"""

import numpy as np
from scipy import ndimage


# ============================================================
# Mask to Mesh (marching cubes)
# ============================================================

def mask_to_mesh(mask_3d):
    """
    Convert a 3D binary mask to a triangular mesh using marching cubes.

    Parameters
    ----------
    mask_3d : np.ndarray
        3D binary mask (uint8 or bool)

    Returns
    -------
    dict
        'vertices': (nv, 3) vertex coordinates in voxel space
        'faces': (nf, 3) face indices (0-based)
    """
    from skimage.measure import marching_cubes

    # Handle empty or full masks
    if np.sum(mask_3d) == 0 or np.all(mask_3d):
        return {
            'vertices': np.empty((0, 3), dtype=np.int32),
            'faces': np.empty((0, 3), dtype=np.int32),
        }

    # marching_cubes expects the surface at 0.5 for binary masks
    # Use ~mask to get the outer boundary (consistent with MATLAB isosurface(~mask, 0))
    verts, faces, _, _ = marching_cubes(~mask_3d.astype(bool), level=0)

    # marching_cubes returns (row, col, slice) = (dim0, dim1, dim2)
    # which matches voxel subscripts directly
    verts = np.round(verts).astype(np.int32)

    # Remove duplicated vertices and unreferenced vertices
    verts, faces = remove_duplicated_vertices(verts, faces)
    verts, faces = remove_unreferenced_vertices(verts, faces)

    return {'vertices': verts, 'faces': faces}


# ============================================================
# Triangle Mesh Area
# ============================================================

def calculate_tri_mesh_area(vertices, faces):
    """
    Calculate area of each triangle in a mesh.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, 3)
    faces : np.ndarray, shape (nf, 3), 0-based indices

    Returns
    -------
    np.ndarray, shape (nf,)
        Area of each triangle
    """
    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :]
    v2 = vertices[faces[:, 2], :]

    vec1 = v1 - v0
    vec2 = v2 - v0

    cross_prod = np.cross(vec1, vec2)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    return areas


def triangle_mesh_area_total(vertices, faces):
    """Return total surface area of a mesh."""
    return np.sum(calculate_tri_mesh_area(vertices, faces))


# ============================================================
# Mesh Topology
# ============================================================

def faces_to_edges(faces):
    """
    Extract all edges from faces.

    Parameters
    ----------
    faces : np.ndarray, shape (nf, 3)

    Returns
    -------
    np.ndarray, shape (ne, 2)
        Sorted edge pairs (each edge appears once per face)
    """
    e0 = np.sort(faces[:, [0, 1]], axis=1)
    e1 = np.sort(faces[:, [1, 2]], axis=1)
    e2 = np.sort(faces[:, [0, 2]], axis=1)
    edges = np.vstack([e0, e1, e2])
    return edges


def _build_edge_to_faces(faces):
    """Build a mapping from sorted edge pairs to face indices."""
    edge_to_faces = {}
    for fi in range(len(faces)):
        for v0, v1 in [(faces[fi, 0], faces[fi, 1]),
                        (faces[fi, 1], faces[fi, 2]),
                        (faces[fi, 0], faces[fi, 2])]:
            key = (min(v0, v1), max(v0, v1))
            if key not in edge_to_faces:
                edge_to_faces[key] = []
            edge_to_faces[key].append(fi)
    return edge_to_faces


def neigh_faces_for_edges(faces):
    """
    Find neighboring faces for each unique edge.

    Returns
    -------
    neigh_faces : list of lists
        neigh_faces[i] contains face indices sharing edge i
    n_neigh : np.ndarray
        Number of neighboring faces per edge
    """
    edge_to_faces = _build_edge_to_faces(faces)
    unique_edges = list(edge_to_faces.keys())
    neigh_faces = [edge_to_faces[e] for e in unique_edges]
    n_neigh = np.array([len(nf) for nf in neigh_faces])

    return neigh_faces, n_neigh


def detect_surface_boundary(faces):
    """
    Detect boundary edges and faces of a surface mesh.

    Boundary edges have exactly one adjacent face.

    Parameters
    ----------
    faces : np.ndarray, shape (nf, 3)

    Returns
    -------
    margin_edges : list of tuples
        Boundary edge vertex pairs
    margin_faces : np.ndarray
        Indices of boundary faces
    """
    edge_to_faces = _build_edge_to_faces(faces)

    margin_edges = []
    margin_face_set = set()
    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 1:
            margin_edges.append(edge)
            margin_face_set.add(face_list[0])

    margin_faces = np.array(sorted(margin_face_set), dtype=int)

    return margin_edges, margin_faces


# ============================================================
# Mesh Processing (MPT functions)
# ============================================================

def remove_duplicated_vertices(vertices, faces):
    """
    Remove duplicated vertices and update face indices.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, 3)
    faces : np.ndarray, shape (nf, 3)

    Returns
    -------
    vertices_out : np.ndarray
    faces_out : np.ndarray
    """
    # Find unique vertices
    _, unique_idx, inverse_idx = np.unique(
        vertices, axis=0, return_index=True, return_inverse=True
    )

    # Sort by original order
    sort_order = np.argsort(unique_idx)
    new_idx = np.empty(len(sort_order), dtype=int)
    new_idx[sort_order] = np.arange(len(sort_order))

    vertices_out = vertices[unique_idx[sort_order]]
    # Map old vertex indices to new ones
    faces_out = new_idx[inverse_idx[faces.ravel()]].reshape(faces.shape)

    return vertices_out, faces_out


def remove_unreferenced_vertices(vertices, faces):
    """
    Remove vertices not referenced by any face.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, 3)
    faces : np.ndarray, shape (nf, 3)

    Returns
    -------
    vertices_out : np.ndarray
    faces_out : np.ndarray
    """
    if len(faces) == 0:
        return np.empty((0, 3), dtype=vertices.dtype), np.empty((0, 3), dtype=faces.dtype)

    referenced = np.unique(faces.ravel())
    if len(referenced) == len(vertices):
        return vertices, faces

    # Create mapping from old to new indices
    max_idx = np.max(faces) + 1
    old_to_new = np.full(max_idx, -1, dtype=int)
    old_to_new[referenced] = np.arange(len(referenced))

    vertices_out = vertices[referenced]
    faces_out = old_to_new[faces]

    return vertices_out, faces_out


def extract_faces_with_vertices(faces, vertices, target_vertices):
    """
    Extract faces that have at least one vertex in the target set.

    Parameters
    ----------
    faces : np.ndarray, shape (nf, 3)
    vertices : np.ndarray, shape (nv, 3)
    target_vertices : np.ndarray, shape (nt, 3)

    Returns
    -------
    faces_out : np.ndarray
        Extracted faces (indices refer to original vertices array)
    """
    # Find which vertex indices are in the target set
    # Use set-based lookup for efficiency
    target_set = set(map(tuple, target_vertices.tolist()))
    vertex_mask = np.array([tuple(v) in target_set for v in vertices])

    # Find faces where at least one vertex is in target set
    face_mask = np.any(vertex_mask[faces], axis=1)
    faces_out = faces[face_mask]

    return faces_out


def segment_connected_components(faces, mode='explicit'):
    """
    Segment mesh into connected components.

    Parameters
    ----------
    faces : np.ndarray, shape (nf, 3)
    mode : str
        'explicit' returns list of face arrays

    Returns
    -------
    components : list of np.ndarray
        Each element is a faces array for one component
    """
    nf = len(faces)
    if nf == 0:
        return []

    edge_to_faces = _build_edge_to_faces(faces)

    # BFS to find connected components
    visited = np.zeros(nf, dtype=bool)
    components = []

    for start in range(nf):
        if visited[start]:
            continue

        # BFS
        queue = [start]
        visited[start] = True
        component_faces = [start]

        while queue:
            current = queue.pop(0)
            # Find neighbors via shared edges
            for v0, v1 in [(faces[current, 0], faces[current, 1]),
                            (faces[current, 1], faces[current, 2]),
                            (faces[current, 0], faces[current, 2])]:
                key = (min(v0, v1), max(v0, v1))
                for neighbor in edge_to_faces.get(key, []):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        component_faces.append(neighbor)

        components.append(faces[np.array(component_faces)])

    return components


def delete_small_components(components, min_size):
    """
    Keep only components with >= min_size faces.

    Returns
    -------
    faces_out : np.ndarray
    """
    kept = [comp for comp in components if len(comp) >= min_size]
    if not kept:
        return np.empty((0, 3), dtype=int)
    return np.vstack(kept)


def delete_large_components(components, max_size):
    """
    Keep only components with < max_size faces.

    Returns
    -------
    faces_out : np.ndarray
    """
    kept = [comp for comp in components if len(comp) < max_size]
    if not kept:
        return np.empty((0, 3), dtype=int)
    return np.vstack(kept)


# ============================================================
# Surface Extraction
# ============================================================

def extract_outer_surface(mask_3d):
    """
    Extract the outer surface (cartilage-air interface) of a 3D mask.

    The outer surface is the boundary between mask voxels and background.

    Parameters
    ----------
    mask_3d : np.ndarray, shape (X, Y, Z), binary

    Returns
    -------
    dict : mesh with 'vertices' and 'faces'
    """
    return mask_to_mesh(mask_3d)


def extract_inner_surface(mask_cartilage, mask_bone, voxel_size):
    """
    Extract the inner surface (bone-cartilage interface).

    The inner surface consists of cartilage boundary voxels that are
    adjacent to bone voxels.

    Parameters
    ----------
    mask_cartilage : np.ndarray, binary
    mask_bone : np.ndarray, binary
    voxel_size : np.ndarray, shape (3,)

    Returns
    -------
    dict : mesh with 'vertices' and 'faces'
    """
    # Dilate bone mask by 1 voxel to catch adjacent cartilage voxels
    bone_dilated = ndimage.binary_dilation(mask_bone, iterations=1)

    # Inner surface = cartilage voxels adjacent to bone
    inner_mask = mask_cartilage & bone_dilated

    if np.sum(inner_mask) == 0:
        return {'vertices': np.empty((0, 3)), 'faces': np.empty((0, 3), dtype=int)}

    return mask_to_mesh(inner_mask.astype(np.uint8))


_STRUCT_6CONN = ndimage.generate_binary_structure(3, 1)


def surface_closing(mask_3d, iterations=1):
    """
    Morphological closing to repair small holes in the surface.

    Parameters
    ----------
    mask_3d : np.ndarray, binary
    iterations : int

    Returns
    -------
    np.ndarray : closed mask
    """
    closed = ndimage.binary_closing(mask_3d, structure=_STRUCT_6CONN, iterations=iterations)
    return closed.astype(mask_3d.dtype)


def surface_dilation(mask_3d, iterations=1):
    """Dilate a binary mask."""
    return ndimage.binary_dilation(mask_3d, structure=_STRUCT_6CONN, iterations=iterations).astype(mask_3d.dtype)


def surface_erosion(mask_3d, iterations=1):
    """Erode a binary mask."""
    return ndimage.binary_erosion(mask_3d, structure=_STRUCT_6CONN, iterations=iterations).astype(mask_3d.dtype)


def bwareaopen_3d(mask_3d, min_size=10, connectivity=26):
    """
    Remove small connected components from a 3D binary mask.

    Parameters
    ----------
    mask_3d : np.ndarray, binary
    min_size : int
        Minimum number of voxels to keep
    connectivity : int
        6, 18, or 26

    Returns
    -------
    np.ndarray : cleaned mask
    """
    conn_to_rank = {6: 1, 18: 2, 26: 3}
    rank = conn_to_rank.get(connectivity, 3)
    struct = ndimage.generate_binary_structure(3, rank)

    labeled, num_features = ndimage.label(mask_3d, structure=struct)

    if num_features == 0:
        return mask_3d

    # Count voxels per component
    component_sizes = np.bincount(labeled.ravel())
    # component_sizes[0] is background

    # Keep components >= min_size
    keep_labels = np.where(component_sizes >= min_size)[0]
    keep_labels = keep_labels[keep_labels > 0]  # exclude background

    mask_out = np.isin(labeled, keep_labels)
    return mask_out.astype(mask_3d.dtype)


def convert_voxel_indices_to_mask_3d(indices, img_size):
    """
    Convert voxel indices to a 3D binary mask.

    Parameters
    ----------
    indices : np.ndarray, shape (n, 3), integer voxel coordinates
    img_size : tuple of 3 ints

    Returns
    -------
    np.ndarray, shape img_size, binary
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    valid = (
        (indices[:, 0] >= 0) & (indices[:, 0] < img_size[0]) &
        (indices[:, 1] >= 0) & (indices[:, 1] < img_size[1]) &
        (indices[:, 2] >= 0) & (indices[:, 2] < img_size[2])
    )
    idx = indices[valid]
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return mask
