#!/usr/bin/env python3
"""
GPU CartiMorph morphometric analysis.

Inputs
------
1) subject segmentation (femur, femoral cartilage, tibia, medial/lateral tibial cartilage)
2) registration result, accepted as either:
   - the warped template tissue segmentation used by CartiMorph (preferred),
   - a warped 20-ROI atlas (accepted and collapsed back to FC/mTC/lTC prior), or
   - a VoxelMorph-style dense deformation field plus --template-seg.

Outputs
-------
Primary:
  MorphQuant.csv: 4 x 20 in MATLAB CartiMorph order
    FCL, Mean Thickness, Surface Area, Volume
Final reporting sidecars:
  MorphQuant_Compartments.csv: MTC/cMFC/LTC/cLFC direct union-domain metrics
  FCL_Areas.csv: ROI tAB/cAB/dAB/FCL audit quantities

MATLAB compatibility notes
--------------------------
* ROI order follows CartiMorphToolbox.cal_MorphQuant2table.
* FCL (%) = 100*max((tAB-cAB)/tAB, 0), NaN when tAB == 0.
* Mean Thickness follows quant_ThCtAB: thickness values over covered interface
  vertices, zero-padded for denuded subchondral-bone vertices.
* Volume defaults to physical voxel volume sx*sy*sz.  The historical
  norm(voxSize) behavior is available only through --legacy-matlab-volume-norm.
* MATLAB round() semantics (ties away from zero) are used where voxel subscripts
  are reconstructed.
* Expensive morphometric reductions, nearest-neighbour mappings, PCA normals and
  ray/triangle intersections use torch tensors on CUDA.

The registration result is used only as the warped template cartilage prior for
subchondral-bone/FCL reconstruction, as in the MATLAB Toolbox.  The final 20 ROI
partition is generated from reconstructed iC/scB surfaces by the MATLAB FC/TC
SurfaceParcellation rules; registration labels are not used as final ROI labels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure


ROI_NAMES = [
    "aMFC", "ecMFC", "ccMFC", "icMFC", "pMFC",
    "aLFC", "ecLFC", "ccLFC", "icLFC", "pLFC",
    "aMTC", "eMTC", "pMTC", "iMTC", "cMTC",
    "aLTC", "eLTC", "pLTC", "iLTC", "cLTC",
]
ROW_NAMES = ["FCL", "Mean Thickness", "Surface Area", "Volume"]


@dataclass
class LabelConfig:
    background: int = 0
    femur: int = 1
    femoral_cartilage: int = 2
    tibia: int = 3
    medial_tibial_cartilage: int = 4
    lateral_tibial_cartilage: int = 5


@dataclass
class Mesh:
    vertices_sub: np.ndarray  # MATLAB-like 1-based voxel subscripts, shape [N,3]
    faces: np.ndarray         # zero-based vertex IDs, shape [M,3]

    @property
    def empty(self) -> bool:
        return self.vertices_sub.size == 0 or self.faces.size == 0


# -----------------------------------------------------------------------------
# Minimal NIfTI I/O (read-only) so this module does not require nibabel.
# It supports the scalar/vector datatypes used by CartiMorph outputs.
# -----------------------------------------------------------------------------
_NIFTI_DTYPES = {
    2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8",
    256: "i1", 512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
}


def load_nifti(path: str | os.PathLike) -> Tuple[np.ndarray, np.ndarray]:
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        hdr = f.read(348)
        if len(hdr) != 348:
            raise ValueError(f"Invalid NIfTI header: {path}")
        sh_le = struct.unpack("<i", hdr[0:4])[0]
        sh_be = struct.unpack(">i", hdr[0:4])[0]
        if sh_le == 348:
            endian = "<"
        elif sh_be == 348:
            endian = ">"
        else:
            raise ValueError(f"Not a NIfTI-1 file: {path}")
        dim = struct.unpack(endian + "8h", hdr[40:56])
        datatype = struct.unpack(endian + "h", hdr[70:72])[0]
        pixdim = struct.unpack(endian + "8f", hdr[76:108])
        vox_offset = int(round(struct.unpack(endian + "f", hdr[108:112])[0]))
        slope = struct.unpack(endian + "f", hdr[112:116])[0]
        inter = struct.unpack(endian + "f", hdr[116:120])[0]
        if datatype not in _NIFTI_DTYPES:
            raise ValueError(f"Unsupported NIfTI datatype code {datatype}")
        shape = tuple(int(x) for x in dim[1: dim[0] + 1])
        f.seek(vox_offset)
        count = int(np.prod(shape))
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.dtype(endian + _NIFTI_DTYPES[datatype]), count=count)
    # NIfTI stores the first axis fastest; MATLAB/nibabel logical shape is Fortran order.
    arr = arr.reshape(shape, order="F")
    if slope not in (0.0, 1.0):
        arr = arr.astype(np.float64) * slope + inter
    elif inter != 0.0:
        arr = arr.astype(np.float64) + inter
    vox = np.asarray(pixdim[1:4], dtype=np.float64)
    return arr, vox


def load_volume(path: str | os.PathLike) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p), None
    if p.endswith(".npz"):
        z = np.load(p)
        key = "arr_0" if "arr_0" in z else list(z.keys())[0]
        return z[key], None
    if p.endswith(".nii") or p.endswith(".nii.gz"):
        return load_nifti(p)
    raise ValueError(f"Unsupported input format: {path}")


# -----------------------------------------------------------------------------
# MATLAB numeric/voxel compatibility helpers
# -----------------------------------------------------------------------------
def matlab_round(x: np.ndarray | float) -> np.ndarray:
    """MATLAB round: halves away from zero (not NumPy bankers rounding)."""
    a = np.asarray(x)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def _valid_subs(subs1: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    return np.all((subs1 >= 1) & (subs1 <= np.asarray(shape)[None, :]), axis=1)


def _subs1_to_idx0(subs1: np.ndarray) -> np.ndarray:
    return subs1.astype(np.int64) - 1


def _sphere1() -> np.ndarray:
    x, y, z = np.mgrid[-1:2, -1:2, -1:2]
    return (x*x + y*y + z*z) <= 1


def _foreground_bbox(mask: np.ndarray, pad: int = 0) -> Tuple[Tuple[slice, ...], np.ndarray]:
    """Return a tight foreground bounding box and its 0-based origin.

    The box contains *all* foreground voxels plus ``pad`` background voxels where
    the image boundary permits.  Operations whose support is local, or connected-
    component / hole operations with the full foreground present, are therefore
    exactly equivalent inside this box while avoiding scans over empty OAI volume.
    """
    a = np.asarray(mask, dtype=bool)
    pts = np.argwhere(a)
    if len(pts) == 0:
        z = np.zeros(a.ndim, dtype=np.int64)
        return tuple(slice(0, 0) for _ in range(a.ndim)), z
    lo = np.maximum(pts.min(axis=0) - int(pad), 0)
    hi = np.minimum(pts.max(axis=0) + int(pad) + 1, np.asarray(a.shape))
    return tuple(slice(int(lo[d]), int(hi[d])) for d in range(a.ndim)), lo.astype(np.int64)


def preprocess_binary(mask: np.ndarray, n_vox_tbr: int) -> np.ndarray:
    """Port of cal_preprocessImg, evaluated only on the exact foreground bbox.

    Cropping is result-preserving: the crop contains every foreground voxel and a
    one-voxel exterior background collar whenever one exists.  Thus 26-connected
    component sizes and 6-connected hole filling are identical to full-volume
    evaluation; only empty background work is removed.
    """
    src = np.asarray(mask, dtype=bool)
    if not np.any(src):
        return np.zeros(src.shape, dtype=bool)
    sl, _ = _foreground_bbox(src, pad=1)
    m = src[sl].copy()
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3), dtype=bool))
    if n:
        counts = np.bincount(lab.ravel())
        keep = counts >= int(max(n_vox_tbr, 1))
        keep[0] = False
        m = keep[lab]
    st6 = np.zeros((3, 3, 3), dtype=bool)
    st6[1, 1, :] = True
    st6[1, :, 1] = True
    st6[:, 1, 1] = True
    m = ndimage.binary_fill_holes(m, structure=st6)
    out = np.zeros(src.shape, dtype=bool)
    out[sl] = m
    return out


def bwperim2d(a: np.ndarray) -> np.ndarray:
    st4 = ndimage.generate_binary_structure(2, 1)
    return a.astype(bool) & ~ndimage.binary_erosion(a.astype(bool), structure=st4, border_value=0)


def get_boundary2d_3d(vol: np.ndarray, slicing_dim: int) -> np.ndarray:
    """Slice-wise 4-connected perimeter without a Python loop.

    A 3-D erosion whose structuring element has support only inside the requested
    slicing plane is exactly equivalent to applying ``bwperim2d`` independently
    to every slice, including image-border handling.
    """
    a = np.asarray(vol, dtype=bool)
    if not np.any(a):
        return np.zeros(a.shape, dtype=bool)
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1, 1, 1] = True
    if slicing_dim == 0:       # independent Y-Z slices
        st[1, 0, 1] = st[1, 2, 1] = True
        st[1, 1, 0] = st[1, 1, 2] = True
    elif slicing_dim == 1:     # independent X-Z slices
        st[0, 1, 1] = st[2, 1, 1] = True
        st[1, 1, 0] = st[1, 1, 2] = True
    elif slicing_dim == 2:     # independent X-Y slices
        st[0, 1, 1] = st[2, 1, 1] = True
        st[1, 0, 1] = st[1, 2, 1] = True
    else:
        raise ValueError(f"slicing_dim must be 0, 1 or 2, got {slicing_dim}")
    return a & ~ndimage.binary_erosion(a, structure=st, border_value=0)


def _cartilage_bone_grown_local(cart: np.ndarray, bone: np.ndarray):
    """Return CartiMorph's locally evaluated ``bone_grown`` mask and bbox origin.

    This is the exact preprocessing already used by ``init_boundary_split_w_bone``:
    radius-1 bone dilation excluding cartilage, 26-connected small-component
    filtering, then filling the cartilage+bone union.  Keeping this as one helper
    lets the surface-mesh split use the *same* geometric object as the voxel
    boundary equations instead of trying to rematch rounded marching-cubes
    vertices to an adjacent one-voxel interface set.
    """
    cart0 = np.asarray(cart, dtype=bool)
    bone0 = np.asarray(bone, dtype=bool)
    support = cart0 | bone0
    if not np.any(support):
        return np.empty((0, 0, 0), dtype=bool), np.zeros(3, dtype=np.int64), None
    sl, origin = _foreground_bbox(support, pad=2)
    c = cart0[sl]
    b = bone0[sl]

    bone_d = ndimage.binary_dilation(b, structure=_sphere1()) & ~c
    nmin = int(matlab_round(np.array([bone_d.sum() / 10.0]))[0])
    lab, n = ndimage.label(bone_d, structure=np.ones((3, 3, 3), bool))
    if n:
        cnt = np.bincount(lab.ravel())
        keep = cnt >= max(nmin, 1)
        keep[0] = False
        bone_d = keep[lab]
    cart_bone = ndimage.binary_fill_holes(c | bone_d)
    bone_grown = (cart_bone.astype(np.uint8) - c.astype(np.uint8)).astype(bool)
    return bone_grown, origin.astype(np.int64, copy=False), sl


def init_boundary_split_w_bone(cart: np.ndarray, bone: np.ndarray, need_outer: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Port of cal_initBoundarySplit_wBone / cal_initBoundarySplit2D_wBone.

    All foreground-dependent morphology is evaluated on the union bbox of bone and
    cartilage.  A two-voxel collar contains the radius-1 bone dilation plus an
    exterior background layer, so the result is exactly the same as full-volume
    evaluation while avoiding repeated scans of empty image space.
    """
    cart0 = np.asarray(cart, dtype=bool)
    bone0 = np.asarray(bone, dtype=bool)
    bone_grown, origin, sl = _cartilage_bone_grown_local(cart0, bone0)
    if sl is None:
        z = np.empty((0, 3), dtype=np.int64)
        return z, z.copy()
    c = cart0[sl]
    non_bone = ~bone_grown

    outputs = []
    for slicing_dim in (0, 1):
        b_cart = get_boundary2d_3d(c, slicing_dim)
        b_nonbone = get_boundary2d_3d(non_bone, slicing_dim)
        inter = b_cart & b_nonbone
        # split_cartilage_mesh only consumes the interface.  Retain outer output
        # for compatibility callers, but skip its argwhere/unique work when unused.
        if need_outer:
            outer = b_cart & ~inter
            outer_sub = np.argwhere(outer) + origin[None, :] + 1
        else:
            outer_sub = np.empty((0, 3), dtype=np.int64)
        # Local 0-based index + global origin + MATLAB 1-based offset.
        outputs.append((np.argwhere(inter) + origin[None, :] + 1, outer_sub))
    inter = np.unique(np.vstack([outputs[0][0], outputs[1][0]]), axis=0)
    if not need_outer:
        return inter.astype(np.int64, copy=False), np.empty((0, 3), dtype=np.int64)
    outer = np.unique(np.vstack([outputs[0][1], outputs[1][1]]), axis=0)
    if len(inter) and len(outer):
        # Exact row subtraction, replacing the old Python tuple/set loop.
        outer = outer[~_row_membership(outer, inter)]
    return inter.astype(np.int64, copy=False), outer.astype(np.int64, copy=False)


# -----------------------------------------------------------------------------
# Mesh helpers.  MATLAB's cal_getBoundaryMesh does isosurface, swaps x/y, rounds
# vertices to voxel subscripts, removes duplicated/unreferenced vertices.
# -----------------------------------------------------------------------------
def _dedup_mesh(vertices_sub: np.ndarray, faces: np.ndarray) -> Mesh:
    if len(vertices_sub) == 0 or len(faces) == 0:
        return Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
    v = vertices_sub.astype(np.int64)
    uniq, inv = np.unique(v, axis=0, return_inverse=True)
    f = inv[faces]
    # remove degenerate and duplicated triangles
    nondeg = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[nondeg]
    if len(f) == 0:
        return Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
    key = np.sort(f, axis=1)
    _, keep = np.unique(key, axis=0, return_index=True)
    f = f[np.sort(keep)]
    used = np.unique(f)
    remap = -np.ones(len(uniq), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(uniq[used].astype(np.float64), remap[f])


def boundary_mesh(mask: np.ndarray) -> Mesh:
    src = np.asarray(mask, dtype=bool)
    if not np.any(src):
        return Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
    # marching_cubes has no dependence on distant all-zero voxels.  Evaluate on
    # the exact foreground bbox, retain the same explicit one-voxel zero padding,
    # then translate vertices back to global MATLAB subscripts.
    sl, origin = _foreground_bbox(src, pad=0)
    p = np.pad(src[sl].astype(np.float32), 1, mode="constant")
    try:
        verts, faces, _, _ = measure.marching_cubes(p, level=0.5, allow_degenerate=False)
    except ValueError:
        return Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
    # In the old full-volume padded array, local padded coordinates are shifted by
    # exactly ``origin``.  The previous ``verts-=1; round(verts+1)`` cancels, so
    # this is algebraically identical to round(full_padded_verts).
    subs1 = matlab_round(verts + origin[None, :]).astype(np.int64)
    okv = _valid_subs(subs1, src.shape)
    okf = np.all(okv[faces], axis=1)
    return _dedup_mesh(subs1, faces[okf])


def _row_membership(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact integer-row membership without Python tuple/set loops."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    aa = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
    bb = np.ascontiguousarray(np.asarray(b, dtype=np.int64))
    # _row_keys_int is defined later in the module; lookup happens at call time.
    return np.isin(_row_keys_int(aa), _row_keys_int(bb))


def extract_faces_and(mesh: Mesh, target_subs: np.ndarray) -> np.ndarray:
    m = _row_membership(mesh.vertices_sub.astype(np.int64), target_subs.astype(np.int64))
    return mesh.faces[np.all(m[mesh.faces], axis=1)]


def extract_faces_or(mesh: Mesh, target_subs: np.ndarray) -> np.ndarray:
    m = _row_membership(mesh.vertices_sub.astype(np.int64), target_subs.astype(np.int64))
    return mesh.faces[np.any(m[mesh.faces], axis=1)]


def submesh(mesh: Mesh, faces: np.ndarray) -> Mesh:
    if faces.size == 0:
        return Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
    used = np.unique(faces)
    remap = -np.ones(len(mesh.vertices_sub), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.vertices_sub[used], remap[faces])


# -----------------------------------------------------------------------------
# Exact MATLAB volume parcellation rules used as a fallback when registration is
# a warped tissue segmentation rather than an already warped 20-ROI atlas.
# -----------------------------------------------------------------------------
def volume_parcellation_fc(mask_fc: np.ndarray, knee_side: str, cc_percentage: float = 0.6) -> np.ndarray:
    subs = np.argwhere(mask_fc) + 1
    atlas = np.zeros(mask_fc.shape, np.uint8)
    if len(subs) == 0:
        return atlas
    lr = np.sort(np.unique(subs[:, 0]))
    n = len(lr)
    # MATLAB indexing: round(n*.25):round(n*.75), inclusive and 1-based.
    lo = max(int(matlab_round(np.array([n * 0.25]))[0]), 1) - 1
    hi = max(int(matlab_round(np.array([n * 0.75]))[0]), 1) - 1
    center_lr = lr[lo:hi + 1]
    mins = np.array([subs[subs[:, 0] == x, 1].min() for x in center_lr])
    notch_ap = mins.max()
    cand = np.flatnonzero(mins == notch_ap)
    idx_mat = int(math.ceil(len(cand) / 2.0)) - 1
    notch_lr = center_lr[cand[idx_mat]]
    lfc = subs[subs[:, 0] >= notch_lr]
    mfc = subs[subs[:, 0] < notch_lr]
    if len(lfc) == 0 or len(mfc) == 0:
        return atlas

    def side_parts(s: np.ndarray, lateral: bool):
        min_ap = s[:, 1].min()
        min_c = notch_ap - int(matlab_round(np.array([(notch_ap - min_ap) * cc_percentage]))[0])
        c = s[(s[:, 1] < notch_ap) & (s[:, 1] > min_c)]
        a = s[(s[:, 1] <= s[:, 1].max()) & (s[:, 1] >= notch_ap)]
        p = s[(s[:, 1] <= min_c) & (s[:, 1] >= min_ap)]
        e, cc, ii = [], [], []
        for ap in np.unique(c[:, 1]) if len(c) else []:
            q = c[c[:, 1] == ap]
            mx, mn = q[:, 0].max(), q[:, 0].min()
            f1 = mn + (mx - mn) / 3.0
            f2 = mn + 2.0 * (mx - mn) / 3.0
            if lateral:
                em = (q[:, 0] <= mx) & (q[:, 0] > f2)
                cm = (q[:, 0] <= f2) & (q[:, 0] >= f1)
                im = (q[:, 0] < f1) & (q[:, 0] >= mn)
            else:
                im = (q[:, 0] <= mx) & (q[:, 0] > f2)
                cm = (q[:, 0] <= f2) & (q[:, 0] >= f1)
                em = (q[:, 0] < f1) & (q[:, 0] >= mn)
            e.append(q[em]); cc.append(q[cm]); ii.append(q[im])
        emp = lambda xs: np.vstack(xs) if xs and any(len(x) for x in xs) else np.empty((0, 3), int)
        return {"a": a, "p": p, "e": emp(e), "c": emp(cc), "i": emp(ii)}

    L = side_parts(lfc, True)
    M = side_parts(mfc, False)
    if knee_side.lower().startswith("l"):
        L, M = M, L
    mapping = {
        1: M["a"], 2: M["e"], 3: M["c"], 4: M["i"], 5: M["p"],
        6: L["a"], 7: L["e"], 8: L["c"], 9: L["i"], 10: L["p"],
    }
    for lab, s in mapping.items():
        if len(s):
            idx = _subs1_to_idx0(s.astype(int))
            atlas[tuple(idx.T)] = lab
    return atlas


def _principal_transform(vers: np.ndarray):
    center = vers.mean(axis=0)
    vc = vers - center
    _, s, vt = np.linalg.svd(vc, full_matrices=False)
    V = vt.T
    # MATLAB picks a unique nearest R/A/S axis for each singular vector.
    angles = np.degrees(np.arccos(np.clip(np.abs(V), 0, 1)))
    dirs = []
    used = set()
    for j in range(3):
        a = angles[:, j].copy()
        for u in used:
            a[u] = 90.0
        d = int(np.argmin(a))
        dirs.append(d)
        used.add(d)
    trans = np.zeros((3, 3), float)
    for j, d in enumerate(dirs):
        trans[:, d] = V[:, j]
    for d in range(3):
        if trans[d, d] < 0:
            trans[:, d] *= -1
    vct = np.linalg.solve(trans, vc.T).T
    return center, s, np.asarray(dirs), trans, vct


def _central_tc(vers: np.ndarray):
    center, s, dirs, trans, vct = _principal_transform(vers)
    # diagS(direction==2)/diagS(direction==1) in MATLAB; dirs are zero-based axis IDs.
    i2 = int(np.flatnonzero(dirs == 1)[0])
    i1 = int(np.flatnonzero(dirs == 0)[0])
    ratio = math.sqrt(float(s[i2] / s[i1])) if s[i1] != 0 else 1.0
    A = 5.0
    last_act = None
    oscill = 0
    idx = np.zeros(len(vers), bool)
    while True:
        B = A * ratio
        idx = (vct[:, 0] ** 2 / (A*A) + vct[:, 1] ** 2 / (B*B) - 1.0) < 0
        frac = idx.sum() / max(len(vers), 1)
        if abs(frac - 0.2) <= 0.005 or oscill >= 1000:
            break
        act = 1 if frac > 0.2 else -1
        A += -0.5 if act == 1 else 0.5
        if last_act is not None and act != last_act:
            oscill += 1
        last_act = act
        if A <= 0:
            A = 0.5
    return center, trans, vct, idx


def volume_parcellation_tc(mask_mtc: np.ndarray, mask_ltc: np.ndarray, knee_side: str, vox: np.ndarray) -> np.ndarray:
    atlas = np.zeros(mask_mtc.shape, np.uint8)
    sm = np.argwhere(mask_mtc) + 1
    sl = np.argwhere(mask_ltc) + 1
    if len(sm) < 3 or len(sl) < 3:
        return atlas
    vm = sm * vox[None, :]
    vl = sl * vox[None, :]
    cm, tm, vmct, idx_cm = _central_tc(vm)
    cl, tl, vlct, idx_cl = _central_tc(vl)

    def four_clusters(vct, idx_c, center, trans, other_center):
        v = vct[~idx_c]
        vec = np.linalg.solve(trans, (other_center - center))
        def slope(angle_deg):
            a = math.radians(angle_deg)
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0,0,1.]])
            z = R @ vec
            return z[1] / z[0] if abs(z[0]) > 1e-12 else math.copysign(np.inf, z[1])
        k1, k2 = slope(45), slope(-45)
        c1 = v[(v[:,1] > k1*v[:,0]) & (v[:,1] > k2*v[:,0])]
        c2 = v[(v[:,1] < k1*v[:,0]) & (v[:,1] < k2*v[:,0])]
        c3 = v[(v[:,1] <= k1*v[:,0]) & (v[:,1] >= k2*v[:,0])]
        c4 = v[(v[:,1] >= k1*v[:,0]) & (v[:,1] <= k2*v[:,0])]
        out = []
        for q in (c1,c2,c3,c4):
            out.append((trans @ q.T).T + center if len(q) else np.empty((0,3)))
        centers = np.array([q.mean(axis=0) if len(q) else [np.nan]*3 for q in out])
        return out, centers

    mc, mcent = four_clusters(vmct, idx_cm, cm, tm, cl)
    lc, lcent = four_clusters(vlct, idx_cl, cl, tl, cm)
    # MATLAB assignment rules.
    ma = int(np.nanargmax(mcent[:,1])); mp = int(np.nanargmin(mcent[:,1]));
    mi = int(np.nanargmax(mcent[:,0])); me = int(np.nanargmin(mcent[:,0]))
    la = int(np.nanargmax(lcent[:,1])); lp = int(np.nanargmin(lcent[:,1]));
    le = int(np.nanargmax(lcent[:,0])); li = int(np.nanargmin(lcent[:,0]))
    parts = {
        11: mc[ma], 12: mc[me], 13: mc[mp], 14: mc[mi],
        16: lc[la], 17: lc[le], 18: lc[lp], 19: lc[li],
    }
    if knee_side.lower().startswith("l"):
        parts[12], parts[14] = parts[14], parts[12]
        parts[17], parts[19] = parts[19], parts[17]
    # central regions are original points selected by the ellipse.
    parts[15] = vm[idx_cm]
    parts[20] = vl[idx_cl]
    for lab, vv in parts.items():
        if len(vv):
            ss = matlab_round(vv / vox[None,:]).astype(int)
            ok = _valid_subs(ss, atlas.shape)
            ss = ss[ok]
            if len(ss):
                ii = _subs1_to_idx0(ss)
                atlas[tuple(ii.T)] = lab
    return atlas


def build_roi_atlas_from_tissue_seg(reg_seg: np.ndarray, labels: LabelConfig, knee_side: str, vox: np.ndarray, cc_percentage=0.6):
    fc = reg_seg == labels.femoral_cartilage
    mtc = reg_seg == labels.medial_tibial_cartilage
    ltc = reg_seg == labels.lateral_tibial_cartilage
    return volume_parcellation_fc(fc, knee_side, cc_percentage) + volume_parcellation_tc(mtc, ltc, knee_side, vox)



# -----------------------------------------------------------------------------
# MATLAB rule-based *surface* parcellation used by the Toolbox regional analysis.
# These routines return independent per-ROI vertex masks (rather than a single
# label per vertex), because CM_cal_extractFaces_OR can make boundary faces count
# toward more than one neighboring ROI exactly as in MATLAB.
# -----------------------------------------------------------------------------
def _face_components(faces: np.ndarray) -> list[np.ndarray]:
    """MPT_segment_connected_components: triangles connect only by a shared edge."""
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return []
    n = len(faces)
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # MPT_find_neighbor_triangle_indices defines neighbors as sharing two
    # vertex IDs, i.e. one complete edge (not merely touching at a vertex).
    first_edge: dict[tuple[int, int], int] = {}
    for i, f in enumerate(faces):
        edges = (tuple(sorted((int(f[0]), int(f[1])))),
                 tuple(sorted((int(f[1]), int(f[2])))),
                 tuple(sorted((int(f[2]), int(f[0])))))
        for e in edges:
            if e in first_edge:
                union(i, first_edge[e])
            else:
                first_edge[e] = i
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [faces[np.asarray(ix, dtype=np.int64)] for ix in groups.values()]


def _component_percentages(comps: list[np.ndarray]) -> np.ndarray:
    if not comps:
        return np.empty((0,), dtype=np.int64)
    sizes = np.asarray([len(c) for c in comps], dtype=np.float64)
    return matlab_round(100.0 * sizes / sizes.sum()).astype(np.int64)


def _concat_faces(comps: list[np.ndarray], selector: np.ndarray) -> np.ndarray:
    chosen = [c for c, keep in zip(comps, selector.tolist()) if keep]
    return np.vstack(chosen) if chosen else np.empty((0, 3), dtype=np.int64)


def _used_vertices_mask(mesh: Mesh, faces: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mesh.vertices_sub), dtype=bool)
    if len(faces):
        out[np.unique(faces)] = True
    return out


def _faces_or_vertex_mask(mesh: Mesh, vmask: np.ndarray) -> np.ndarray:
    if mesh.empty or not np.any(vmask):
        return np.empty((0, 3), dtype=np.int64)
    return mesh.faces[np.any(vmask[mesh.faces], axis=1)]


def _region_membership_from_subs(vertices_sub: np.ndarray, region_subs: np.ndarray) -> np.ndarray:
    if len(region_subs) == 0:
        return np.zeros(len(vertices_sub), dtype=bool)
    return _row_membership(vertices_sub.astype(np.int64), np.asarray(region_subs, dtype=np.int64))


def _mesh_face_areas_numpy(mesh: Mesh, vox: np.ndarray) -> np.ndarray:
    if mesh.empty:
        return np.empty((0,), dtype=np.float64)
    p = mesh.vertices_sub.astype(np.float64) * vox[None, :]
    t = p[mesh.faces]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def _mesh_area_numpy(mesh: Mesh, vox: np.ndarray) -> float:
    """Exact total triangle area in mm^2; used only by optional diagnostics."""
    return float(_mesh_face_areas_numpy(mesh, vox).sum()) if not mesh.empty else 0.0


def _faces_area_numpy(parent: Mesh, faces: np.ndarray, vox: np.ndarray) -> float:
    """Area of a face subset on ``parent``; debug-only and numerically exact."""
    faces = np.asarray(faces, dtype=np.int64)
    if parent.empty or len(faces) == 0:
        return 0.0
    p = parent.vertices_sub.astype(np.float64) * vox[None, :]
    t = p[faces]
    ar = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    return float(ar.sum())


def surface_parcellation_fc(inner: Mesh, scb: Mesh, vox: np.ndarray,
                             knee_side: str, cc_percentage: float = 0.6
                             ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Port of CM_cal_SurfaceParcellation_FC.m / cal_SurfaceParcellation_FC.

    Returns (inner_masks, scb_masks), keyed by global ROI label 1..10.
    """
    im = {r: np.zeros(len(inner.vertices_sub), dtype=bool) for r in range(1, 11)}
    sm = {r: np.zeros(len(scb.vertices_sub), dtype=bool) for r in range(1, 11)}
    if inner.empty or scb.empty:
        return im, sm

    subs = matlab_round(scb.vertices_sub).astype(np.int64)
    subs_i = matlab_round(inner.vertices_sub).astype(np.int64)
    lr = np.sort(np.unique(subs[:, 0]))
    n = len(lr)
    if n < 2:
        return im, sm
    lo = max(int(matlab_round(n * 0.25)), 1) - 1
    hi = max(int(matlab_round(n * 0.75)), 1) - 1
    center_lr = lr[lo:hi + 1]
    mins = np.asarray([subs[subs[:, 0] == x, 1].min() for x in center_lr], dtype=np.int64)
    notch_ap = int(mins.max())
    cand = np.flatnonzero(mins == notch_ap)
    notch_lr = int(center_lr[cand[int(math.ceil(len(cand) / 2.0)) - 1]])

    # Raw right-knee geometric groups. The final left-knee switch follows MATLAB.
    side_masks = {
        'L': subs[:, 0] >= notch_lr,
        'M': subs[:, 0] < notch_lr,
    }
    raw: Dict[str, Dict[str, np.ndarray]] = {}
    for side in ('L', 'M'):
        side_mask = side_masks[side]
        side_subs = subs[side_mask]
        if len(side_subs) == 0:
            raw[side] = {k: np.empty((0, 3), dtype=np.int64) for k in ('a', 'p', 'e', 'c', 'i')}
            continue
        min_ap = int(side_subs[:, 1].min())
        max_ap = int(side_subs[:, 1].max())
        min_c = notch_ap - int(matlab_round((notch_ap - min_ap) * cc_percentage))
        cand_v = side_mask & (subs[:, 1] < notch_ap) & (subs[:, 1] > min_c)
        cand_faces = _faces_or_vertex_mask(scb, cand_v)
        comps = _face_components(cand_faces)
        pct = _component_percentages(comps)
        large_faces = _concat_faces(comps, pct >= 50)
        if len(large_faces):
            c_mask = _used_vertices_mask(scb, large_faces)
            central_subs = subs[c_mask].copy()
        else:
            # Normal cases have a large component; this fallback avoids crashing on
            # tiny synthetic/edge-case masks while preserving the defining predicate.
            central_subs = subs[cand_v].copy()

        partial_a: list[np.ndarray] = []
        partial_p: list[np.ndarray] = []
        partial_c: list[np.ndarray] = []
        small_faces = _concat_faces(comps, pct <= 50)
        for comp in _face_components(small_faces):
            used = np.unique(comp)
            vsub = subs[used]
            if len(vsub) == 0:
                continue
            center_si = int(matlab_round(np.mean(vsub[:, 2])))
            center_ap = int(matlab_round(np.mean(vsub[:, 1])))
            if len(central_subs):
                c_center_si = int(matlab_round(np.mean(central_subs[:, 2])))
                c_range_si = int(matlab_round(central_subs[:, 2].max() - central_subs[:, 2].min()))
            else:
                c_center_si, c_range_si = center_si, 0
            if (center_si - c_center_si) > c_range_si:
                partial_p.append(vsub)
            elif abs(center_ap - notch_ap) <= 2:
                partial_a.append(vsub)
            else:
                partial_c.append(vsub)
        if partial_c:
            central_subs = np.vstack([central_subs] + partial_c) if len(central_subs) else np.vstack(partial_c)

        a_subs = side_subs[(side_subs[:, 1] <= max_ap) & (side_subs[:, 1] >= notch_ap)]
        p_subs = side_subs[(side_subs[:, 1] <= min_c) & (side_subs[:, 1] >= min_ap)]
        if partial_a:
            a_subs = np.vstack([a_subs] + partial_a)
        if partial_p:
            p_subs = np.vstack([p_subs] + partial_p)

        ext, cen, inte = [], [], []
        if len(central_subs):
            for ap in np.unique(central_subs[:, 1]):
                q = central_subs[central_subs[:, 1] == ap]
                mx, mn = q[:, 0].max(), q[:, 0].min()
                f1 = mn + (mx - mn) / 3.0
                f2 = mn + 2.0 * (mx - mn) / 3.0
                if side == 'L':
                    me = (q[:, 0] <= mx) & (q[:, 0] > f2)
                    mc = (q[:, 0] <= f2) & (q[:, 0] >= f1)
                    mi = (q[:, 0] < f1) & (q[:, 0] >= mn)
                else:
                    mi = (q[:, 0] <= mx) & (q[:, 0] > f2)
                    mc = (q[:, 0] <= f2) & (q[:, 0] >= f1)
                    me = (q[:, 0] < f1) & (q[:, 0] >= mn)
                ext.append(q[me]); cen.append(q[mc]); inte.append(q[mi])
        stack = lambda xs: np.vstack([x for x in xs if len(x)]) if any(len(x) for x in xs) else np.empty((0, 3), dtype=np.int64)
        raw[side] = {'a': a_subs, 'p': p_subs, 'e': stack(ext), 'c': stack(cen), 'i': stack(inte)}

    if knee_side.lower().startswith('l'):
        raw['L'], raw['M'] = raw['M'], raw['L']
    region_subs = {
        1: raw['M']['a'], 2: raw['M']['e'], 3: raw['M']['c'], 4: raw['M']['i'], 5: raw['M']['p'],
        6: raw['L']['a'], 7: raw['L']['e'], 8: raw['L']['c'], 9: raw['L']['i'], 10: raw['L']['p'],
    }
    for r, rs in region_subs.items():
        sm[r] = _region_membership_from_subs(subs, rs)
        # MATLAB: vers_iC = vers_iC_FC(ismember(subs_iC_FC, subs_scB_region,'rows'),:)
        im[r] = _region_membership_from_subs(subs_i, rs)
    return im, sm


def _central_tc_surface_mask(mesh: Mesh, vox: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vers = mesh.vertices_sub.astype(np.float64) * vox[None, :]
    center, sing, dirs, trans, vct = _principal_transform(vers)
    i2 = int(np.flatnonzero(dirs == 1)[0])
    i1 = int(np.flatnonzero(dirs == 0)[0])
    ratio = math.sqrt(float(sing[i2] / sing[i1])) if sing[i1] != 0 else 1.0
    areas = _mesh_face_areas_numpy(mesh, vox)
    total_area = float(areas.sum())
    A = 5.0
    first_act: Optional[int] = None
    oscill = 0
    counter = 0
    while True:
        B = A * ratio
        idx = (vct[:, 0] ** 2 / (A * A) + vct[:, 1] ** 2 / (B * B) - 1.0) < 0
        region_faces = np.any(idx[mesh.faces], axis=1)
        area = float(areas[region_faces].sum())
        frac = area / total_area if total_area != 0 else 0.0
        if abs(frac - 0.2) <= 0.005 or oscill >= 1000:
            break
        act = 1 if frac > 0.2 else -1
        counter += 1
        A = A - 0.5 if act == 1 else A + 0.5
        if counter == 1:
            first_act = act
        if first_act is not None and act != first_act:
            # MATLAB's recorder_lastAct is deliberately never updated after action #1.
            oscill += 1
    return idx, center, trans, vct, vers


def _tc_four_masks(vct: np.ndarray, central: np.ndarray, center: np.ndarray,
                   trans: np.ndarray, other_center: np.ndarray) -> Tuple[list[np.ndarray], np.ndarray]:
    non_idx = np.flatnonzero(~central)
    v = vct[non_idx]
    vec = np.linalg.solve(trans, other_center - center)
    def slope(deg: float) -> float:
        a = math.radians(deg)
        R = np.array([[math.cos(a), -math.sin(a), 0.0],
                      [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        z = R @ vec
        return z[1] / z[0] if abs(z[0]) > 1e-15 else math.copysign(np.inf, z[1])
    k1, k2 = slope(45.0), slope(-45.0)
    local = [
        (v[:, 1] > k1 * v[:, 0]) & (v[:, 1] > k2 * v[:, 0]),
        (v[:, 1] < k1 * v[:, 0]) & (v[:, 1] < k2 * v[:, 0]),
        (v[:, 1] <= k1 * v[:, 0]) & (v[:, 1] >= k2 * v[:, 0]),
        (v[:, 1] >= k1 * v[:, 0]) & (v[:, 1] <= k2 * v[:, 0]),
    ]
    full = []
    centers = []
    for lm in local:
        fm = np.zeros(len(vct), dtype=bool)
        fm[non_idx[lm]] = True
        full.append(fm)
        pts = ((trans @ vct[fm].T).T + center) if np.any(fm) else np.empty((0, 3))
        centers.append(pts.mean(axis=0) if len(pts) else np.array([np.nan, np.nan, np.nan]))
    return full, np.asarray(centers)


def surface_parcellation_tc(inner_m: Mesh, inner_l: Mesh, scb_m: Mesh, scb_l: Mesh,
                             vox: np.ndarray, knee_side: str
                             ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
                                        Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Port of CM_cal_SurfaceParcellation_TC.m.

    Returns inner/scB masks separately for MTC and LTC, all keyed by global labels.
    """
    im = {r: np.zeros(len(inner_m.vertices_sub), bool) for r in range(11, 16)}
    sm = {r: np.zeros(len(scb_m.vertices_sub), bool) for r in range(11, 16)}
    il = {r: np.zeros(len(inner_l.vertices_sub), bool) for r in range(16, 21)}
    sl = {r: np.zeros(len(scb_l.vertices_sub), bool) for r in range(16, 21)}
    if inner_m.empty or inner_l.empty or scb_m.empty or scb_l.empty:
        return im, sm, il, sl

    cmask, cm, tm, vmct, vm = _central_tc_surface_mask(scb_m, vox)
    clmask, cl, tl, vlct, vl = _central_tc_surface_mask(scb_l, vox)
    mc, mcent = _tc_four_masks(vmct, cmask, cm, tm, cl)
    lc, lcent = _tc_four_masks(vlct, clmask, cl, tl, cm)
    ma = int(np.nanargmax(mcent[:, 1])); mp = int(np.nanargmin(mcent[:, 1]))
    mi0 = int(np.nanargmax(mcent[:, 0])); me0 = int(np.nanargmin(mcent[:, 0]))
    la = int(np.nanargmax(lcent[:, 1])); lp = int(np.nanargmin(lcent[:, 1]))
    le0 = int(np.nanargmax(lcent[:, 0])); li0 = int(np.nanargmin(lcent[:, 0]))

    m_masks = {11: mc[ma], 12: mc[me0], 13: mc[mp], 14: mc[mi0], 15: cmask}
    l_masks = {16: lc[la], 17: lc[le0], 18: lc[lp], 19: lc[li0], 20: clmask}
    if knee_side.lower().startswith('l'):
        m_masks[12], m_masks[14] = m_masks[14], m_masks[12]
        l_masks[17], l_masks[19] = l_masks[19], l_masks[17]

    subs_m = matlab_round(scb_m.vertices_sub).astype(np.int64)
    subs_l = matlab_round(scb_l.vertices_sub).astype(np.int64)
    subs_im = matlab_round(inner_m.vertices_sub).astype(np.int64)
    subs_il = matlab_round(inner_l.vertices_sub).astype(np.int64)
    for r, mask in m_masks.items():
        sm[r] = mask.copy()
        rs = subs_m[mask]
        im[r] = _region_membership_from_subs(subs_im, rs)
    for r, mask in l_masks.items():
        sl[r] = mask.copy()
        rs = subs_l[mask]
        il[r] = _region_membership_from_subs(subs_il, rs)
    return im, sm, il, sl

# -----------------------------------------------------------------------------
# CUDA primitives
# -----------------------------------------------------------------------------
def choose_device(require_cuda: bool, allow_cpu_fallback: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda and not allow_cpu_fallback:
        raise RuntimeError("CUDA GPU is required. Use --allow-cpu-fallback only for debugging/testing.")
    return torch.device("cpu")


def nearest_indices_gpu(src: torch.Tensor, dst: torch.Tensor, dst_chunk: int = 4096, src_chunk: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
    """For every dst point, return distance/index of nearest src point.

    This is the GPU equivalent of MATLAB pdist2(..., 'Smallest', 1), but chunks
    both operands so real knee surfaces do not require a multi-GB distance matrix.
    """
    if src.numel() == 0 or dst.numel() == 0:
        return (torch.empty((dst.shape[0],), device=dst.device, dtype=dst.dtype),
                torch.empty((dst.shape[0],), dtype=torch.long, device=dst.device))
    d_all, i_all = [], []
    for ds in range(0, dst.shape[0], dst_chunk):
        q = dst[ds:ds+dst_chunk]
        best_d = torch.full((len(q),), float("inf"), dtype=q.dtype, device=q.device)
        best_i = torch.zeros((len(q),), dtype=torch.long, device=q.device)
        for ss in range(0, src.shape[0], src_chunk):
            d = torch.cdist(q, src[ss:ss+src_chunk])
            v, i = torch.min(d, dim=1)
            take = v < best_d
            best_d = torch.where(take, v, best_d)
            best_i = torch.where(take, i + ss, best_i)
        d_all.append(best_d); i_all.append(best_i)
    return torch.cat(d_all), torch.cat(i_all)



def transfer_roi_masks_nn(source: Mesh, target: Mesh, source_masks: Dict[int, np.ndarray],
                          device: torch.device) -> Dict[int, np.ndarray]:
    """Transfer ROI membership from compatibility iC to native iC by nearest vertex.

    MATLAB's mesh pipeline preserves rounded voxel-row identity between mesh_iC
    and mesh_scB. skimage marching-cubes does not. v11 keeps the bone-rebased
    iC only as a correspondence chart: parcellate it as in v9, then transfer ROI
    membership to the original native cartilage iC without moving native metric
    geometry.
    """
    out = {int(r): np.zeros(len(target.vertices_sub), dtype=bool) for r in source_masks}
    if source.empty or target.empty:
        return out
    src = torch.as_tensor(source.vertices_sub, dtype=torch.float32, device=device)
    dst = torch.as_tensor(target.vertices_sub, dtype=torch.float32, device=device)
    _, idx = nearest_indices_gpu(src, dst)
    ii = idx.detach().cpu().numpy()
    for r, mask0 in source_masks.items():
        mask = np.asarray(mask0, dtype=bool)
        if len(mask) == len(source.vertices_sub):
            out[int(r)] = mask[ii]
    return out

def knn_indices_multi_gpu(points: torch.Tensor, ks: Sequence[int], chunk: int = 4096) -> Dict[int, torch.Tensor]:
    """Compute multiple self-KNN index sets from one distance-matrix pass.

    CartiMorph's normal estimation, orientation voting and normal smoothing all
    query nearest neighbours on the same inner surface.  The old port recomputed
    ``torch.cdist(points_chunk, points)`` three times.  This routine computes the
    distance matrix once per chunk and then calls ``topk`` with the *same k values*
    used by the original routines, preserving their neighbour-selection semantics
    (including the original topk tie behaviour for each k).
    """
    n = int(points.shape[0])
    if n == 0:
        return {int(k): torch.empty((0, 0), dtype=torch.long, device=points.device) for k in ks}

    norm_ks = sorted({min(max(int(k), 1), n) for k in ks})
    out = {k: torch.empty((n, k), dtype=torch.long, device=points.device) for k in norm_ks}
    for s in range(0, n, chunk):
        q = points[s:s + chunk]
        d = torch.cdist(q, points)
        for k in norm_ks:
            # Keep a separate topk(k=...) call for each requested k rather than
            # slicing topk(max_k), so tied distances behave like the old code.
            out[k][s:s + len(q)] = torch.topk(d, k=k, dim=1, largest=False).indices
    return out


def estimate_normals_gpu(points: torch.Tensor, k: int,
                         knn_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    n = points.shape[0]
    if n == 0:
        return torch.empty_like(points)
    k = min(max(int(k), 3), n)
    normals = torch.empty_like(points)
    chunk = 4096
    for s in range(0, n, chunk):
        q = points[s:s+chunk]
        if knn_idx is None:
            d = torch.cdist(q, points)
            idx = torch.topk(d, k=k, dim=1, largest=False).indices
        else:
            idx = knn_idx[s:s + len(q), :k]
        pool = points[idx]
        centered = pool - pool.mean(dim=1, keepdim=True)
        # right singular vector corresponding to smallest singular value.
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        v = vh[:, -1, :]
        normals[s:s+len(q)] = F.normalize(v, dim=1)
    return normals


def reorient_normals_gpu(normals: torch.Tensor, inner: torch.Tensor, outer: torch.Tensor,
                         vox: torch.Tensor, k_vote: int,
                         vote_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    if len(inner) == 0 or len(outer) == 0:
        return normals
    scale = torch.min(vox)
    ns = normals * scale
    ep = inner + ns
    em = inner - ns
    dp, _ = nearest_indices_gpu(outer, ep)
    dm, _ = nearest_indices_gpu(outer, em)
    n1 = torch.where((dp > dm)[:, None], -ns, ns)
    # MATLAB majority vote using n_neigh nearest oriented normals.
    n = len(inner); k = min(max(int(k_vote), 2), n)
    out = torch.empty_like(n1)
    chunk = 4096
    for s in range(0, n, chunk):
        q = inner[s:s+chunk]
        if vote_idx is None:
            d = torch.cdist(q, inner)
            idx = torch.topk(d, k=k, dim=1, largest=False).indices
        else:
            idx = vote_idx[s:s + len(q), :k]
        neigh = n1[idx]
        cur = n1[s:s+len(q)]
        dots = torch.sum(neigh * cur[:, None, :], dim=2)
        same = torch.sum(dots > 0, dim=1) - 1
        keep = same > math.floor((k - 1) / 2)
        out[s:s+len(q)] = torch.where(keep[:,None], cur, -cur)
    return F.normalize(out, dim=1)


def smooth_normals_gpu(normals: torch.Tensor, points: torch.Tensor, k: int,
                       knn_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    n = len(points); k = min(max(int(k), 1), n)
    out = torch.empty_like(normals)
    chunk = 4096
    for s in range(0, n, chunk):
        q = points[s:s+chunk]
        if knn_idx is None:
            d = torch.cdist(q, points)
            idx = torch.topk(d, k=k, dim=1, largest=False).indices
        else:
            idx = knn_idx[s:s + len(q), :k]
        out[s:s+len(q)] = F.normalize(normals[idx].mean(dim=1), dim=1)
    return out


def ray_mesh_nearest_gpu(origins: torch.Tensor, dirs: torch.Tensor,
                         tri: torch.Tensor, max_depth: float,
                         ray_chunk: int = 512, tri_chunk: int = 8192,
                         spatial_prune: bool = True) -> torch.Tensor:
    """Möller-Trumbore nearest hit with exact conservative finite-ray pruning.

    For cartilage thickness ``max_depth`` is finite (normally 7 mm).  Any valid
    hit must lie inside the AABB of that finite ray segment, so triangles whose
    expanded AABBs cannot overlap any ray in the current chunk are skipped before
    running the *unchanged* Möller-Trumbore arithmetic.  The triangle AABB is
    expanded by the exact barycentric-tolerance scale used by the inclusive
    TriangleRayIntersection test, making the filter conservative rather than an
    approximation.  Infinite-depth uncertainty rays use the original brute-force
    path.
    """
    nr = origins.shape[0]
    best = torch.full((nr,), float("inf"), device=origins.device, dtype=origins.dtype)
    # TriangleRayIntersection.m default epsilon; calls use border='inclusive'.
    eps = torch.as_tensor(1e-5, dtype=origins.dtype, device=origins.device)

    # Precompute triangle geometry once.  The old implementation recomputed these
    # edge differences for every ray chunk even though the mesh is constant.
    v0_all = tri[:, 0]
    e1_all = tri[:, 1] - v0_all
    e2_all = tri[:, 2] - v0_all

    finite_depth = math.isfinite(float(max_depth)) and float(max_depth) > 0.0
    do_prune = bool(spatial_prune and finite_depth and tri.shape[0] >= 1024)
    if do_prune:
        tri_min = torch.amin(tri, dim=1)
        tri_max = torch.amax(tri, dim=1)
        # Inclusive barycentric tolerance permits a point to lie up to roughly
        # eps*(|e1|+|e2|) outside the strict triangle AABB. Expand by that bound.
        tri_pad = eps * torch.sum(torch.abs(e1_all) + torch.abs(e2_all), dim=1)
        tri_min = tri_min - tri_pad[:, None]
        tri_max = tri_max + tri_pad[:, None]
        # Smaller ray groups make the union-AABB candidate set much tighter while
        # keeping the expensive Möller-Trumbore kernel well batched.
        work_ray_chunk = min(int(ray_chunk), 128)
    else:
        tri_min = tri_max = None
        work_ray_chunk = int(ray_chunk)

    for rs in range(0, nr, work_ray_chunk):
        o = origins[rs:rs+work_ray_chunk]
        d = dirs[rs:rs+work_ray_chunk]
        local = torch.full((len(o),), float("inf"), device=o.device, dtype=o.dtype)

        if do_prune:
            # Include the tiny negative-t interval accepted by the MATLAB
            # inclusive test.  max_depth itself is also included conservatively;
            # the final output still uses the original strict < max_depth rule.
            p0 = o - d * eps
            p1 = o + d * float(max_depth)
            ray_min = torch.minimum(p0, p1)
            ray_max = torch.maximum(p0, p1)
            # First intersect the triangle AABBs with the union AABB of this ray
            # chunk.  Marching-cubes vertex order is spatially coherent, so this
            # inexpensive O(T) coarse pass usually removes most triangles before
            # forming the per-ray overlap matrix.  It is still fully conservative.
            chunk_min = torch.amin(ray_min, dim=0)
            chunk_max = torch.amax(ray_max, dim=0)
            coarse = (
                (tri_max[:, 0] >= chunk_min[0]) & (tri_min[:, 0] <= chunk_max[0]) &
                (tri_max[:, 1] >= chunk_min[1]) & (tri_min[:, 1] <= chunk_max[1]) &
                (tri_max[:, 2] >= chunk_min[2]) & (tri_min[:, 2] <= chunk_max[2])
            )
            coarse_idx = torch.nonzero(coarse, as_tuple=False).flatten()
            if coarse_idx.numel() == 0:
                best[rs:rs+len(o)] = local
                continue
            cmin = tri_min[coarse_idx]
            cmax = tri_max[coarse_idx]
            overlap = (
                (ray_max[:, None, 0] >= cmin[None, :, 0]) &
                (ray_min[:, None, 0] <= cmax[None, :, 0]) &
                (ray_max[:, None, 1] >= cmin[None, :, 1]) &
                (ray_min[:, None, 1] <= cmax[None, :, 1]) &
                (ray_max[:, None, 2] >= cmin[None, :, 2]) &
                (ray_min[:, None, 2] <= cmax[None, :, 2])
            )
            keep_local = torch.nonzero(torch.any(overlap, dim=0), as_tuple=False).flatten()
            cand_idx = coarse_idx[keep_local]
            if cand_idx.numel() == 0:
                best[rs:rs+len(o)] = local
                continue
        else:
            cand_idx = None

        ntri = int(cand_idx.numel()) if cand_idx is not None else int(tri.shape[0])
        for ts in range(0, ntri, tri_chunk):
            if cand_idx is None:
                sl = slice(ts, min(ts + tri_chunk, ntri))
                v0 = v0_all[sl]; e1 = e1_all[sl]; e2 = e2_all[sl]
            else:
                ii = cand_idx[ts:ts + tri_chunk]
                v0 = v0_all[ii]; e1 = e1_all[ii]; e2 = e2_all[ii]
            nt = len(v0)
            # Keep the original Möller-Trumbore operation order unchanged.
            pvec = torch.cross(d[:,None,:].expand(-1,nt,-1), e2[None,:,:].expand(len(o),-1,-1), dim=2)
            det = torch.sum(e1[None,:,:] * pvec, dim=2)
            valid = torch.abs(det) > eps
            inv_det = torch.where(valid, 1.0/det, torch.zeros_like(det))
            tvec = o[:,None,:] - v0[None,:,:]
            u = torch.sum(tvec*pvec, dim=2) * inv_det
            qvec = torch.cross(tvec, e1[None,:,:].expand(len(o),-1,-1), dim=2)
            v = torch.sum(d[:,None,:]*qvec, dim=2) * inv_det
            tt = torch.sum(e2[None,:,:]*qvec, dim=2) * inv_det
            hit = valid & (u >= -eps) & (v >= -eps) & ((u+v) <= 1+eps) & (tt >= -eps)
            cand = torch.where(hit, torch.abs(tt), torch.full_like(tt, float("inf")))
            local = torch.minimum(local, cand.min(dim=1).values)
        best[rs:rs+len(o)] = local
    best = torch.where(best < float(max_depth), best, torch.zeros_like(best))
    return best


def approximate_zeros_gpu(values: torch.Tensor, points: torch.Tensor, k: int) -> torch.Tensor:
    out = values.clone()
    nz = values != 0
    z = ~nz
    if not torch.any(z) or not torch.any(nz):
        return out
    src_p = points[nz]; src_v = values[nz]
    q = points[z]
    k = min(max(int(k), 1), len(src_p))
    vals = []
    for s in range(0, len(q), 4096):
        d = torch.cdist(q[s:s+4096], src_p)
        idx = torch.topk(d, k=k, dim=1, largest=False).indices
        vals.append(src_v[idx].mean(dim=1))
    out[z] = torch.cat(vals)
    return out


def _matlab_ellipsoid_triangles(vox: torch.Tensor) -> torch.Tensor:
    """Triangular mesh produced by create_ellipsoid_mesh/ellipsoidMesh in MATLAB.

    matGeom defaults: nPhi=32, nTheta=16; yPeriodic=true; each quad is split as
    [1,2,3] and [1,3,4].  In ellipsoidMesh, a/b/c are half-axis lengths and
    CartiMorph passes size_voxel directly, hence radii == voxel size.
    """
    dev, dt = vox.device, vox.dtype
    n_phi, n_theta = 32, 16
    theta = torch.linspace(0.0, math.pi, n_theta + 1, device=dev, dtype=dt)
    # Drop the duplicated 2*pi row exactly as surfToMesh(...,'yPeriodic',true).
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, device=dev, dtype=dt)[:-1]
    st = torch.sin(theta)
    x = torch.cos(phi)[:, None] * st[None, :] * vox[0]
    y = torch.sin(phi)[:, None] * st[None, :] * vox[1]
    z = torch.ones((n_phi, 1), device=dev, dtype=dt) * torch.cos(theta)[None, :] * vox[2]
    # MATLAB x(:) is column-major: transpose first, then flatten in row-major.
    vertices = torch.stack([x.T.reshape(-1), y.T.reshape(-1), z.T.reshape(-1)], dim=1)
    inds = torch.arange(n_phi * (n_theta + 1), device=dev, dtype=torch.long).reshape(n_theta + 1, n_phi).T
    ie = torch.cat([inds, inds[:1]], dim=0)  # periodic phi closure
    v1 = ie[:-1, :-1].T.reshape(-1)
    v2 = ie[:-1, 1:].T.reshape(-1)
    v3 = ie[1:, 1:].T.reshape(-1)
    v4 = ie[1:, :-1].T.reshape(-1)
    q1 = torch.stack([v1, v2, v3], dim=1)
    q2 = torch.stack([v1, v3, v4], dim=1)
    return vertices[torch.cat([q1, q2], dim=0)]


def ellipsoid_uncertainty_gpu(normals: torch.Tensor, vox: torch.Tensor) -> torch.Tensor:
    """GPU port of cal_uncertainty(...,'ellipsoid') + uncertaintyEstimation.

    It uses the same matGeom ellipsoid tessellation as MATLAB and the same
    TriangleRayIntersection inclusive-border rule instead of a closed-form shortcut.
    """
    if normals.numel() == 0:
        return torch.empty((0,), dtype=normals.dtype, device=normals.device)
    vox = vox.to(dtype=normals.dtype, device=normals.device)
    tri = _matlab_ellipsoid_triangles(vox)
    origins = torch.zeros_like(normals)
    # Every point of the triangulated ellipsoid lies inside the convex hull of
    # vertices whose Euclidean norm is <= ||vox||.  Using 2*||vox|| as a finite
    # ray bound is therefore guaranteed not to clip a real center-to-surface hit,
    # while allowing the exact AABB pruning path to skip most ellipsoid faces.
    safe_depth = float((2.0 * torch.linalg.vector_norm(vox)).item())
    return ray_mesh_nearest_gpu(origins, normals, tri, safe_depth, ray_chunk=512, tri_chunk=2048)


def triangle_area_gpu(vertices_phys: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.numel() == 0:
        return torch.empty((0,), device=vertices_phys.device, dtype=vertices_phys.dtype)
    p = vertices_phys[faces]
    return 0.5 * torch.linalg.vector_norm(torch.cross(p[:,1]-p[:,0], p[:,2]-p[:,0], dim=1), dim=1)


# -----------------------------------------------------------------------------
# Registration handling
# -----------------------------------------------------------------------------
def warp_atlas_with_flow(template_atlas: np.ndarray, flow: np.ndarray, device: torch.device) -> np.ndarray:
    """Warp nearest-neighbour label atlas with VoxelMorph-style voxel displacement.

    Expected flow shape [X,Y,Z,3], with components in array-axis order.  This is
    the same pull convention used by VoxelMorph SpatialTransformer: sample source
    at regular_grid + flow.
    """
    if flow.ndim != 4 or flow.shape[-1] != 3:
        raise ValueError("Dense deformation field must have shape [X,Y,Z,3]")
    x, y, z = flow.shape[:3]
    src = torch.as_tensor(np.array(template_atlas, copy=True), dtype=torch.float32, device=device)[None,None]
    fl = torch.as_tensor(np.array(flow, copy=True), dtype=torch.float32, device=device)
    gx, gy, gz = torch.meshgrid(
        torch.arange(x, device=device), torch.arange(y, device=device), torch.arange(z, device=device), indexing="ij"
    )
    loc = torch.stack([gx,gy,gz], dim=-1).float() + fl
    # grid_sample expects coordinates ordered x(last spatial), y, z(first spatial).
    grid = torch.empty((1,x,y,z,3), dtype=torch.float32, device=device)
    grid[0,...,0] = 2.0 * loc[...,2] / max(z-1,1) - 1.0
    grid[0,...,1] = 2.0 * loc[...,1] / max(y-1,1) - 1.0
    grid[0,...,2] = 2.0 * loc[...,0] / max(x-1,1) - 1.0
    out = F.grid_sample(src, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    return out[0,0].round().to(torch.uint8).cpu().numpy()


def atlas20_to_tissue(atlas: np.ndarray, labels: LabelConfig) -> np.ndarray:
    """Convert a 20-ROI cartilage atlas to the tissue-label prior used by MATLAB.

    The Toolbox registration branch ultimately supplies one warped cartilage mask
    at a time to cal_splitBoundary3D_wBone.  A pre-warped 20-region atlas can still
    be accepted as input, but it is collapsed back to FC/mTC/lTC before morphology.
    """
    a = np.asarray(atlas)
    out = np.full(a.shape, labels.background, dtype=np.int16)
    out[(a >= 1) & (a <= 10)] = labels.femoral_cartilage
    out[(a >= 11) & (a <= 15)] = labels.medial_tibial_cartilage
    out[(a >= 16) & (a <= 20)] = labels.lateral_tibial_cartilage
    return out


def resolve_warped_tissue(reg: np.ndarray, labels: LabelConfig, device: torch.device,
                          template_seg: Optional[np.ndarray]) -> np.ndarray:
    """Resolve registration input to the warped template tissue segmentation.

    This matches the Toolbox morphology call chain: registration provides a warped
    template *tissue/cartilage* prior for scB/FCL reconstruction.  The final 20 ROI
    labels are generated later from reconstructed surfaces by SurfaceParcellation.
    A 20-ROI atlas is accepted for compatibility and collapsed to tissue labels.
    """
    if reg.ndim == 4 and reg.shape[-1] == 3:
        if template_seg is None:
            raise ValueError(
                "Registration is a deformation field; provide --template-seg "
                "(a template tissue segmentation; a 20-ROI atlas is also accepted)."
            )
        warped = warp_atlas_with_flow(template_seg, reg, device)
    elif reg.ndim == 3:
        warped = np.asarray(reg)
    else:
        raise ValueError("Registration result must be a 3D label map or [X,Y,Z,3] flow")

    if np.issubdtype(np.asarray(warped).dtype, np.integer):
        # Exact fast path for the normal warped-labelmap case.  np.unique over an
        # OAI-sized 3-D array was ~1 s of pure CPU work; the old decision only
        # needs max(label) and whether any label lies in 1..20.
        maxv = int(np.max(warped)) if warped.size else 0
        if maxv >= 20 and np.any((warped >= 1) & (warped <= 20)):
            return atlas20_to_tissue(warped, labels)
        return np.asarray(warped, dtype=np.int16)
    vals = np.unique(warped[np.isfinite(warped)]).astype(int)
    maxv = int(vals.max()) if len(vals) else 0
    if maxv >= 20 and np.any(np.isin(vals, np.arange(1, 21))):
        return atlas20_to_tissue(warped, labels)
    return np.asarray(matlab_round(warped), dtype=np.int16)


# -----------------------------------------------------------------------------
# Morphology pipeline
# -----------------------------------------------------------------------------
def label_points_from_atlas_gpu(points_sub: np.ndarray, atlas: np.ndarray, device: torch.device) -> np.ndarray:
    """Direct sample atlas; for zeros, fill from nearest nonzero atlas voxel on GPU."""
    if len(points_sub) == 0:
        return np.empty((0,), np.uint8)
    ss = matlab_round(points_sub).astype(int)
    ii = _subs1_to_idx0(ss)
    ok = _valid_subs(ss, atlas.shape)
    lab = np.zeros(len(ss), np.uint8)
    lab[ok] = atlas[tuple(ii[ok].T)]
    need = lab == 0
    src0 = np.argwhere(atlas > 0)
    if np.any(need) and len(src0):
        src = torch.as_tensor(src0 + 1, dtype=torch.float32, device=device)
        dst = torch.as_tensor(ss[need], dtype=torch.float32, device=device)
        _, ni = nearest_indices_gpu(src, dst)
        src_np = src0[ni.cpu().numpy()]
        lab[need] = atlas[tuple(src_np.T)]
    return lab


def _surface_boundary_vertex_ids(faces: np.ndarray) -> np.ndarray:
    edges = _surface_boundary_edges(faces)
    return np.unique(edges) if len(edges) else np.empty((0,), dtype=np.int64)


def _coarse_cartilage_contact_faces(cart: np.ndarray, bone: np.ndarray, whole: Mesh):
    """Classify shared cartilage-mesh faces by the CartiMorph grown-bone mask.

    ``boundary_mesh`` rounds marching-cubes vertices to MATLAB-style integer
    subscripts.  A cartilage/bone interface lies at a half voxel and can therefore
    round onto the bone-side voxel, while Eq. (11)'s inner *cartilage* voxel set is
    one index away.  Exact row matching then loses most interface vertices.

    Sampling the already-computed ``bone_grown`` mask at the same rounded shared
    mesh vertices avoids that half-voxel mismatch.  Faces whose three vertices are
    on grown bone are coarse inner faces; faces whose three vertices are off grown
    bone are coarse outer faces; mixed faces are the transition band left for
    restricted surface dilation.
    """
    bg, origin, sl = _cartilage_bone_grown_local(cart, bone)
    if sl is None or whole.empty:
        z = np.empty((0, 3), dtype=np.int64)
        return z, z.copy(), z.copy(), np.zeros(len(whole.vertices_sub), dtype=bool)

    ss1 = matlab_round(whole.vertices_sub).astype(np.int64)
    idx0 = _subs1_to_idx0(ss1)
    loc = idx0 - origin[None, :]
    valid = np.all((loc >= 0) & (loc < np.asarray(bg.shape)[None, :]), axis=1)
    on_bone = np.zeros(len(ss1), dtype=bool)
    if np.any(valid):
        on_bone[valid] = bg[tuple(loc[valid].T)]

    fc = on_bone[whole.faces]
    fi = _unique_faces(whole.faces[np.all(fc, axis=1)])
    fo = _unique_faces(whole.faces[np.all(~fc, axis=1)])
    trans = _unique_faces(whole.faces[~(np.all(fc, axis=1) | np.all(~fc, axis=1))])
    return fi, fo, trans, on_bone


def split_cartilage_mesh(cart: np.ndarray, bone: np.ndarray, whole: Mesh,
                         finetune: bool = False, closing_iterations: int = 4,
                         debug_info: Optional[dict] = None,
                         vox: Optional[np.ndarray] = None,
                         contact_augment_inner: bool = False) -> Tuple[Mesh, Mesh]:
    """Split the whole cartilage mesh into inner/outer surfaces.

    ``finetune=False`` is the bit-for-bit pre-v6 compatibility path.

    ``finetune=True`` is the v7 CartiMorph candidate.  It follows the public
    MATLAB face-extraction semantics directly: Eq. (11)/(12) boundary voxels are
    mapped to the *shared whole-cartilage mesh* with OR face extraction (a face is
    selected when at least one of its vertices belongs to the target set), then
    the inner patch is surface-closed and the outer patch is grown by restricted
    surface dilation in ``whole - fine_inner``.

    v6's grown-bone contact classifier is retained only as a diagnostic because
    the real OAIZIB case showed that it destroys tibial inner surfaces
    (mTC/lTC), while the direct Eq.11 OR areas are anatomically plausible.
    """
    inter_subs, outer_subs = init_boundary_split_w_bone(cart, bone, need_outer=True)

    if not finetune:
        fi = extract_faces_and(whole, inter_subs)
        # Fallback if coarse split is sparse: mark vertices adjacent to dilated bone.
        if len(fi) < 3:
            bd = ndimage.binary_dilation(bone, structure=_sphere1())
            ss = whole.vertices_sub.astype(int)
            ii = _subs1_to_idx0(ss)
            vin = bd[tuple(ii.T)]
            fi = whole.faces[np.all(vin[whole.faces], axis=1)]
        used_i = np.zeros(len(whole.vertices_sub), bool)
        if len(fi):
            used_i[np.unique(fi)] = True
        fo = whole.faces[~np.all(used_i[whole.faces], axis=1)]
        if debug_info is not None:
            debug_info.update({
                "surface_split_mode": "legacy_and_complement",
                "coarse_inner_faces": int(len(fi)),
                "coarse_outer_faces": int(len(fo)),
            })
        return submesh(whole, fi), submesh(whole, fo)

    # Public CM_cal_extractFaces_OR semantics: >=1 target vertex selects a face.
    fi0 = _unique_faces(extract_faces_or(whole, inter_subs))
    fo0 = _unique_faces(extract_faces_or(whole, outer_subs))
    fi_eq11 = fi0.copy()

    # v6 contact classifier is diagnostics only in the v7 default path because it
    # failed badly on tibial cartilage.  v16 introduces a *femur-only* compatibility
    # augmentation: keep every official Eq.11 OR face, then add bone-facing contact
    # faces that skimage half-voxel/rounding can miss during exact row matching.
    # The caller enables this only for FC; mTC/lTC remain bit-for-bit v7.
    contact_i, contact_o, ftransition, _ = _coarse_cartilage_contact_faces(cart, bone, whole)
    contact_added = np.empty((0, 3), dtype=np.int64)
    if contact_augment_inner and len(contact_i):
        eq_keys = _row_keys_int(np.sort(fi0, axis=1)) if len(fi0) else np.empty((0,), dtype=np.dtype((np.void, 24)))
        ci_keys = _row_keys_int(np.sort(contact_i, axis=1))
        contact_added = contact_i[~np.isin(ci_keys, eq_keys)] if len(eq_keys) else contact_i.copy()
        fi0 = _unique_faces(np.vstack([fi0, contact_i])) if len(fi0) else _unique_faces(contact_i)

    fallback = "none"
    if len(fi0) < 3:
        # Degenerate edge-case fallback only.  Preserve the older robust path,
        # but never prefer it when Eq.11 OR yields a real patch.
        fi0 = _unique_faces(extract_faces_and(whole, inter_subs))
        fallback = "legacy_and"
    if len(fi0) < 3:
        bd = ndimage.binary_dilation(bone, structure=_sphere1())
        ss = matlab_round(whole.vertices_sub).astype(np.int64)
        ok = _valid_subs(ss, cart.shape)
        vin = np.zeros(len(ss), dtype=bool)
        ii = _subs1_to_idx0(ss[ok])
        vin[ok] = bd[tuple(ii.T)]
        fi0 = _unique_faces(whole.faces[np.all(vin[whole.faces], axis=1)])
        fallback = "dilated_bone_and"

    it = max(int(closing_iterations), 0)
    fi_fine = _surface_closing(fi0, whole.faces, it, it) if it else _unique_faces(fi0)

    # Outer fine-tuning: start from Eq.12 OR faces, remove any face already owned
    # by fine inner, then grow only inside the remaining whole-cartilage surface.
    outer_source = _remove_faces(fi_fine, whole.faces)
    fo_seed = _remove_faces(fi_fine, fo0)
    if len(fo_seed) < 3:
        # If Eq.12 is degenerate, use all non-inner faces as seed/source.  This is
        # only a fallback; normal cases stay on the Eq.12 path.
        fo_seed = outer_source
        fallback = fallback + "+outer_source" if fallback != "none" else "outer_source"

    border_ids = _surface_boundary_vertex_ids(fi_fine)
    border_vertices = (whole.vertices_sub[border_ids] if len(border_ids)
                       else np.empty((0, 3), dtype=np.float64))
    fo_fine = _surface_dilation_restricted(
        fo_seed, outer_source, whole.vertices_sub, border_vertices
    )
    if len(fi_fine) and len(fo_fine):
        fo_fine = _remove_faces(fi_fine, fo_fine)

    if debug_info is not None:
        debug_info.update({
            "surface_split_mode": ("v16_fc_eq11or_plus_contact_closing_restricted_outer"
                                   if contact_augment_inner else "v7_eq11or_closing_restricted_outer"),
            "surface_split_fallback": fallback,
            "surface_closing_iterations_dilation": int(it),
            "surface_closing_iterations_erosion": int(it),
            "eq11_or_inner_faces": int(len(fi_eq11)),
            "contact_augmented_inner_faces_preclosing": int(len(fi0)),
            "contact_added_inner_faces_preclosing": int(len(contact_added)),
            "eq12_or_outer_faces": int(len(fo0)),
            "v6_contact_inner_faces_diagnostic": int(len(contact_i)),
            "v6_contact_outer_faces_diagnostic": int(len(contact_o)),
            "v6_transition_faces_diagnostic": int(len(ftransition)),
            "fine_inner_faces": int(len(fi_fine)),
            "fine_outer_faces": int(len(fo_fine)),
            "fine_inner_boundary_vertices": int(len(border_ids)),
            "whole_cartilage_faces": int(len(whole.faces)),
        })
        if vox is not None:
            vv = np.asarray(vox, dtype=np.float64)
            debug_info.update({
                "whole_cartilage_area_mm2": _faces_area_numpy(whole, whole.faces, vv),
                "eq11_or_inner_area_mm2": _faces_area_numpy(whole, fi_eq11, vv),
                "contact_augmented_inner_area_mm2_preclosing": _faces_area_numpy(whole, fi0, vv),
                "contact_added_inner_area_mm2_preclosing": _faces_area_numpy(whole, contact_added, vv),
                "eq12_or_outer_area_mm2": _faces_area_numpy(whole, fo0, vv),
                "v6_contact_inner_area_mm2_diagnostic": _faces_area_numpy(whole, contact_i, vv),
                "v6_contact_outer_area_mm2_diagnostic": _faces_area_numpy(whole, contact_o, vv),
                "v6_transition_area_mm2_diagnostic": _faces_area_numpy(whole, ftransition, vv),
                "fine_inner_area_mm2": _faces_area_numpy(whole, fi_fine, vv),
                "fine_outer_area_mm2": _faces_area_numpy(whole, fo_fine, vv),
            })

    return submesh(whole, _unique_faces(fi_fine)), submesh(whole, _unique_faces(fo_fine))

def _largest_component_26(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m.copy()
    sl, _ = _foreground_bbox(m, pad=0)
    q = m[sl]
    lab, n = ndimage.label(q, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return m.copy()
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = lab == int(np.argmax(counts))
    out = np.zeros(m.shape, dtype=bool)
    out[sl] = keep
    return out


def _unique_faces(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.int64)
    f = np.asarray(faces, dtype=np.int64)
    key = np.sort(f, axis=1)
    _, idx = np.unique(key, axis=0, return_index=True)
    return f[np.sort(idx)]


def _row_keys_int(a: np.ndarray) -> np.ndarray:
    """Hash-free exact row keys for vectorized membership tests."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
    if a.ndim != 2:
        raise ValueError(f"row-key input must be 2-D, got {a.shape}")
    if len(a) == 0:
        return np.empty((0,), dtype=np.dtype((np.void, max(a.shape[1], 1) * 8)))
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def _remove_faces(remove: np.ndarray, source: np.ndarray) -> np.ndarray:
    if len(source) == 0 or len(remove) == 0:
        return np.asarray(source, dtype=np.int64).copy()
    rem = np.sort(np.asarray(remove, dtype=np.int64), axis=1)
    src = np.asarray(source, dtype=np.int64)
    sk = _row_keys_int(np.sort(src, axis=1))
    rk = _row_keys_int(rem)
    return src[~np.isin(sk, rk)]


def _face_edges(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return np.empty((0, 2), dtype=np.int64)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return np.sort(e, axis=1)


def _surface_boundary_edges(faces: np.ndarray) -> np.ndarray:
    e = _face_edges(faces)
    if len(e) == 0:
        return e
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


def _face_edge_keys_per_face(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        z = np.empty((0,), dtype=np.dtype((np.void, 16)))
        return z, z, z
    e01 = _row_keys_int(np.sort(f[:, [0, 1]], axis=1))
    e12 = _row_keys_int(np.sort(f[:, [1, 2]], axis=1))
    e20 = _row_keys_int(np.sort(f[:, [2, 0]], axis=1))
    return e01, e12, e20


def _surface_dilation(faces_in: np.ndarray, faces_source: np.ndarray, iteration: int) -> np.ndarray:
    """Face-index equivalent of MATLAB cal_surfaceDilation, vectorized.

    The iteration semantics are unchanged; only the old Python per-face edge loop
    is replaced by exact row-membership operations on integer edge IDs.
    """
    base = _unique_faces(faces_in)
    if len(base) == 0 or iteration <= 0:
        return base
    src = np.asarray(faces_source, dtype=np.int64)
    src_e0, src_e1, src_e2 = _face_edge_keys_per_face(src)
    base_keys = _row_keys_int(np.sort(base, axis=1))
    added = np.empty((0, 3), dtype=np.int64)
    frontier = base
    for _ in range(int(iteration)):
        margin_arr = _surface_boundary_edges(frontier)
        if len(margin_arr) == 0:
            break
        margin = _row_keys_int(margin_arr)
        hit = np.isin(src_e0, margin) | np.isin(src_e1, margin) | np.isin(src_e2, margin)
        if not np.any(hit):
            break
        cand = src[hit]
        cand_keys = _row_keys_int(np.sort(cand, axis=1))
        keep = ~np.isin(cand_keys, base_keys)
        if len(added):
            added_keys = _row_keys_int(np.sort(added, axis=1))
            keep &= ~np.isin(cand_keys, added_keys)
        nxt = cand[keep]
        if len(nxt) == 0:
            break
        added = _unique_faces(np.vstack([added, nxt]))
        frontier = added
    return _unique_faces(np.vstack([base, added])) if len(added) else base


def _surface_erosion(faces_in: np.ndarray, iteration: int) -> np.ndarray:
    """Face-index equivalent of MATLAB cal_surfaceErosion, vectorized."""
    cur = _unique_faces(faces_in)
    for _ in range(int(iteration)):
        if len(cur) == 0:
            break
        margin_arr = _surface_boundary_edges(cur)
        if len(margin_arr) == 0:
            break
        margin = _row_keys_int(margin_arr)
        e0, e1, e2 = _face_edge_keys_per_face(cur)
        rm = np.isin(e0, margin) | np.isin(e1, margin) | np.isin(e2, margin)
        if not np.any(rm):
            break
        cur = cur[~rm]
    return cur


def _surface_closing(faces_in: np.ndarray, faces_source: np.ndarray,
                     iteration_dilation: int, iteration_erosion: int) -> np.ndarray:
    return _surface_erosion(
        _surface_dilation(faces_in, faces_source, iteration_dilation),
        iteration_erosion,
    )


def _surface_dilation_restricted(faces_in: np.ndarray, faces_source: np.ndarray,
                                 vertices_source: np.ndarray, vertices_border: np.ndarray) -> np.ndarray:
    """Port of CM_cal_surfaceDilation_restricted.m.

    The routine repeatedly adds source faces adjacent to the boundary of the
    accumulated added patch, but excludes any candidate face containing a vertex
    listed in ``vertices_border``. Iteration stops only when no further face can
    be added. Faces remain indexed into the source vertex table throughout.
    """
    base = _unique_faces(faces_in)
    src = np.asarray(faces_source, dtype=np.int64)
    if len(base) == 0 or len(src) == 0:
        return base

    vsrc = np.asarray(vertices_source)
    vb = np.asarray(vertices_border)
    if len(vb):
        border_mask = _row_membership(matlab_round(vsrc).astype(np.int64),
                                      matlab_round(vb).astype(np.int64))
        border_ids = np.flatnonzero(border_mask)
    else:
        border_ids = np.empty((0,), dtype=np.int64)

    src_e0, src_e1, src_e2 = _face_edge_keys_per_face(src)
    base_keys = _row_keys_int(np.sort(base, axis=1))

    def candidates_from_margin(margin_faces: np.ndarray, added: np.ndarray) -> np.ndarray:
        margin_arr = _surface_boundary_edges(margin_faces)
        if len(margin_arr) == 0:
            return np.empty((0, 3), dtype=np.int64)
        margin = _row_keys_int(margin_arr)
        hit = np.isin(src_e0, margin) | np.isin(src_e1, margin) | np.isin(src_e2, margin)
        cand = src[hit]
        if len(cand) == 0:
            return cand
        ckeys = _row_keys_int(np.sort(cand, axis=1))
        keep = ~np.isin(ckeys, base_keys)
        if len(added):
            akeys = _row_keys_int(np.sort(added, axis=1))
            keep &= ~np.isin(ckeys, akeys)
        if len(border_ids):
            keep &= ~np.any(np.isin(cand, border_ids), axis=1)
        return _unique_faces(cand[keep])

    added = candidates_from_margin(base, np.empty((0, 3), dtype=np.int64))
    if len(added) == 0:
        return base

    while True:
        nxt = candidates_from_margin(added, added)
        if len(nxt) == 0:
            break
        added = _unique_faces(np.vstack([added, nxt]))

    return _unique_faces(np.vstack([base, added]))


def _delete_sharp_edge_tri(faces: np.ndarray) -> np.ndarray:
    """Port of cal_deleteSharpEdgeTri's repeated marginal-triangle removal."""
    out = _unique_faces(faces)
    while len(out):
        margin = _surface_boundary_edges(out)
        if len(margin) == 0:
            break
        mv = np.unique(margin)
        rm = np.all(np.isin(out, mv), axis=1)
        if not np.any(rm):
            break
        out = out[~rm]
    return out


def _fill_closed_surface_holes(faces_seed: np.ndarray, faces_bone: np.ndarray) -> np.ndarray:
    """Port of cal_reconCartDefect_conn on faces sharing the bone vertex table."""
    non = _remove_faces(faces_seed, faces_bone)
    comps = _face_components(non)
    pct = _component_percentages(comps)
    # cal_deleteLargeComponents(...,50): retain components <= 50%.
    filling = _concat_faces(comps, pct <= 50)
    if len(filling):
        return _unique_faces(np.vstack([faces_seed, filling]))
    return _unique_faces(faces_seed)


def _best_polyfit_matlab(x: np.ndarray, y: np.ndarray, max_order: int) -> np.ndarray:
    """Port cal_bestCurveFit_poly: orders 3..maxOrder, minimum fit() RMSE."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    best_coef = np.array([0.0])
    best_rmse = float("inf")
    for order in range(3, int(max_order) + 1):
        if len(x) <= order:
            continue
        with np.errstate(all="ignore"):
            try:
                coef = np.polyfit(x, y, order)
                resid = y - np.polyval(coef, x)
                dfe = len(x) - (order + 1)
                rmse = math.sqrt(float(np.sum(resid * resid)) / dfe) if dfe > 0 else float("inf")
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse, best_coef = rmse, coef
    if best_rmse == float("inf"):
        # MATLAB inputs normally support poly3; fallback only protects degenerate masks.
        deg = min(1, max(len(x) - 1, 0))
        return np.polyfit(x, y, deg) if len(x) else np.array([0.0])
    return best_coef


def _nearest_dist_numpy(src: np.ndarray, dst: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """For each dst row return min Euclidean distance to src (pdist2 Smallest=1)."""
    if len(src) == 0 or len(dst) == 0:
        return np.full(len(dst), np.inf, dtype=np.float64)
    out = np.empty(len(dst), dtype=np.float64)
    src = np.asarray(src, dtype=np.float64)
    for i in range(0, len(dst), chunk):
        q = np.asarray(dst[i:i + chunk], dtype=np.float64)
        d2 = np.sum((q[:, None, :] - src[None, :, :]) ** 2, axis=2)
        out[i:i + len(q)] = np.sqrt(np.min(d2, axis=1))
    return out


def _slice_needs_curve_fill(subs_slice: np.ndarray, size_img: Sequence[int]) -> bool:
    """Exact single-slice equivalent of the old 3-D connected-component test.

    ``recon_cart_defect_curve_*`` calls this routine with points from one fixed
    first-axis slice.  On such a plane, MATLAB/Python 26-connectivity in the full
    3-D volume is exactly 8-connectivity in the remaining two dimensions.  The
    previous implementation nevertheless allocated the complete knee volume and
    ran *two* 3-D labels for every slice, which dominates wall time on OAI-sized
    images.

    Removing components with <10 voxels cannot split or merge any surviving
    component, so the second label is also exactly equivalent to simply counting
    first-pass components whose size is >=10.
    """
    if len(subs_slice) == 0:
        return False
    ss = np.asarray(subs_slice, dtype=np.int64)
    ok = _valid_subs(ss, size_img)
    ss = ss[ok]
    if len(ss) == 0:
        return False
    # All callers group by the first subscript.  If this helper is ever used with
    # mixed slices, retain the legacy semantics rather than silently changing it.
    if np.any(ss[:, 0] != ss[0, 0]):
        vol = np.zeros(tuple(int(x) for x in size_img), dtype=bool)
        ii = _subs1_to_idx0(ss); vol[tuple(ii.T)] = True
        lab, n = ndimage.label(vol, structure=np.ones((3, 3, 3), dtype=bool))
        if not n:
            return False
        cnt = np.bincount(lab.ravel())
        return int(np.count_nonzero(cnt[1:] >= 10)) > 1

    yz = ss[:, 1:3] - 1
    # Connectivity depends only on occupied pixels, not on distant background.
    # Label the tight 2-D point bbox instead of allocating the full slice.
    lo = yz.min(axis=0)
    hi = yz.max(axis=0) + 1
    plane = np.zeros(tuple((hi - lo).astype(int)), dtype=bool)
    q = yz - lo[None, :]
    plane[q[:, 0], q[:, 1]] = True
    lab, n = ndimage.label(plane, structure=np.ones((3, 3), dtype=bool))
    if not n:
        return False
    cnt = np.bincount(lab.ravel())
    return int(np.count_nonzero(cnt[1:] >= 10)) > 1


def recon_cart_defect_curve_fc(subs_in: np.ndarray, size_img: Sequence[int]) -> np.ndarray:
    """Port of cal_reconCartDefect_curve_FC."""
    subs = np.asarray(subs_in, dtype=np.int64)
    if len(subs) == 0:
        return subs.copy()
    to_fill = [int(x) for x in np.unique(subs[:, 0])
               if _slice_needs_curve_fill(subs[subs[:, 0] == x], size_img)]
    filled = [subs]
    for sag in to_fill:
        sl = subs[subs[:, 0] == sag]
        if len(sl) < 20:
            continue
        pts = sl[:, [1, 2]].astype(np.float64)
        x, y = pts[:, 0], pts[:, 1]
        # cal_fitCircle: a = -[x y 1] \\ (x^2+y^2), cx=-a1/2.
        A = np.column_stack([x, y, np.ones(len(x))])
        a, *_ = np.linalg.lstsq(A, -(x * x + y * y), rcond=None)
        cx = -0.5 * a[0]
        center = matlab_round(np.array([cx, np.max(y + 1.0)])).astype(np.float64)
        q = pts - center[None, :]
        theta = np.arctan2(q[:, 1], q[:, 0])
        rho = np.hypot(q[:, 0], q[:, 1])
        tr = np.unique(np.column_stack([theta, rho]), axis=0)
        theta, rho = tr[:, 0], tr[:, 1]
        if len(theta) < 4 or np.max(rho) <= 0:
            continue
        coef = _best_polyfit_matlab(theta, rho, 7)
        arg = 1.0 / float(np.max(rho))
        if not (-1.0 <= arg <= 1.0):
            continue
        step = math.asin(arg) / 10.0
        if step <= 0:
            continue
        grid_theta = np.arange(float(np.min(theta)), float(np.max(theta)) + step * 0.5, step)
        grid_rho = np.polyval(coef, grid_theta)
        grid = np.column_stack([grid_rho * np.cos(grid_theta) + center[0],
                                grid_rho * np.sin(grid_theta) + center[1]])
        if len(grid) == 0:
            continue
        if (np.min(grid[:, 0]) > 0 and np.min(grid[:, 1]) > 0 and
                np.max(grid[:, 0]) <= size_img[1] and np.max(grid[:, 1]) <= size_img[2]):
            dist = _nearest_dist_numpy(pts, grid)
            pick = dist > math.sqrt(3.0)
            if np.any(pick):
                add2 = np.unique(matlab_round(grid[pick]).astype(np.int64), axis=0)
                filled.append(np.column_stack([np.full(len(add2), sag, dtype=np.int64), add2]))
    return np.vstack(filled)


def recon_cart_defect_curve_tc(subs_in: np.ndarray, size_img: Sequence[int]) -> np.ndarray:
    """Port of cal_reconCartDefect_curve_TC."""
    subs = np.asarray(subs_in, dtype=np.int64)
    if len(subs) == 0:
        return subs.copy()
    to_fill = [int(x) for x in np.unique(subs[:, 0])
               if _slice_needs_curve_fill(subs[subs[:, 0] == x], size_img)]
    filled = [subs]
    for sag in to_fill:
        sl = subs[subs[:, 0] == sag]
        if len(sl) < 20:
            continue
        pts = sl[:, [1, 2]].astype(np.float64)
        center = matlab_round(np.array([np.mean(pts[:, 0]), np.min(pts[:, 1]) - 1.0])).astype(np.float64)
        q = np.unique(pts - center[None, :], axis=0)
        x, y = q[:, 0], q[:, 1]
        if len(x) < 4:
            continue
        coef = _best_polyfit_matlab(x, y, 5)
        grid_xn = np.arange(float(np.min(x)), float(np.max(x)) + 0.05, 0.1)
        grid = np.column_stack([grid_xn + center[0], np.polyval(coef, grid_xn) + center[1]])
        if len(grid) == 0:
            continue
        if (np.min(grid[:, 0]) > 0 and np.min(grid[:, 1]) > 0 and
                np.max(grid[:, 0]) <= size_img[1] and np.max(grid[:, 1]) <= size_img[2]):
            dist = _nearest_dist_numpy(pts, grid)
            pick = dist > math.sqrt(3.0)
            if np.any(pick):
                add2 = np.unique(matlab_round(grid[pick]).astype(np.int64), axis=0)
                filled.append(np.column_stack([np.full(len(add2), sag, dtype=np.int64), add2]))
    return np.vstack(filled)


def build_total_subchondral_mesh(bone: np.ndarray, cart: np.ndarray, warped_cartilage: np.ndarray,
                                 cart_inner: Mesh, vox: np.ndarray, device: torch.device,
                                 cartilage_name: str, bone_mesh: Optional[Mesh] = None,
                                 debug_info: Optional[Dict[str, object]] = None,
                                 physical_mapping: bool = False,
                                 paper_fcl_geometry: bool = False,
                                 balanced_scb_closing: bool = False,
                                 stage_roi_debug: bool = False,
                                 interface_scb_seed: bool = False,
                                 overlap_interface_scb_seed: bool = False,
                                 constrained_interface_scb_seed: bool = False) -> Tuple[Mesh, Mesh]:
    """Reconstruct scB using the warped template cartilage prior.

    This follows the FCL reconstruction definition: use the principal SUBJECT bone
    surface, rebase the observed interface, map the overlap of subject cartilage and
    warped healthy-template cartilage to that bone surface, then apply component and
    filling, polynomial boundary-gap curve filling, surface closing and component
    cleanup.  Face operations keep the same percentage thresholds and iteration rules.
    """
    if debug_info is not None:
        debug_info.clear()
        debug_info.update({
            "cartilage_name": cartilage_name,
            "input_inner_vertices": int(len(cart_inner.vertices_sub)),
            "input_inner_faces": int(len(cart_inner.faces)),
            "input_inner_area_mm2": _mesh_area_numpy(cart_inner, vox),
            "cart_voxels": int(np.count_nonzero(cart)),
            "prior_voxels": int(np.count_nonzero(warped_cartilage)),
        })

    if not np.any(bone):
        if debug_info is not None:
            debug_info["status"] = "empty_bone"
        empty = Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
        return cart_inner, empty

    # FCL reconstruction is defined on the SUBJECT bone surface M_i,b.  Bone
    # region-growing belongs to the earlier cartilage-surface split (Eq. 11), not
    # to the tAB/FCL surface in Eqs. 21-22.  The previous port accidentally built
    # this mesh from a Gaussian-expanded bone mask, moving the interface several
    # voxels away and inflating tAB.
    if bone_mesh is None:
        b = _largest_component_26(bone)
        bm = boundary_mesh(b)
    else:
        bm = bone_mesh
    if bm.empty:
        if debug_info is not None:
            debug_info["status"] = "empty_bone_mesh"
        return cart_inner, cart_inner
    if debug_info is not None:
        debug_info.update({
            "bone_mesh_vertices": int(len(bm.vertices_sub)),
            "bone_mesh_faces": int(len(bm.faces)),
            "bone_mesh_area_mm2": _mesh_area_numpy(bm, vox),
        })
        if stage_roi_debug:
            # Private, in-memory diagnostic payload.  These arrays are stripped
            # before CSV/JSON serialization and never affect morphometrics.
            debug_info["_bone_mesh_vertices_sub"] = np.asarray(bm.vertices_sub, dtype=np.float64).copy()
            debug_info["_bone_mesh_faces"] = np.asarray(bm.faces, dtype=np.int64).copy()

    # Reuse the same GPU copy for all nearest-neighbour mappings on this bone.
    # Candidate parity fix: geometric nearest-neighbour operations should be
    # evaluated in physical mm coordinates on anisotropic images.  The legacy
    # port used raw voxel indices, which underweights the 0.7-mm LR axis relative
    # to the ~0.365-mm AP/SI axes in OAI DESS and can project FC voxels to the
    # wrong parts of the curved femoral surface.
    map_scale = np.asarray(vox, dtype=np.float64) if physical_mapping else np.ones(3, dtype=np.float64)
    src_b = torch.as_tensor(bm.vertices_sub * map_scale[None, :], dtype=torch.float32, device=device)
    if debug_info is not None:
        debug_info["nn_mapping_space"] = "physical_mm" if physical_mapping else "legacy_voxel_index"

    # Rebase the observed cartilage-bone interface onto the original subject bone
    # mesh.  This is only a marching-cubes compatibility shim: MATLAB's meshes
    # share many vertex rows exactly, while skimage's triangulation can differ.
    rebased_faces = np.empty((0, 3), dtype=np.int64)
    if len(cart_inner.vertices_sub):
        dst_i = torch.as_tensor(cart_inner.vertices_sub * map_scale[None, :], dtype=torch.float32, device=device)
        _, idx_i = nearest_indices_gpu(src_b, dst_i)
        mapped_ids = np.unique(idx_i.cpu().numpy())
        mapped = np.zeros(len(bm.vertices_sub), dtype=bool)
        mapped[mapped_ids] = True
        # Use faces touching mapped vertices, then complete the local patch.  This
        # compensates only for the triangulator coordinate convention; all later
        # regional membership is exact row membership on the rebased bone mesh.
        touched = bm.faces[np.any(mapped[bm.faces], axis=1)]
        if debug_info is not None:
            debug_info.update({
                "rebase_mapped_unique_bone_vertices": int(len(mapped_ids)),
                "rebase_touched_bone_faces": int(len(touched)),
                "rebase_touched_bone_area_mm2": _faces_area_numpy(bm, touched, vox),
            })
        if len(touched):
            local = np.zeros(len(bm.vertices_sub), dtype=bool)
            local[np.unique(touched)] = True
            rebased_faces = bm.faces[np.all(local[bm.faces], axis=1)]
            rebased_faces = _delete_sharp_edge_tri(rebased_faces)
    rebased_inner = submesh(bm, _unique_faces(rebased_faces)) if len(rebased_faces) else cart_inner
    if debug_info is not None:
        debug_info.update({
            "rebased_inner_vertices": int(len(rebased_inner.vertices_sub)),
            "rebased_inner_faces": int(len(rebased_inner.faces)),
            "rebased_inner_area_mm2": _mesh_area_numpy(rebased_inner, vox),
        })
        if stage_roi_debug:
            debug_info["_rebased_faces_full"] = np.asarray(_unique_faces(rebased_faces), dtype=np.int64).copy()

    # Eq. 22 of CartiMorph: M^b_i,c = O_m(S_i,c INTERSECT
    # (S^t_c o phi_i), M_i,b).  The previous port effectively used a UNION by
    # mapping the warped prior and then separately adding the whole subject
    # cartilage surface.  That makes the reconstructed total subchondral surface
    # much too large, especially in healthy knees.
    c_bool = np.asarray(cart, dtype=bool)
    w_bool = np.asarray(warped_cartilage, dtype=bool)
    if debug_info is not None:
        overlap_bool_dbg = c_bool & w_bool
        prior_only_bool_dbg = w_bool & ~c_bool
        subject_only_bool_dbg = c_bool & ~w_bool
        n_cart_dbg = int(np.count_nonzero(c_bool))
        n_overlap_dbg = int(np.count_nonzero(overlap_bool_dbg))
        debug_info.update({
            "prior_overlap_voxels_exact": n_overlap_dbg,
            "prior_only_voxels": int(np.count_nonzero(prior_only_bool_dbg)),
            "subject_only_voxels": int(np.count_nonzero(subject_only_bool_dbg)),
            "prior_overlap_fraction_of_subject": (float(n_overlap_dbg) / float(n_cart_dbg) if n_cart_dbg else float("nan")),
        })

        # Diagnostic only: Eq.22 still seeds from subject∩prior.  These mapped
        # footprints show whether the warped healthy prior contains surface that
        # is absent from the observed cartilage and should be recoverable as FCL.
        def _diag_map_voxels_to_bone(mask0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pts0 = np.argwhere(mask0)
            vm = np.zeros(len(bm.vertices_sub), dtype=bool)
            if len(pts0):
                dst0 = torch.as_tensor((pts0 + 1) * map_scale[None, :], dtype=torch.float32, device=device)
                _, ii0 = nearest_indices_gpu(src_b, dst0)
                vm[np.unique(ii0.cpu().numpy())] = True
            fm = np.any(vm[bm.faces], axis=1) if len(bm.faces) else np.zeros((0,), dtype=bool)
            return vm, fm

        prior_all_v_dbg, prior_all_f_dbg = _diag_map_voxels_to_bone(w_bool)
        prior_only_v_dbg, prior_only_f_dbg = _diag_map_voxels_to_bone(prior_only_bool_dbg)
        prior_all_faces_dbg = bm.faces[prior_all_f_dbg]
        prior_only_faces_dbg = bm.faces[prior_only_f_dbg]
        rebased_keys_dbg = _row_keys_int(np.sort(rebased_faces, axis=1)) if len(rebased_faces) else np.empty((0,), dtype=np.dtype((np.void, 24)))
        if len(prior_only_faces_dbg) and len(rebased_keys_dbg):
            po_keys_dbg = _row_keys_int(np.sort(prior_only_faces_dbg, axis=1))
            po_uncovered_dbg = prior_only_faces_dbg[~np.isin(po_keys_dbg, rebased_keys_dbg)]
        else:
            po_uncovered_dbg = prior_only_faces_dbg.copy()
        debug_info.update({
            "prior_all_mapped_bone_vertices": int(np.count_nonzero(prior_all_v_dbg)),
            "prior_all_mapped_faces": int(len(prior_all_faces_dbg)),
            "prior_all_mapped_area_mm2": _faces_area_numpy(bm, prior_all_faces_dbg, vox),
            "prior_only_mapped_bone_vertices": int(np.count_nonzero(prior_only_v_dbg)),
            "prior_only_mapped_faces": int(len(prior_only_faces_dbg)),
            "prior_only_mapped_area_mm2": _faces_area_numpy(bm, prior_only_faces_dbg, vox),
            "prior_only_faces_outside_observed_inner": int(len(po_uncovered_dbg)),
            "prior_only_area_outside_observed_inner_mm2": _faces_area_numpy(bm, po_uncovered_dbg, vox),
        })
        if stage_roi_debug:
            debug_info["_prior_all_faces_full"] = np.asarray(prior_all_faces_dbg, dtype=np.int64).copy()
            debug_info["_prior_only_faces_full"] = np.asarray(prior_only_faces_dbg, dtype=np.int64).copy()
            debug_info["_prior_miss_faces_full"] = np.asarray(po_uncovered_dbg, dtype=np.int64).copy()
    else:
        prior_only_faces_dbg = np.empty((0, 3), dtype=np.int64)
        po_uncovered_dbg = np.empty((0, 3), dtype=np.int64)

    if np.any(c_bool):
        sl_c, org_c = _foreground_bbox(c_bool, pad=0)
        overlap_local = c_bool[sl_c] & w_bool[sl_c]
        overlap0 = np.argwhere(overlap_local) + org_c[None, :]
    else:
        overlap0 = np.empty((0, 3), dtype=np.int64)
    if debug_info is not None:
        debug_info["cart_prior_overlap_voxels"] = int(len(overlap0))

    # Seed A/B modes.
    #
    # v15 compatibility candidate: keep v14's strict subject/prior overlap
    # bone-facing interface voxels, but DO NOT search the whole folded bone mesh
    # for their nearest vertices.  Every source voxel belongs to observed subject
    # cartilage, so its Eq.22 seed correspondence must remain on the observed
    # cartilage/bone interface.  We therefore search only bone vertices already
    # represented by the rebased observed iC, retain v14's voxel closing/component
    # filtering, then clamp the resulting bone faces back to that same observed
    # interface face domain.  This preserves a dense voxel-domain seed while
    # preventing the cross-condyle/global-nearest projection seen in v14.
    if constrained_interface_scb_seed:
        overlap_mask = c_bool & w_bool
        inter_overlap_subs, _ = init_boundary_split_w_bone(overlap_mask, bone, need_outer=False)
        rb = _unique_faces(rebased_faces) if len(rebased_faces) else np.empty((0, 3), dtype=np.int64)
        obs_vertex_ids = np.unique(rb) if len(rb) else np.empty((0,), dtype=np.int64)
        mapped_obs_ids = np.empty((0,), dtype=np.int64)
        if len(inter_overlap_subs) and len(obs_vertex_ids):
            src_obs = torch.as_tensor(
                bm.vertices_sub[obs_vertex_ids] * map_scale[None, :],
                dtype=torch.float32, device=device,
            )
            dst_p = torch.as_tensor(
                inter_overlap_subs * map_scale[None, :],
                dtype=torch.float32, device=device,
            )
            _, idx_local = nearest_indices_gpu(src_obs, dst_p)
            mapped_obs_ids = obs_vertex_ids[np.unique(idx_local.cpu().numpy())]

        seed_mask = np.zeros(cart.shape, dtype=bool)
        if len(mapped_obs_ids):
            ss = matlab_round(bm.vertices_sub[mapped_obs_ids]).astype(np.int64)
            ok = _valid_subs(ss, cart.shape)
            ii = _subs1_to_idx0(ss[ok])
            seed_mask[tuple(ii.T)] = True
        seed_before = int(np.count_nonzero(seed_mask))
        if np.any(seed_mask):
            sl_seed, _ = _foreground_bbox(seed_mask, pad=1)
            sm = ndimage.binary_closing(seed_mask[sl_seed], structure=np.ones((3, 3, 3), bool))
            nmin = int(matlab_round(float(sm.sum()) / 10.0))
            if nmin > 0:
                lab, n = ndimage.label(sm, structure=np.ones((3, 3, 3), bool))
                if n:
                    cnt = np.bincount(lab.ravel())
                    keep = cnt >= nmin; keep[0] = False
                    sm = keep[lab]
            seed_mask.fill(False)
            seed_mask[sl_seed] = sm

        # Convert the closed voxel seed back to the subject bone surface exactly
        # as v14 does, then discard any face outside the observed-iC face domain.
        ss_b = matlab_round(bm.vertices_sub).astype(np.int64)
        ok_b = _valid_subs(ss_b, cart.shape)
        vv = np.zeros(len(ss_b), dtype=bool)
        ii_b = _subs1_to_idx0(ss_b[ok_b])
        vv[ok_b] = seed_mask[tuple(ii_b.T)]
        faces_preclamp = _unique_faces(bm.faces[np.any(vv[bm.faces], axis=1)])
        if len(faces_preclamp) and len(rb):
            fp_keys = _row_keys_int(np.sort(faces_preclamp, axis=1))
            rb_keys = _row_keys_int(np.sort(rb, axis=1))
            faces = faces_preclamp[np.isin(fp_keys, rb_keys)]
        else:
            faces = np.empty((0, 3), dtype=np.int64)

        if debug_info is not None:
            pre_area = _faces_area_numpy(bm, faces_preclamp, vox)
            seed_area = _faces_area_numpy(bm, faces, vox)
            rb_area = _faces_area_numpy(bm, rb, vox)
            debug_info.update({
                "scb_seed_mode": "v15_constrained_overlap_interface_observed_iC",
                "overlap_interface_subs_count": int(len(inter_overlap_subs)),
                "overlap_interface_candidate_observed_vertices": int(len(obs_vertex_ids)),
                "overlap_interface_mapped_observed_vertices": int(len(mapped_obs_ids)),
                "seed_voxels_before_closing": seed_before,
                "seed_voxels_after_closing_component_filter": int(np.count_nonzero(seed_mask)),
                "seed_faces_before_observed_clamp": int(len(faces_preclamp)),
                "seed_area_before_observed_clamp_mm2": pre_area,
                "seed_faces_removed_by_observed_clamp": int(len(faces_preclamp) - len(faces)),
                "seed_area_removed_by_observed_clamp_mm2": float(max(pre_area - seed_area, 0.0)),
                "seed_faces_supported_by_observed_iC": int(len(faces)),
                "seed_area_supported_by_observed_iC_mm2": seed_area,
                "seed_observed_support_fraction_area": (1.0 if seed_area > 0 else float("nan")),
                "seed_rebased_coverage_fraction_area": (seed_area / rb_area if rb_area > 0 else float("nan")),
            })

    # v14 retains the same overlap-interface source but still maps it to the
    # globally nearest vertex of the full subject bone surface.  Kept only for
    # A/B reproducibility after the real folded-femur case exposed cross-sheet
    # seed correspondence.
    elif overlap_interface_scb_seed:
        overlap_mask = c_bool & w_bool
        inter_overlap_subs, _ = init_boundary_split_w_bone(overlap_mask, bone, need_outer=False)
        keep_v = np.zeros(len(bm.vertices_sub), dtype=bool)
        if len(inter_overlap_subs):
            dst_p = torch.as_tensor(inter_overlap_subs * map_scale[None, :], dtype=torch.float32, device=device)
            _, idx = nearest_indices_gpu(src_b, dst_p)
            keep_v[np.unique(idx.cpu().numpy())] = True

        seed_mask = np.zeros(cart.shape, dtype=bool)
        if np.any(keep_v):
            ss = matlab_round(bm.vertices_sub[keep_v]).astype(np.int64)
            ok = _valid_subs(ss, cart.shape)
            ii = _subs1_to_idx0(ss[ok])
            seed_mask[tuple(ii.T)] = True
        seed_before = int(np.count_nonzero(seed_mask))
        if np.any(seed_mask):
            sl_seed, _ = _foreground_bbox(seed_mask, pad=1)
            sm = ndimage.binary_closing(seed_mask[sl_seed], structure=np.ones((3, 3, 3), bool))
            nmin = int(matlab_round(float(sm.sum()) / 10.0))
            if nmin > 0:
                lab, n = ndimage.label(sm, structure=np.ones((3, 3, 3), bool))
                if n:
                    cnt = np.bincount(lab.ravel())
                    keep = cnt >= nmin; keep[0] = False
                    sm = keep[lab]
            seed_mask.fill(False)
            seed_mask[sl_seed] = sm

        ss_b = matlab_round(bm.vertices_sub).astype(np.int64)
        ok_b = _valid_subs(ss_b, cart.shape)
        vv = np.zeros(len(ss_b), dtype=bool)
        ii_b = _subs1_to_idx0(ss_b[ok_b])
        vv[ok_b] = seed_mask[tuple(ii_b.T)]
        faces = bm.faces[np.any(vv[bm.faces], axis=1)]

        if debug_info is not None:
            rb_keys = _row_keys_int(np.sort(_unique_faces(rebased_faces), axis=1)) if len(rebased_faces) else np.empty((0,), dtype=np.dtype((np.void, 24)))
            sf = _unique_faces(faces)
            if len(sf) and len(rb_keys):
                sf_keys = _row_keys_int(np.sort(sf, axis=1))
                supported_by_observed = sf[np.isin(sf_keys, rb_keys)]
            else:
                supported_by_observed = np.empty((0, 3), dtype=np.int64)
            seed_area = _faces_area_numpy(bm, sf, vox)
            obs_seed_area = _faces_area_numpy(bm, supported_by_observed, vox)
            debug_info.update({
                "scb_seed_mode": "v14_overlap_bone_interface_voxels",
                "overlap_interface_subs_count": int(len(inter_overlap_subs)),
                "overlap_interface_mapped_bone_vertices": int(np.count_nonzero(keep_v)),
                "seed_voxels_before_closing": seed_before,
                "seed_voxels_after_closing_component_filter": int(np.count_nonzero(seed_mask)),
                "seed_faces_supported_by_observed_iC": int(len(supported_by_observed)),
                "seed_area_supported_by_observed_iC_mm2": obs_seed_area,
                "seed_observed_support_fraction_area": (obs_seed_area / seed_area if seed_area > 0 else float("nan")),
            })

    # v13 experimental compatibility mode retained only for reproducibility.
    elif interface_scb_seed and len(rebased_faces):
        pts_prior = np.argwhere(w_bool)
        prior_v = np.zeros(len(bm.vertices_sub), dtype=bool)
        if len(pts_prior):
            dst_prior = torch.as_tensor((pts_prior + 1) * map_scale[None, :], dtype=torch.float32, device=device)
            _, idx_prior = nearest_indices_gpu(src_b, dst_prior)
            prior_v[np.unique(idx_prior.cpu().numpy())] = True
        prior_face_mask = np.any(prior_v[bm.faces], axis=1) if len(bm.faces) else np.zeros((0,), dtype=bool)
        prior_faces = bm.faces[prior_face_mask]

        rb = _unique_faces(rebased_faces)
        rb_keys = _row_keys_int(np.sort(rb, axis=1))
        pr_keys = _row_keys_int(np.sort(prior_faces, axis=1)) if len(prior_faces) else np.empty((0,), dtype=rb_keys.dtype)
        supported = np.isin(rb_keys, pr_keys) if len(rb_keys) else np.zeros((0,), dtype=bool)
        faces = rb[supported]

        if debug_info is not None:
            rb_area = _faces_area_numpy(bm, rb, vox)
            seed_area = _faces_area_numpy(bm, faces, vox)
            debug_info.update({
                "scb_seed_mode": "v13_observed_iC_intersect_prior_surface",
                "interface_seed_prior_mapped_vertices": int(np.count_nonzero(prior_v)),
                "interface_seed_prior_mapped_faces": int(len(prior_faces)),
                "interface_seed_rebased_faces": int(len(rb)),
                "interface_seed_supported_faces": int(len(faces)),
                "interface_seed_supported_area_mm2": seed_area,
                "interface_seed_rebased_support_fraction_area": (seed_area / rb_area if rb_area > 0 else float("nan")),
                "seed_voxels_before_closing": 0,
                "seed_voxels_after_closing_component_filter": 0,
            })
    else:
        # Legacy v3-v12 seed mapper: source overlap voxels -> nearest bone vertex.
        keep_v = np.zeros(len(bm.vertices_sub), dtype=bool)
        if len(overlap0):
            dst_p = torch.as_tensor((overlap0 + 1) * map_scale[None, :], dtype=torch.float32, device=device)
            _, idx = nearest_indices_gpu(src_b, dst_p)
            keep_v[np.unique(idx.cpu().numpy())] = True
        if debug_info is not None:
            debug_info["overlap_mapped_bone_vertices"] = int(np.count_nonzero(keep_v))
            debug_info["scb_seed_mode"] = "legacy_overlap_voxels_to_nearest_bone"

        seed_mask = np.zeros(cart.shape, dtype=bool)
        if np.any(keep_v):
            ss = matlab_round(bm.vertices_sub[keep_v]).astype(np.int64)
            ok = _valid_subs(ss, cart.shape)
            ii = _subs1_to_idx0(ss[ok])
            seed_mask[tuple(ii.T)] = True
        if debug_info is not None:
            debug_info["seed_voxels_before_closing"] = int(np.count_nonzero(seed_mask))
        if np.any(seed_mask):
            sl_seed, _ = _foreground_bbox(seed_mask, pad=1)
            sm = ndimage.binary_closing(seed_mask[sl_seed], structure=np.ones((3, 3, 3), bool))
            nmin = int(matlab_round(float(sm.sum()) / 10.0))
            if nmin > 0:
                lab, n = ndimage.label(sm, structure=np.ones((3, 3, 3), bool))
                if n:
                    cnt = np.bincount(lab.ravel())
                    keep = cnt >= nmin; keep[0] = False
                    sm = keep[lab]
            seed_mask.fill(False)
            seed_mask[sl_seed] = sm
        if debug_info is not None:
            debug_info["seed_voxels_after_closing_component_filter"] = int(np.count_nonzero(seed_mask))

        ss_b = matlab_round(bm.vertices_sub).astype(np.int64)
        ok_b = _valid_subs(ss_b, cart.shape)
        vv = np.zeros(len(ss_b), dtype=bool)
        ii_b = _subs1_to_idx0(ss_b[ok_b])
        vv[ok_b] = seed_mask[tuple(ii_b.T)]
        faces = bm.faces[np.any(vv[bm.faces], axis=1)]

    if debug_info is not None:
        debug_info.update({
            "seed_surface_faces": int(len(faces)),
            "seed_surface_area_mm2": _faces_area_numpy(bm, faces, vox),
        })
        if stage_roi_debug:
            debug_info["_stage_faces_seed"] = np.asarray(_unique_faces(faces), dtype=np.int64).copy()

    # 2.6(2): connected defect filling.
    faces = _fill_closed_surface_holes(faces, bm.faces)
    if debug_info is not None:
        debug_info.update({
            "after_hole_fill_faces": int(len(faces)),
            "after_hole_fill_area_mm2": _faces_area_numpy(bm, faces, vox),
        })
        if stage_roi_debug:
            debug_info["_stage_faces_hole"] = np.asarray(_unique_faces(faces), dtype=np.int64).copy()

    # 2.6(3): polynomial curve filling for gaps that reach the surface boundary.
    filled_mesh_1 = submesh(bm, _unique_faces(faces))
    subs_filled_1 = matlab_round(filled_mesh_1.vertices_sub).astype(np.int64)
    if cartilage_name == "FemoralCartilage":
        curve_subs = recon_cart_defect_curve_fc(subs_filled_1, cart.shape)
    else:
        tmp = recon_cart_defect_curve_tc(subs_filled_1, cart.shape)
        tmp_swap = tmp[:, [1, 0, 2]] if len(tmp) else tmp
        size_swap = (cart.shape[1], cart.shape[0], cart.shape[2])
        tmp2_swap = recon_cart_defect_curve_tc(tmp_swap, size_swap)
        curve_subs = tmp2_swap[:, [1, 0, 2]] if len(tmp2_swap) else tmp2_swap
    if len(curve_subs):
        dst_c = torch.as_tensor(curve_subs * map_scale[None, :], dtype=torch.float32, device=device)
        _, idx_c = nearest_indices_gpu(src_b, dst_c)
        curve_v = np.zeros(len(bm.vertices_sub), dtype=bool)
        curve_v[np.unique(idx_c.cpu().numpy())] = True
        faces = bm.faces[np.any(curve_v[bm.faces], axis=1)]
    if debug_info is not None:
        debug_info.update({
            "curve_subs_count": int(len(curve_subs)),
            "after_curve_fill_faces": int(len(faces)),
            "after_curve_fill_area_mm2": _faces_area_numpy(bm, faces, vox),
        })
        if stage_roi_debug:
            debug_info["_stage_faces_curve"] = np.asarray(_unique_faces(faces), dtype=np.int64).copy()

    # 2.6(4-7): fine mesh closing, sharp-edge removal, edge completion, keep large comps.
    extent = 4.0 if cartilage_name == "FemoralCartilage" else 3.0
    it_d = int(matlab_round(extent / float(vox[0])))
    # v9 A/B candidate.  The public CartiMorph closing operator takes separate
    # dilation/erosion counts.  The old port hard-coded erosion=dilation+4; in
    # oaizib_467 this removed most of the surface just added by curve filling.
    # Test a balanced closing without changing any earlier reconstruction step.
    it_e = it_d if balanced_scb_closing else it_d + 4
    faces = _surface_closing(faces, bm.faces, it_d, it_e)
    if debug_info is not None:
        debug_info.update({
            "scb_closing_extent_mm_nominal": float(extent),
            "scb_closing_dilation_iterations": int(it_d),
            "scb_closing_erosion_iterations": int(it_e),
            "scb_closing_mode": "balanced_dilation_erosion" if balanced_scb_closing else "legacy_erosion_plus4",
        })
    faces = _delete_sharp_edge_tri(faces)
    if len(faces):
        used = np.zeros(len(bm.vertices_sub), dtype=bool)
        used[np.unique(faces)] = True
        faces = bm.faces[np.all(used[bm.faces], axis=1)]
        comps = _face_components(faces)
        pct = _component_percentages(comps)
        faces = _concat_faces(comps, pct >= 50)
    if debug_info is not None:
        debug_info.update({
            "after_surface_closing_faces": int(len(faces)),
            "after_surface_closing_area_mm2": _faces_area_numpy(bm, faces, vox),
        })
        if stage_roi_debug:
            debug_info["_stage_faces_close"] = np.asarray(_unique_faces(faces), dtype=np.int64).copy()
    # Paper Eq. (21)-(22): tAB is the reconstructed surface obtained from the
    # subject/template overlap seed plus hole filling/closing.  The observed
    # cartilage surface is SUBTRACTED from that reconstructed surface for FCL;
    # it is not unioned back into tAB.  Older Python revisions forced
    # ``rebased_faces`` into tAB to make iC a subset, which can erase genuine
    # denuded area (most visibly when rebased_iC is larger than reconstructed
    # tibial tAB).  Keep the legacy union unless the explicit paper-parity
    # candidate is requested so A/B testing remains reproducible.
    faces_pre_observed_union = _unique_faces(faces)
    if debug_info is not None:
        debug_info.update({
            "pre_observed_union_faces": int(len(faces_pre_observed_union)),
            "pre_observed_union_area_mm2": _faces_area_numpy(bm, faces_pre_observed_union, vox),
            "observed_union_applied": bool((not paper_fcl_geometry) and len(rebased_faces)),
        })
    if (not paper_fcl_geometry) and len(rebased_faces):
        faces = _unique_faces(np.vstack([faces_pre_observed_union, rebased_faces])) if len(faces_pre_observed_union) else _unique_faces(rebased_faces)
    else:
        faces = faces_pre_observed_union
    scb = submesh(bm, _unique_faces(faces))
    if debug_info is not None:
        final_keys_dbg = _row_keys_int(np.sort(faces, axis=1)) if len(faces) else np.empty((0,), dtype=np.dtype((np.void, 24)))
        if len(po_uncovered_dbg) and len(final_keys_dbg):
            po_u_keys_dbg = _row_keys_int(np.sort(po_uncovered_dbg, axis=1))
            po_in_scb_dbg = po_uncovered_dbg[np.isin(po_u_keys_dbg, final_keys_dbg)]
        else:
            po_in_scb_dbg = np.empty((0, 3), dtype=np.int64)
        po_area_dbg = _faces_area_numpy(bm, po_uncovered_dbg, vox) if len(po_uncovered_dbg) else 0.0
        po_in_area_dbg = _faces_area_numpy(bm, po_in_scb_dbg, vox) if len(po_in_scb_dbg) else 0.0
        debug_info.update({
            "status": "ok",
            "final_scb_vertices": int(len(scb.vertices_sub)),
            "final_scb_faces": int(len(scb.faces)),
            "final_scb_area_mm2": _mesh_area_numpy(scb, vox),
            "prior_only_uncovered_faces_recovered_in_final_scb": int(len(po_in_scb_dbg)),
            "prior_only_uncovered_area_recovered_in_final_scb_mm2": float(po_in_area_dbg),
            "prior_only_uncovered_area_recovery_fraction": (float(po_in_area_dbg / po_area_dbg) if po_area_dbg > 0 else float("nan")),
        })
    return rebased_inner, scb


def map_surface_labels_to_cartilage_gpu(cart_mask: np.ndarray, interface_subs: np.ndarray,
                                        interface_labels: np.ndarray, device: torch.device) -> np.ndarray:
    """Port of cal_mapSeg2Mask for regional volume assignment using GPU NN."""
    out = np.zeros(cart_mask.shape, np.uint8)
    src_ok = interface_labels > 0
    if not np.any(cart_mask) or not np.any(src_ok):
        return out
    src = torch.as_tensor(interface_subs[src_ok], dtype=torch.float32, device=device)
    src_lab = torch.as_tensor(interface_labels[src_ok], dtype=torch.uint8, device=device)
    dest0 = np.argwhere(cart_mask)
    dst = torch.as_tensor(dest0 + 1, dtype=torch.float32, device=device)
    _, idx = nearest_indices_gpu(src, dst)
    labs = src_lab[idx].cpu().numpy()
    out[tuple(dest0.T)] = labs
    return out


def cartilage_thickness_gpu(inner: Mesh, outer: Mesh, vox: np.ndarray, depth: float,
                            n_roi: int, device: torch.device) -> torch.Tensor:
    if inner.empty or outer.empty:
        return torch.zeros((len(inner.vertices_sub),), dtype=torch.float64, device=device)
    v = torch.as_tensor(inner.vertices_sub * vox[None,:], dtype=torch.float64, device=device)
    vo = torch.as_tensor(outer.vertices_sub * vox[None,:], dtype=torch.float64, device=device)
    fo = torch.as_tensor(outer.faces, dtype=torch.long, device=device)
    tri = vo[fo]
    n_neigh = max(int(math.ceil(len(v) / float(n_roi))), 9)
    k_vote = min(max(int(n_neigh * 10), 2), len(v))
    # The same inner-surface KNN queries are used by normal estimation,
    # orientation voting and smoothing. Compute their distance matrix once.
    knn = knn_indices_multi_gpu(v, [n_neigh, k_vote])
    k_norm = min(max(int(n_neigh), 3), len(v))
    k_smooth = min(max(int(n_neigh), 1), len(v))
    sn = estimate_normals_gpu(v, n_neigh, knn[k_norm])
    # CMT_cal_reorientSN first removes outer vertices duplicated on the interface.
    dup = _row_membership(outer.vertices_sub.astype(np.int64), inner.vertices_sub.astype(np.int64))
    vo_reorient_np = (outer.vertices_sub[~dup] if np.any(~dup) else outer.vertices_sub) * vox[None,:]
    vo_reorient = torch.as_tensor(vo_reorient_np, dtype=torch.float64, device=device)
    sn = reorient_normals_gpu(
        sn, v, vo_reorient, torch.as_tensor(vox, dtype=torch.float64, device=device),
        n_neigh * 10, knn[k_vote]
    )
    sn = smooth_normals_gpu(sn, v, n_neigh, knn[k_smooth])
    # KNN index caches can be tens of MB on a full femoral surface; release the
    # references before the ray kernel so CUDA can reuse that memory immediately.
    del knn
    th = ray_mesh_nearest_gpu(v, sn, tri, depth)
    th = approximate_zeros_gpu(th, v, int(math.ceil(n_neigh/2)))
    unc = ellipsoid_uncertainty_gpu(sn, torch.as_tensor(vox, dtype=torch.float64, device=device))
    # MATLAB ThicknessMap adds twofold uncertainty.
    return th + 2.0 * unc


def area_by_roi_gpu(mesh: Mesh, labels_v: np.ndarray, vox: np.ndarray, device: torch.device) -> torch.Tensor:
    out = torch.zeros(20, dtype=torch.float64, device=device)
    if mesh.empty:
        return out
    v = torch.as_tensor(mesh.vertices_sub * vox[None,:], dtype=torch.float64, device=device)
    f = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
    lv = torch.as_tensor(labels_v.astype(np.int64), dtype=torch.long, device=device)
    ar = triangle_area_gpu(v, f)
    fl = lv[f]
    # MATLAB surface subregion meshes are extracted with cal_extractFaces_OR in the
    # parcellation code: at least one target vertex makes a face part of that ROI.
    for r in range(1, 21):
        out[r-1] = ar[torch.any(fl == r, dim=1)].sum()
    return out





def _target_face_covered_by_mesh(target: Mesh, coverage: Mesh) -> np.ndarray:
    """Exact face-set intersection on shared integer-coordinate geometry.

    CartiMorph Eq. (21) defines FCL as reconstructed tAB surface minus the
    observed cartilage-covered surface.  After the compatibility rebase both
    meshes are subsets of the same subject bone mesh, so coverage is a true
    geometric face-set intersection rather than an area subtraction between two
    independently-sized surfaces.
    """
    if target.empty or coverage.empty:
        return np.zeros(len(target.faces), dtype=bool)
    cv = np.asarray(coverage.vertices_sub, dtype=np.int64)
    tv = np.asarray(target.vertices_sub, dtype=np.int64)
    allv = np.vstack([cv, tv])
    _, inv = np.unique(allv, axis=0, return_inverse=True)
    ci = inv[:len(cv)]
    ti = inv[len(cv):]
    cf = np.sort(ci[np.asarray(coverage.faces, dtype=np.int64)], axis=1)
    tf = np.sort(ti[np.asarray(target.faces, dtype=np.int64)], axis=1)
    return _row_membership(tf, cf)


def _target_vertex_source_indices(target: Mesh, source: Mesh) -> np.ndarray:
    """Map each target vertex to the exact same-coordinate source vertex or -1."""
    if target.empty:
        return np.empty((0,), dtype=np.int64)
    if source.empty:
        return np.full(len(target.vertices_sub), -1, dtype=np.int64)
    sv = np.asarray(source.vertices_sub, dtype=np.int64)
    tv = np.asarray(target.vertices_sub, dtype=np.int64)
    allv = np.vstack([sv, tv])
    _, inv = np.unique(allv, axis=0, return_inverse=True)
    si = inv[:len(sv)]
    ti = inv[len(sv):]
    lut = np.full(int(inv.max()) + 1, -1, dtype=np.int64)
    # Mesh vertices are deduplicated, so each source coordinate has one index.
    lut[si] = np.arange(len(sv), dtype=np.int64)
    return lut[ti]


def paper_cab_tab_by_roi_gpu(inner: Mesh, scb: Mesh, scb_masks: Dict[int, np.ndarray],
                              vox: np.ndarray, device: torch.device
                              ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Compute cAB/tAB on one common tAB face domain (paper Eq. 21)."""
    cab = torch.zeros(20, dtype=torch.float64, device=device)
    tab = torch.zeros(20, dtype=torch.float64, device=device)
    covered = _target_face_covered_by_mesh(scb, inner)
    if scb.empty:
        return cab, tab, covered
    v = torch.as_tensor(scb.vertices_sub * vox[None, :], dtype=torch.float64, device=device)
    f = torch.as_tensor(scb.faces, dtype=torch.long, device=device)
    ar = triangle_area_gpu(v, f)
    cov = torch.as_tensor(covered, dtype=torch.bool, device=device)
    for r, mask in scb_masks.items():
        if r < 1 or r > 20 or len(mask) != len(scb.vertices_sub) or not np.any(mask):
            continue
        m = torch.as_tensor(mask, dtype=torch.bool, device=device)
        rf = torch.any(m[f], dim=1)  # CM_cal_extractFaces_OR regional semantics
        tab[r - 1] += ar[rf].sum()
        cab[r - 1] += ar[rf & cov].sum()
    return cab, tab, covered

def area_by_roi_masks_gpu(mesh: Mesh, roi_masks: Dict[int, np.ndarray], vox: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    """Regional area with MATLAB cal_extractFaces_OR semantics."""
    out = torch.zeros(20, dtype=torch.float64, device=device)
    if mesh.empty:
        return out
    v = torch.as_tensor(mesh.vertices_sub * vox[None, :], dtype=torch.float64, device=device)
    f = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
    ar = triangle_area_gpu(v, f)
    for r, mask in roi_masks.items():
        if r < 1 or r > 20 or len(mask) != len(mesh.vertices_sub) or not np.any(mask):
            continue
        m = torch.as_tensor(mask, dtype=torch.bool, device=device)
        out[r - 1] += ar[torch.any(m[f], dim=1)].sum()
    return out


def roi_vertex_labels(roi_masks: Dict[int, np.ndarray], n_vertices: int) -> np.ndarray:
    """Reproduce the wrapper_MorphQuant ROI loop overwrite order (1 -> 20)."""
    lab = np.zeros(n_vertices, dtype=np.uint8)
    for r in range(1, 21):
        m = roi_masks.get(r)
        if m is not None and len(m) == n_vertices:
            lab[np.asarray(m, dtype=bool)] = r
    return lab

def regional_fcl_gpu(cAB: torch.Tensor, tAB: torch.Tensor, legacy_fraction: bool = False) -> torch.Tensor:
    """CartiMorph FCL/dABp per ROI.

    CartiMorph defines FCL as the percentage of total subchondral bone area that
    is denuded.  Default output is therefore 0..100 percent.  ``legacy_fraction``
    reproduces the earlier Python-port convention 0..1 for backward compatibility.
    """
    out = torch.full_like(tAB, torch.nan, dtype=torch.float64)
    nz = tAB != 0
    frac = torch.clamp((tAB[nz] - cAB[nz]) / tAB[nz], min=0.0)
    out[nz] = frac if legacy_fraction else 100.0 * frac
    return out


def regional_volumes_gpu(volume_labelmap: np.ndarray, vox: np.ndarray, device: torch.device,
                         legacy_matlab_norm: bool = False) -> torch.Tensor:
    """Return 20 regional volumes from a 1..20 label map.

    Default is the physically correct CartiMorph method definition: voxel count
    multiplied by ``sx*sy*sz`` (mm^3).  ``legacy_matlab_norm`` reproduces the
    anomalous ``norm(voxSize)`` expression found in one Toolbox wrapper source and
    is retained only for source-compatibility experiments.
    """
    vt = torch.as_tensor(vox, dtype=torch.float64, device=device)
    scale = torch.linalg.vector_norm(vt) if legacy_matlab_norm else torch.prod(vt)
    vl = torch.as_tensor(np.asarray(volume_labelmap, dtype=np.int64), dtype=torch.long, device=device)
    # Exact integer histogram instead of scanning the whole 3-D volume 20 times.
    counts = torch.bincount(vl.reshape(-1), minlength=21)[1:21]
    return counts.to(dtype=torch.float64) * scale

def morphometrics_gpu(seg: np.ndarray, reg: np.ndarray, vox: np.ndarray, labels: LabelConfig,
                      knee_side: str, device: torch.device, template_seg: Optional[np.ndarray] = None,
                      cc_percentage: float = 0.6, depth: float = 7.0,
                      nroi_fc: int = 1000, nroi_mtc: int = 200, nroi_ltc: int = 200,
                      legacy_matlab_volume_norm: bool = False,
                      legacy_fcl_fraction: bool = False,
                      profile: bool = False,
                      debug_surface_data: Optional[dict] = None,
                      physical_scb_mapping: bool = False,
                      cart_surface_finetune: bool = False,
                      surface_closing_iterations: int = 4,
                      paper_fcl_geometry: bool = False,
                      balanced_scb_closing: bool = False,
                      matlab_native_ic: bool = False,
                      compat_native_ic_map: bool = False,
                      stage_roi_debug: bool = False,
                      interface_scb_seed: bool = False,
                      overlap_interface_scb_seed: bool = False,
                      constrained_interface_scb_seed: bool = False,
                      fc_contact_augment_inner: bool = False,
                      final_export_data: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB-logic morphology analysis with CUDA metric computation.

    Registration is resolved to the warped template tissue prior.  scB/FCL is
    reconstructed per cartilage from that prior, then the 20 regions are generated
    from the reconstructed surfaces using the FC/TC SurfaceParcellation rules.
    """
    profile_times: Dict[str, float] = {}
    total_t0 = time.perf_counter()

    def _prof_start() -> float:
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _prof_end(name: str, t0: float) -> None:
        if not profile:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile_times[name] = profile_times.get(name, 0.0) + (time.perf_counter() - t0)

    if seg.ndim != 3:
        raise ValueError(f"Segmentation must be 3D, got {seg.shape}")
    if reg.shape[:3] != seg.shape:
        raise ValueError(f"Segmentation and registration shapes are incompatible: {seg.shape} vs {reg.shape}")
    _t = _prof_start()
    warped_tissue = resolve_warped_tissue(reg, labels, device, template_seg)
    _prof_end("00_registration_resolve", _t)
    if warped_tissue.shape != seg.shape:
        raise ValueError(f"Warped template segmentation shape {warped_tissue.shape} != segmentation {seg.shape}")

    # cal_preprocessImg threshold: ceil(1 / (sx*sy*sz)).
    voxel_phys = float(np.prod(vox))
    n_remove = int(math.ceil(1.0 / voxel_phys))
    _t = _prof_start()
    femur = preprocess_binary(seg == labels.femur, n_remove)
    tibia = preprocess_binary(seg == labels.tibia, n_remove)
    _prof_end("01_bone_preprocess", _t)

    # mTC and lTC share the same subject tibia.  Build the principal bone mesh
    # once instead of repeating connected-component extraction + marching cubes.
    _t = _prof_start()
    femur_bm = boundary_mesh(_largest_component_26(femur)) if np.any(femur) else Mesh(
        np.empty((0, 3), np.float64), np.empty((0, 3), np.int64)
    )
    tibia_bm = boundary_mesh(_largest_component_26(tibia)) if np.any(tibia) else Mesh(
        np.empty((0, 3), np.float64), np.empty((0, 3), np.int64)
    )
    _prof_end("02_bone_meshes", _t)

    specs = [
        ("FC", "FemoralCartilage", labels.femoral_cartilage, femur, femur_bm, nroi_fc),
        ("mTC", "mTibialCartilage", labels.medial_tibial_cartilage, tibia, tibia_bm, nroi_mtc),
        ("lTC", "lTibialCartilage", labels.lateral_tibial_cartilage, tibia, tibia_bm, nroi_ltc),
    ]
    states: Dict[str, dict] = {}
    for key, matlab_name, cart_label, bone, bone_mesh, nroi in specs:
        _t = _prof_start()
        cart = preprocess_binary(seg == cart_label, n_remove)
        whole = boundary_mesh(cart) if np.any(cart) else Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
        split_debug = {} if debug_surface_data is not None else None
        inner, outer = split_cartilage_mesh(
            cart, bone, whole,
            finetune=cart_surface_finetune,
            closing_iterations=surface_closing_iterations,
            debug_info=split_debug,
            vox=vox,
            contact_augment_inner=(fc_contact_augment_inner and key == "FC"),
        ) if not whole.empty else (
            Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64)),
            Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64)),
        )
        split_inner = inner
        _prof_end(f"10_{key}_cart_mesh_split", _t)
        prior = warped_tissue == cart_label
        scb_debug = {} if debug_surface_data is not None else None
        _t = _prof_start()
        if not split_inner.empty:
            # build_total_subchondral_mesh returns a bone-rebased copy of iC only
            # as a Python marching-cubes compatibility aid for tAB reconstruction.
            # MATLAB keeps mesh_iC and mesh_scB as independent meshes during
            # morphometrics.  v10 therefore does not overwrite the native iC.
            rebased_inner, scb = build_total_subchondral_mesh(
                bone, cart, prior, split_inner, vox, device, matlab_name, bone_mesh=bone_mesh,
                debug_info=scb_debug, physical_mapping=physical_scb_mapping,
                # Both v8 and v10 keep reconstructed tAB independent of observed iC.
                paper_fcl_geometry=(paper_fcl_geometry or matlab_native_ic or compat_native_ic_map),
                balanced_scb_closing=balanced_scb_closing,
                stage_roi_debug=stage_roi_debug,
                interface_scb_seed=interface_scb_seed,
                overlap_interface_scb_seed=overlap_interface_scb_seed,
                constrained_interface_scb_seed=constrained_interface_scb_seed,
            )
        else:
            rebased_inner = split_inner
            scb = Mesh(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
        _prof_end(f"11_{key}_scb_reconstruction", _t)

        # Strict MATLAB metric geometry candidate: thickness is measured from the
        # native cartilage inner surface (mesh_iC), not from its bone-projected
        # compatibility copy.  The old v3-v9 path is preserved when the flag is off.
        metric_inner = split_inner if (matlab_native_ic or compat_native_ic_map) else rebased_inner
        _t = _prof_start()
        th = cartilage_thickness_gpu(metric_inner, outer, vox, depth, nroi, device) \
            if not metric_inner.empty else torch.empty((0,), dtype=torch.float64, device=device)
        _prof_end(f"12_{key}_thickness", _t)
        states[key] = {
            "cart": cart, "inner": metric_inner, "split_inner": split_inner,
            "rebased_inner": rebased_inner,
            "outer": outer, "scb": scb, "scb_debug": scb_debug, "split_debug": split_debug,
            "thickness": th, "inner_masks": {}, "scb_masks": {},
        }

    # MATLAB stage [2] Cartilage Parcellation: rule-based *surface* partitioning.
    _t = _prof_start()
    if compat_native_ic_map:
        # v11: preserve v9's robust row correspondence on rebased iC, then
        # transfer those ROI identities back to native iC for cAB/thickness.
        fc_rb_im, fc_sm = surface_parcellation_fc(
            states["FC"]["rebased_inner"], states["FC"]["scb"], vox, knee_side, cc_percentage
        )
        fc_im = transfer_roi_masks_nn(states["FC"]["rebased_inner"], states["FC"]["inner"], fc_rb_im, device)
        states["FC"]["inner_masks"], states["FC"]["scb_masks"] = fc_im, fc_sm
        states["FC"]["coverage_inner_masks"] = fc_rb_im

        mt_rb_im, mt_sm, lt_rb_im, lt_sm = surface_parcellation_tc(
            states["mTC"]["rebased_inner"], states["lTC"]["rebased_inner"],
            states["mTC"]["scb"], states["lTC"]["scb"], vox, knee_side
        )
        mt_im = transfer_roi_masks_nn(states["mTC"]["rebased_inner"], states["mTC"]["inner"], mt_rb_im, device)
        lt_im = transfer_roi_masks_nn(states["lTC"]["rebased_inner"], states["lTC"]["inner"], lt_rb_im, device)
        states["mTC"]["inner_masks"], states["mTC"]["scb_masks"] = mt_im, mt_sm
        states["lTC"]["inner_masks"], states["lTC"]["scb_masks"] = lt_im, lt_sm
        states["mTC"]["coverage_inner_masks"] = mt_rb_im
        states["lTC"]["coverage_inner_masks"] = lt_rb_im
    else:
        fc_im, fc_sm = surface_parcellation_fc(
            states["FC"]["inner"], states["FC"]["scb"], vox, knee_side, cc_percentage
        )
        states["FC"]["inner_masks"], states["FC"]["scb_masks"] = fc_im, fc_sm

        mt_im, mt_sm, lt_im, lt_sm = surface_parcellation_tc(
            states["mTC"]["inner"], states["lTC"]["inner"],
            states["mTC"]["scb"], states["lTC"]["scb"], vox, knee_side
        )
        states["mTC"]["inner_masks"], states["mTC"]["scb_masks"] = mt_im, mt_sm
        states["lTC"]["inner_masks"], states["lTC"]["scb_masks"] = lt_im, lt_sm
    _prof_end("20_surface_parcellation", _t)

    _t_metrics = _prof_start()
    cAB = torch.zeros(20, dtype=torch.float64, device=device)
    tAB = torch.zeros(20, dtype=torch.float64, device=device)
    mean_th = torch.zeros(20, dtype=torch.float64, device=device)
    volume_labelmap = np.zeros(seg.shape, dtype=np.uint8)

    # cAB / tAB and ThCtAB_aMe.  Areas and reductions are performed on GPU.
    for st in states.values():
        inner: Mesh = st["inner"]
        scb: Mesh = st["scb"]
        imasks: Dict[int, np.ndarray] = st["inner_masks"]
        smasks: Dict[int, np.ndarray] = st["scb_masks"]
        th: torch.Tensor = st["thickness"]

        inner_int = matlab_round(inner.vertices_sub).astype(np.int64) if len(inner.vertices_sub) else np.empty((0, 3), np.int64)
        scb_int = matlab_round(scb.vertices_sub).astype(np.int64) if len(scb.vertices_sub) else np.empty((0, 3), np.int64)

        if paper_fcl_geometry and not matlab_native_ic:
            # Eq. (21): FCL is a SET DIFFERENCE on the reconstructed tAB surface.
            # Compute both cAB and tAB on that single face domain.  This avoids
            # subtracting two independently-sized meshes and guarantees cAB<=tAB.
            cab_st, tab_st, covered_faces = paper_cab_tab_by_roi_gpu(
                inner, scb, smasks, vox, device
            )
            cAB += cab_st
            tAB += tab_st

            # ThCtAB.Me is likewise defined on tAB: covered tAB vertices receive
            # their observed inner-surface thickness; denuded tAB vertices are 0.
            # Observed inner vertices outside reconstructed tAB are not part of the
            # denominator (the old forced-union shim incorrectly added them to tAB).
            src_idx = _target_vertex_source_indices(scb, inner)
            scb_th = torch.zeros(len(scb.vertices_sub), dtype=torch.float64, device=device)
            matched = src_idx >= 0
            if np.any(matched):
                mt = torch.as_tensor(matched, dtype=torch.bool, device=device)
                si = torch.as_tensor(src_idx[matched], dtype=torch.long, device=device)
                scb_th[mt] = th[si]
            # Keep these final paper-geometry arrays for final reporting only.
            # They do NOT alter any metric calculation.
            st["_covered_faces_final"] = np.asarray(covered_faces, dtype=bool).copy()
            st["_scb_th_final"] = scb_th

            for r, smask0 in smasks.items():
                smask = np.asarray(smask0, dtype=bool)
                if not np.any(smask):
                    mean_th[r - 1] = 0.0
                    continue
                vals_t = scb_th[torch.as_tensor(smask, dtype=torch.bool, device=device)]
                mean_th[r - 1] = vals_t.mean() if vals_t.numel() else 0.0

            if debug_surface_data is not None:
                roi_rows = debug_surface_data.setdefault("rois", [])
                cab_dbg = cab_st.detach().cpu().numpy()
                tab_dbg = tab_st.detach().cpu().numpy()
                scb_areas = _mesh_face_areas_numpy(scb, vox)
                for r in sorted(smasks):
                    smask = np.asarray(smasks[r], dtype=bool)
                    sv = scb_int[smask] if len(scb_int) and len(smask) else np.empty((0, 3), np.int64)
                    src_r = src_idx[smask] if len(src_idx) and len(smask) else np.empty((0,), dtype=np.int64)
                    matched_r = src_r >= 0
                    vals_np = (th[torch.as_tensor(src_r[matched_r], dtype=torch.long, device=device)]
                               .detach().cpu().numpy()) if np.any(matched_r) else np.empty((0,), dtype=np.float64)
                    pos = vals_np[vals_np > 0]
                    vals_all = (scb_th[torch.as_tensor(smask, dtype=torch.bool, device=device)]
                                .detach().cpu().numpy()) if np.any(smask) else np.empty((0,), dtype=np.float64)
                    nzero_dbg = int(np.count_nonzero(~matched_r))
                    cabv = float(cab_dbg[r - 1]); tabv = float(tab_dbg[r - 1])
                    fcl_pct = float(max((tabv - cabv) / tabv, 0.0) * 100.0) if tabv != 0 else float("nan")
                    if len(scb.faces) and np.any(smask):
                        region_faces = np.any(smask[scb.faces], axis=1)
                        n_region_faces = int(np.count_nonzero(region_faces))
                        n_cov_faces = int(np.count_nonzero(region_faces & covered_faces))
                    else:
                        n_region_faces = 0; n_cov_faces = 0
                    roi_rows.append({
                        "compartment": next((k for k, v in states.items() if v is st), ""),
                        "roi": ROI_NAMES[r - 1],
                        "roi_index": int(r),
                        "metric_geometry_mode": "paper_tAB_set_difference",
                        "cAB_mm2": cabv,
                        "tAB_mm2": tabv,
                        "FCL_percent": fcl_pct,
                        "covered_scb_faces": n_cov_faces,
                        "regional_scb_faces": n_region_faces,
                        "inner_vertices": int(np.count_nonzero(matched_r)),
                        "scb_vertices": int(len(sv)),
                        "scb_not_inner_nzero": nzero_dbg,
                        "inner_not_scb": 0,
                        "zero_fraction_of_scb_vertices": float(nzero_dbg / len(sv)) if len(sv) else float("nan"),
                        "inner_thickness_zero_hits": int(np.count_nonzero(vals_np == 0)),
                        "inner_thickness_raw_mean_mm": float(vals_np.mean()) if len(vals_np) else float("nan"),
                        "inner_thickness_positive_mean_mm": float(pos.mean()) if len(pos) else float("nan"),
                        "final_zero_padded_mean_mm": float(vals_all.mean()) if len(vals_all) else 0.0,
                        "zero_padded_denominator": int(len(vals_all)),
                    })
        else:
            # MATLAB-style independent metric meshes.  In v10 ``inner`` is the
            # native cartilage iC while tAB remains the independently reconstructed
            # scB.  wrapper_regionalMorphQuant measures their regional areas
            # separately and dABp is clipped later by regional_fcl_gpu.
            # With v10 disabled this also retains the original legacy branch.
            cab_st = area_by_roi_masks_gpu(inner, imasks, vox, device)
            tab_st = area_by_roi_masks_gpu(scb, smasks, vox, device)
            cAB += cab_st
            tAB += tab_st
            for r, imask in imasks.items():
                if not np.any(imask):
                    # wrapper_regionalMorphQuant explicitly returns 0 when vers_iC is empty.
                    mean_th[r - 1] = 0.0
                    continue
                imask_t = torch.as_tensor(imask, dtype=torch.bool, device=device)
                vals = th[imask_t]
                smask = smasks.get(r, np.zeros(len(scb_int), dtype=bool))
                if compat_native_ic_map:
                    rb = st["rebased_inner"]
                    rb_int = matlab_round(rb.vertices_sub).astype(np.int64) if len(rb.vertices_sub) else np.empty((0, 3), np.int64)
                    cmask = np.asarray(st.get("coverage_inner_masks", {}).get(r, np.zeros(len(rb_int), dtype=bool)), dtype=bool)
                    regional_i = {tuple(x) for x in rb_int[cmask].tolist()}
                else:
                    regional_i = {tuple(x) for x in inner_int[imask].tolist()}
                nzero = sum(tuple(x) not in regional_i for x in scb_int[np.asarray(smask, dtype=bool)].tolist())
                denom = vals.numel() + nzero
                mean_th[r - 1] = vals.sum() / float(denom) if denom else 0.0

            if debug_surface_data is not None:
                roi_rows = debug_surface_data.setdefault("rois", [])
                cab_dbg = cab_st.detach().cpu().numpy()
                tab_dbg = tab_st.detach().cpu().numpy()
                for r in sorted(imasks):
                    imask = np.asarray(imasks[r], dtype=bool)
                    smask = np.asarray(smasks.get(r, np.zeros(len(scb_int), dtype=bool)), dtype=bool)
                    iv = inner_int[imask] if len(inner_int) and len(imask) else np.empty((0, 3), np.int64)
                    sv = scb_int[smask] if len(scb_int) and len(smask) else np.empty((0, 3), np.int64)
                    sset = {tuple(x) for x in sv.tolist()}
                    if compat_native_ic_map:
                        rb_cov = st["rebased_inner"]
                        rb_cov_int = matlab_round(rb_cov.vertices_sub).astype(np.int64) if len(rb_cov.vertices_sub) else np.empty((0, 3), np.int64)
                        rb_cov_mask = np.asarray(st.get("coverage_inner_masks", {}).get(r, np.zeros(len(rb_cov_int), dtype=bool)), dtype=bool)
                        iset = {tuple(x) for x in rb_cov_int[rb_cov_mask].tolist()}
                        inner_not_in_scb = 0
                    else:
                        iset = {tuple(x) for x in iv.tolist()}
                        inner_not_in_scb = sum(tuple(x) not in sset for x in iv.tolist())
                    nzero_dbg = sum(tuple(x) not in iset for x in sv.tolist())
                    vals_dbg = th[torch.as_tensor(imask, dtype=torch.bool, device=device)] if len(imask) else th[:0]
                    vals_np = vals_dbg.detach().cpu().numpy()
                    pos = vals_np[vals_np > 0]
                    denom_dbg = int(len(vals_np) + nzero_dbg)
                    final_dbg = float(vals_np.sum() / denom_dbg) if denom_dbg else 0.0
                    cabv = float(cab_dbg[r - 1])
                    tabv = float(tab_dbg[r - 1])
                    fcl_pct = float(max((tabv - cabv) / tabv, 0.0) * 100.0) if tabv != 0 else float("nan")
                    # v10 A/B diagnostic: evaluate how the old bone-rebased
                    # compatibility surface would populate this same scB region,
                    # without using it for the actual metrics.
                    rb_n = None
                    rb_area = None
                    if (matlab_native_ic or compat_native_ic_map):
                        rb = st.get("rebased_inner", inner)
                        rb_int = (matlab_round(rb.vertices_sub).astype(np.int64)
                                  if len(rb.vertices_sub) else np.empty((0, 3), np.int64))
                        rb_mask = _region_membership_from_subs(rb_int, sv) if len(sv) else np.zeros(len(rb_int), dtype=bool)
                        rb_n = int(np.count_nonzero(rb_mask))
                        if len(rb.faces) and len(rb_mask):
                            rb_faces = rb.faces[np.any(rb_mask[rb.faces], axis=1)]
                            rb_area = float(_faces_area_numpy(rb, rb_faces, vox))
                        else:
                            rb_area = 0.0
                    roi_rows.append({
                        "compartment": next((k for k, v in states.items() if v is st), ""),
                        "roi": ROI_NAMES[r - 1],
                        "roi_index": int(r),
                        "metric_geometry_mode": ("v11_native_iC_via_rebased_correspondence" if compat_native_ic_map
                                                 else "matlab_native_iC_independent_cAB_tAB" if matlab_native_ic
                                                 else "legacy_independent_area_difference"),
                        "metric_inner_surface_mode": ("native_cartilage_iC_rebased_correspondence" if compat_native_ic_map
                                                       else "matlab_native_cartilage_iC" if matlab_native_ic
                                                       else "rebased_compatibility_iC"),
                        "cAB_mm2": cabv,
                        "tAB_mm2": tabv,
                        "FCL_percent": fcl_pct,
                        "inner_vertices": int(len(iv)),
                        "rebased_iC_vertices_same_scb_region_diagnostic": rb_n,
                        "rebased_iC_area_same_scb_region_mm2_diagnostic": rb_area,
                        "scb_vertices": int(len(sv)),
                        "scb_not_inner_nzero": int(nzero_dbg),
                        "inner_not_scb": int(inner_not_in_scb),
                        "zero_fraction_of_scb_vertices": float(nzero_dbg / len(sv)) if len(sv) else float("nan"),
                        "inner_thickness_zero_hits": int(np.count_nonzero(vals_np == 0)),
                        "inner_thickness_raw_mean_mm": float(vals_np.mean()) if len(vals_np) else float("nan"),
                        "inner_thickness_positive_mean_mm": float(pos.mean()) if len(pos) else float("nan"),
                        "final_zero_padded_mean_mm": final_dbg,
                        "zero_padded_denominator": int(denom_dbg),
                    })

        # Surface -> volume parcellation: create labels on iC in ascending ROI order,
        # then cal_mapSeg2Mask nearest-neighbour mapping within this cartilage mask.
        ilab = roi_vertex_labels(imasks, len(inner.vertices_sub))
        mapped = map_surface_labels_to_cartilage_gpu(st["cart"], inner.vertices_sub, ilab, device)
        volume_labelmap[mapped > 0] = mapped[mapped > 0]

    # v12 diagnostic only: localize reconstructed-tAB expansion by ROI at each
    # reconstruction stage.  Final scB ROI identities are transferred to the full
    # subject-bone mesh by nearest vertex, then the same OR-face area convention is
    # applied to seed / connectivity-hole / curve-fill / closing faces.  This does
    # not modify cAB, tAB, thickness, volume, or the returned label map.
    if debug_surface_data is not None and stage_roi_debug:
        roi_rows_stage = debug_surface_data.setdefault("rois", [])
        for key, st in states.items():
            dbg = st.get("scb_debug") or {}
            bv = dbg.get("_bone_mesh_vertices_sub")
            bf = dbg.get("_bone_mesh_faces")
            if bv is None or bf is None or st["scb"].empty:
                continue
            full_bm = Mesh(np.asarray(bv, dtype=np.float64), np.asarray(bf, dtype=np.int64))
            smasks = st.get("scb_masks", {})
            full_roi_masks = transfer_roi_masks_nn(st["scb"], full_bm, smasks, device)
            rb_faces_full = np.asarray(dbg.get("_rebased_faces_full", np.empty((0, 3), np.int64)), dtype=np.int64)
            rb_keys = (_row_keys_int(np.sort(rb_faces_full, axis=1)) if len(rb_faces_full)
                       else np.empty((0,), dtype=np.dtype((np.void, 24))))
            prior_miss_faces = np.asarray(dbg.get("_prior_miss_faces_full", np.empty((0, 3), np.int64)), dtype=np.int64)
            row_by_roi = {int(x.get("roi_index", -1)): x for x in roi_rows_stage if x.get("compartment") == key}

            def _stage_region_stats(stage_faces: np.ndarray, vmask: np.ndarray) -> tuple[float, float, float]:
                sf = np.asarray(stage_faces, dtype=np.int64)
                if not len(sf) or not len(vmask):
                    return 0.0, 0.0, 0.0
                rf = sf[np.any(vmask[sf], axis=1)]
                if not len(rf):
                    return 0.0, 0.0, 0.0
                area = _faces_area_numpy(full_bm, rf, vox)
                if len(rb_keys):
                    rk = _row_keys_int(np.sort(rf, axis=1))
                    cov = np.isin(rk, rb_keys)
                    cov_area = _faces_area_numpy(full_bm, rf[cov], vox) if np.any(cov) else 0.0
                else:
                    cov_area = 0.0
                den_area = max(float(area - cov_area), 0.0)
                return float(area), float(cov_area), float(den_area)

            for r, vmask0 in full_roi_masks.items():
                row = row_by_roi.get(int(r))
                if row is None:
                    continue
                vmask = np.asarray(vmask0, dtype=bool)
                stage_vals = {}
                for stage_name in ("seed", "hole", "curve", "close"):
                    sf = np.asarray(dbg.get(f"_stage_faces_{stage_name}", np.empty((0, 3), np.int64)), dtype=np.int64)
                    a, ca, da = _stage_region_stats(sf, vmask)
                    stage_vals[stage_name] = a
                    row[f"{stage_name}_tAB_mm2"] = a
                    row[f"{stage_name}_covered_mm2_exact"] = ca
                    row[f"{stage_name}_denuded_mm2_exact"] = da
                    row[f"{stage_name}_denuded_percent_exact"] = (100.0 * da / a if a > 0 else float("nan"))
                row["hole_minus_seed_tAB_mm2"] = stage_vals["hole"] - stage_vals["seed"]
                row["curve_minus_hole_tAB_mm2"] = stage_vals["curve"] - stage_vals["hole"]
                row["close_minus_curve_tAB_mm2"] = stage_vals["close"] - stage_vals["curve"]
                if len(prior_miss_faces):
                    pf = prior_miss_faces[np.any(vmask[prior_miss_faces], axis=1)]
                    row["prior_miss_area_mm2_stage_roi"] = _faces_area_numpy(full_bm, pf, vox) if len(pf) else 0.0
                else:
                    row["prior_miss_area_mm2_stage_roi"] = 0.0

    if debug_surface_data is not None:
        comp_rows = debug_surface_data.setdefault("compartments", [])
        for key, st in states.items():
            split_inner = st["split_inner"]
            inner = st["inner"]
            rebased_inner = st.get("rebased_inner", inner)
            scb = st["scb"]
            row = {
                "compartment": key,
                "cart_voxels": int(np.count_nonzero(st["cart"])),
                "split_inner_vertices": int(len(split_inner.vertices_sub)),
                "split_inner_faces": int(len(split_inner.faces)),
                "split_inner_area_mm2": _mesh_area_numpy(split_inner, vox),
                "rebased_inner_vertices": int(len(rebased_inner.vertices_sub)),
                "rebased_inner_faces": int(len(rebased_inner.faces)),
                "rebased_inner_area_mm2": _mesh_area_numpy(rebased_inner, vox),
                "metric_inner_vertices": int(len(inner.vertices_sub)),
                "metric_inner_faces": int(len(inner.faces)),
                "metric_inner_area_mm2": _mesh_area_numpy(inner, vox),
                "metric_inner_surface_mode": ("native_cartilage_iC_rebased_correspondence" if compat_native_ic_map
                                               else "matlab_native_cartilage_iC" if matlab_native_ic
                                               else "rebased_compatibility_iC"),
                "final_scb_vertices": int(len(scb.vertices_sub)),
                "final_scb_faces": int(len(scb.faces)),
                "final_scb_area_mm2": _mesh_area_numpy(scb, vox),
            }
            if paper_fcl_geometry and not matlab_native_ic:
                src_idx_dbg = _target_vertex_source_indices(scb, inner)
                matched_src = np.unique(src_idx_dbg[src_idx_dbg >= 0]) if len(src_idx_dbg) else np.empty((0,), dtype=np.int64)
                cov_faces_dbg = _target_face_covered_by_mesh(scb, inner)
                areas_dbg = _mesh_face_areas_numpy(scb, vox)
                row.update({
                    "metric_geometry_mode": "paper_tAB_set_difference",
                    "scb_vertices_with_observed_inner": int(np.count_nonzero(src_idx_dbg >= 0)),
                    "scb_vertices_without_observed_inner": int(np.count_nonzero(src_idx_dbg < 0)),
                    "observed_inner_vertices_outside_scb": int(max(len(inner.vertices_sub) - len(matched_src), 0)),
                    "scb_faces_covered_by_observed_inner": int(np.count_nonzero(cov_faces_dbg)),
                    "scb_faces_denuded": int(len(cov_faces_dbg) - np.count_nonzero(cov_faces_dbg)),
                    "scb_covered_area_mm2_exact": float(areas_dbg[cov_faces_dbg].sum()) if len(areas_dbg) else 0.0,
                    "scb_denuded_area_mm2_exact": float(areas_dbg[~cov_faces_dbg].sum()) if len(areas_dbg) else 0.0,
                })
            else:
                row["metric_geometry_mode"] = ("v11_native_iC_via_rebased_correspondence" if compat_native_ic_map
                                               else "matlab_native_iC_independent_cAB_tAB" if matlab_native_ic
                                               else "legacy_independent_area_difference")
                if matlab_native_ic:
                    # Diagnostic: how much the compatibility rebase changed the
                    # cartilage-side surface before it was previously used as iC.
                    row["native_vs_rebased_area_delta_mm2"] = float(
                        _mesh_area_numpy(split_inner, vox) - _mesh_area_numpy(rebased_inner, vox)
                    )
            if st.get("split_debug"):
                row.update(st["split_debug"])
            if st.get("scb_debug"):
                row.update({k: v for k, v in st["scb_debug"].items() if not str(k).startswith("_")})
            comp_rows.append(row)

    # CartiMorph reports FCL/dABp in percent.  The earlier Python port returned
    # the raw 0..1 fraction; keep that convention only as an explicit legacy mode.
    fcl = regional_fcl_gpu(cAB, tAB, legacy_fcl_fraction)

    # Physical cartilage volume is voxel count times voxel volume (sx*sy*sz).
    # The old port preserved a Toolbox-source `norm(voxSize)` expression, which
    # produces an ~10x scale error on anisotropic OAI DESS data and contradicts
    # the method definition in the CartiMorph paper.
    vols = regional_volumes_gpu(volume_labelmap, vox, device, legacy_matlab_volume_norm)

    # v16-final reporting sidecars. The morphology above is frozen; this block
    # only exposes quantities already computed on the final v16 geometry.
    if final_export_data is not None:
        fcl_np = fcl.detach().cpu().numpy()
        cab_np = cAB.detach().cpu().numpy()
        tab_np = tAB.detach().cpu().numpy()
        dAB_np = np.maximum(tab_np - cab_np, 0.0)
        final_export_data["roi_areas"] = [
            {
                "ROI": ROI_NAMES[i],
                "tAB_mm2": float(tab_np[i]),
                "cAB_mm2": float(cab_np[i]),
                "dAB_mm2": float(dAB_np[i]),
                "FCL_percent": float(fcl_np[i]),
            }
            for i in range(20)
        ]

        # Chondrometrics-compatible aggregate domains. Union masks are used
        # directly on the final surface, so face-OR boundary triangles are
        # counted once rather than summed across overlapping ROI outputs.
        comp_specs = {
            "MTC": ("mTC", [11, 12, 13, 14, 15]),
            "cMFC": ("FC", [2, 3, 4]),
            "LTC": ("lTC", [16, 17, 18, 19, 20]),
            "cLFC": ("FC", [7, 8, 9]),
        }
        comp_rows = {}
        voxel_scale = float(np.linalg.norm(vox) if legacy_matlab_volume_norm else np.prod(vox))
        for comp, (state_name, roi_ids) in comp_specs.items():
            st = states[state_name]
            scb: Mesh = st["scb"]
            smasks = st["scb_masks"]
            union_v = np.zeros(len(scb.vertices_sub), dtype=bool)
            for rid in roi_ids:
                mm = np.asarray(smasks.get(rid, np.zeros(len(union_v), dtype=bool)), dtype=bool)
                if len(mm) == len(union_v):
                    union_v |= mm

            if len(scb.faces) and np.any(union_v):
                rf = np.any(union_v[np.asarray(scb.faces, dtype=np.int64)], axis=1)
                areas = _mesh_face_areas_numpy(scb, vox)
                tab_c = float(areas[rf].sum())
                covf = np.asarray(
                    st.get("_covered_faces_final", np.zeros(len(scb.faces), dtype=bool)),
                    dtype=bool,
                )
                cab_c = float(areas[rf & covf].sum())
            else:
                tab_c = 0.0
                cab_c = 0.0

            dab_c = max(tab_c - cab_c, 0.0)
            fcl_c = (100.0 * dab_c / tab_c) if tab_c > 0 else float("nan")
            scb_th_c = st.get("_scb_th_final")
            if scb_th_c is not None and np.any(union_v):
                vals = scb_th_c[
                    torch.as_tensor(union_v, dtype=torch.bool, device=device)
                ]
                mean_th_c = float(vals.mean().detach().cpu()) if vals.numel() else 0.0
            else:
                mean_th_c = 0.0

            vol_mask = np.isin(volume_labelmap, np.asarray(roi_ids, dtype=np.uint8))
            vol_c = float(np.count_nonzero(vol_mask) * voxel_scale)

            comp_rows[comp] = {
                "FCL": fcl_c,
                "Mean Thickness": mean_th_c,
                "Surface Area": cab_c,
                "Volume": vol_c,
                "tAB_mm2": tab_c,
                "cAB_mm2": cab_c,
                "dAB_mm2": dab_c,
            }
        final_export_data["compartments"] = comp_rows

    result = torch.stack([fcl, mean_th, cAB, vols], dim=0).cpu().numpy()
    _prof_end("30_regional_metrics_and_volume_map", _t_metrics)
    if profile:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total = time.perf_counter() - total_t0
        print("PROFILE_SECONDS")
        for name in sorted(profile_times):
            print(f"  {name}: {profile_times[name]:.6f}")
        print(f"  99_total: {total:.6f}")
    return result, volume_labelmap


def write_csv(path: str | os.PathLike, arr: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # MATLAB table row-dimension name is "Row" when WriteRowNames=true.
        w.writerow(["Row"] + ROI_NAMES)
        for name, row in zip(ROW_NAMES, arr):
            w.writerow([name] + ["NaN" if np.isnan(x) else f"{float(x):.17g}" for x in row])


def write_final_compartments_csv(
    path: str | os.PathLike,
    comp_data: Mapping[str, Mapping[str, float]],
) -> None:
    comps = ["MTC", "cMFC", "LTC", "cLFC"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Row"] + comps)
        for metric in ROW_NAMES:
            vals = [float(comp_data[c][metric]) for c in comps]
            w.writerow(
                [metric]
                + ["NaN" if np.isnan(x) else f"{x:.17g}" for x in vals]
            )


def write_fcl_areas_csv(
    path: str | os.PathLike,
    rows: Sequence[Mapping[str, object]],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["ROI", "tAB_mm2", "cAB_mm2", "dAB_mm2", "FCL_percent"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _write_dict_rows_csv(path: str | os.PathLike, rows: Sequence[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CartiMorph 20-ROI GPU morphometric analysis")
    ap.add_argument("--seg", required=True, help="subject segmentation (.nii/.nii.gz/.npy)")
    ap.add_argument("--registration", required=True, help="warped template tissue seg (preferred), 20-ROI atlas, or deformation field")
    ap.add_argument("--output", required=True, help="output MorphQuant CSV")
    ap.add_argument("--template-seg", help="template tissue segmentation; required when --registration is a deformation field")
    ap.add_argument("--template-atlas", help="deprecated alias: template 20-ROI atlas; collapsed to tissue prior before morphology")
    ap.add_argument("--knee-side", required=True, choices=["left", "right", "Left", "Right", "L", "R"])
    ap.add_argument("--vox-size", nargs=3, type=float, metavar=("SX","SY","SZ"), help="override voxel size")
    ap.add_argument("--background-label", type=int, default=0)
    ap.add_argument("--femur-label", type=int, default=1)
    ap.add_argument("--fc-label", type=int, default=2)
    ap.add_argument("--tibia-label", type=int, default=3)
    ap.add_argument("--mtc-label", type=int, default=4)
    ap.add_argument("--ltc-label", type=int, default=5)
    ap.add_argument("--cc-percentage", type=float, default=0.6)
    ap.add_argument("--max-thickness", type=float, default=7.0)
    ap.add_argument(
        "--legacy-matlab-volume-norm", action="store_true",
        help="reproduce legacy Toolbox norm(voxSize) volume scaling; default uses sx*sy*sz mm^3",
    )
    ap.add_argument(
        "--legacy-fcl-fraction", action="store_true",
        help="return FCL as the old Python 0..1 fraction; default is CartiMorph/dABp percent (0..100)",
    )
    ap.add_argument("--allow-cpu-fallback", action="store_true", help="debug only; default requires CUDA")
    ap.add_argument("--profile", action="store_true", help="print synchronized per-stage wall times; no numerical changes")
    ap.add_argument(
        "--debug-surfaces", action="store_true",
        help="write diagnostic cAB/tAB/inner/scB tables next to MorphQuant.csv; no numerical changes",
    )
    ap.add_argument(
        "--physical-scb-mapping", action="store_true",
        help="rejected v5 A/B option retained for reproducibility only; official-parity default is raw voxel-index scB mapping",
    )
    ap.add_argument(
        "--cart-surface-finetune", action="store_true",
        help="candidate v7: use public Eq.11/Eq.12 OR face extraction, then CartiMorph inner closing / restricted outer dilation",
    )
    ap.add_argument(
        "--surface-closing-iterations", type=int, default=4,
        help="surface dilation and erosion iterations for candidate cartilage inner-surface closing (default 4)",
    )
    ap.add_argument(
        "--paper-fcl-geometry", action="store_true",
        help="candidate v8: follow paper Eq.21-22 by keeping reconstructed tAB independent; do not force observed iC faces into tAB",
    )
    ap.add_argument(
        "--balanced-scb-closing", action="store_true",
        help="candidate v9: use equal dilation/erosion iterations for final reconstructed-tAB surface closing instead of legacy erosion=dilation+4",
    )
    ap.add_argument(
        "--matlab-native-ic", action="store_true",
        help="candidate v10: keep native cartilage mesh_iC for thickness/cAB/parcellation; use bone rebase only internally for reconstructed tAB",
    )
    ap.add_argument(
        "--compat-native-ic-map", action="store_true",
        help="candidate v11: transfer v9 rebased-iC ROI/coverage correspondence back to native iC metric geometry",
    )
    ap.add_argument(
        "--interface-scb-seed", action="store_true",
        help=("v13 candidate: construct Eq.22 seed in bone-surface face space as "
              "rebased observed iC intersect warped-prior footprint, instead of projecting "
              "all overlap volume voxels to the globally nearest bone surface")
    )
    ap.add_argument(
        "--overlap-interface-scb-seed", action="store_true",
        help=("v14 candidate: construct Eq.22 seed from only the bone-facing interface voxels "
              "of subject-cartilage intersect warped-prior, then use the legacy bone-surface "
              "mapping/closing path; avoids projecting the full overlap cartilage volume")
    )
    ap.add_argument(
        "--constrained-interface-scb-seed", action="store_true",
        help=("v15 candidate: map v14 overlap-interface voxels only to the rebased observed-iC "
              "bone domain, retain voxel closing/component filtering, then clamp seed faces to "
              "observed iC; prevents folded-femur cross-sheet nearest-neighbor mapping")
    )
    ap.add_argument(
        "--fc-contact-augment-inner", action="store_true",
        help=("v16 candidate: FC only, augment the public Eq.11 OR inner-surface seed with "
              "grown-bone contact faces missed by skimage half-voxel/rounding; mTC/lTC remain v7")
    )
    ap.add_argument(
        "--stage-roi-debug", action="store_true",
        help="v12 diagnostic only: report seed/hole/curve/close reconstructed-tAB area and exact-face denuded area by final ROI; requires --debug-surfaces",
    )
    ap.add_argument("--save-subregions-npy", help="optional 20-label subject cartilage parcellation .npy")
    ap.add_argument("--save-meta-json", help="optional metadata JSON")
    args = ap.parse_args(argv)

    # CartiMorph-v16-final is frozen. Older A/B flags remain parseable for
    # backwards-compatible error messages, but this release always executes
    # the validated v16 configuration.
    if args.physical_scb_mapping or args.matlab_native_ic or args.compat_native_ic_map \
            or args.interface_scb_seed or args.overlap_interface_scb_seed:
        ap.error("CartiMorph-v16-final does not allow older A/B morphology modes")
    args.cart_surface_finetune = True
    args.paper_fcl_geometry = True
    args.balanced_scb_closing = True
    args.constrained_interface_scb_seed = True
    args.fc_contact_augment_inner = True
    args.physical_scb_mapping = False
    args.matlab_native_ic = False
    args.compat_native_ic_map = False
    args.interface_scb_seed = False
    args.overlap_interface_scb_seed = False

    if args.paper_fcl_geometry and not args.cart_surface_finetune:
        ap.error("--paper-fcl-geometry requires --cart-surface-finetune (v8 builds on the v7 Eq.11 OR surface split)")
    if args.balanced_scb_closing and not (args.paper_fcl_geometry or args.matlab_native_ic or args.compat_native_ic_map):
        ap.error("--balanced-scb-closing requires --paper-fcl-geometry, --matlab-native-ic, or --compat-native-ic-map")
    if args.matlab_native_ic and not args.cart_surface_finetune:
        ap.error("--matlab-native-ic requires --cart-surface-finetune (v10 builds on the v7 Eq.11 OR surface split)")
    if args.matlab_native_ic and args.paper_fcl_geometry:
        ap.error("--matlab-native-ic and --paper-fcl-geometry are alternative metric-geometry A/B modes; do not enable both")
    if args.compat_native_ic_map and not args.cart_surface_finetune:
        ap.error("--compat-native-ic-map requires --cart-surface-finetune")
    if args.compat_native_ic_map and (args.matlab_native_ic or args.paper_fcl_geometry):
        ap.error("--compat-native-ic-map is an alternative v11 A/B mode; do not combine it with v8/v10")
    if args.stage_roi_debug and not args.debug_surfaces:
        ap.error("--stage-roi-debug requires --debug-surfaces")
    if args.interface_scb_seed and not (args.cart_surface_finetune and args.paper_fcl_geometry):
        ap.error("--interface-scb-seed requires --cart-surface-finetune and --paper-fcl-geometry")
    if args.overlap_interface_scb_seed and not (args.cart_surface_finetune and args.paper_fcl_geometry):
        ap.error("--overlap-interface-scb-seed requires --cart-surface-finetune and --paper-fcl-geometry")
    if args.constrained_interface_scb_seed and not (args.cart_surface_finetune and args.paper_fcl_geometry):
        ap.error("--constrained-interface-scb-seed requires --cart-surface-finetune and --paper-fcl-geometry")
    if args.fc_contact_augment_inner and not args.cart_surface_finetune:
        ap.error("--fc-contact-augment-inner requires --cart-surface-finetune")
    if sum(bool(x) for x in (args.interface_scb_seed, args.overlap_interface_scb_seed, args.constrained_interface_scb_seed)) > 1:
        ap.error("--interface-scb-seed, --overlap-interface-scb-seed and --constrained-interface-scb-seed are alternative seed A/B modes")

    seg, vox0 = load_volume(args.seg)
    reg, _ = load_volume(args.registration)
    seg = np.asarray(seg).squeeze()
    reg = np.asarray(reg)
    if seg.ndim != 3:
        raise ValueError(f"Segmentation must be 3D, got {seg.shape}")
    vox = np.asarray(args.vox_size if args.vox_size else vox0, dtype=np.float64)
    if vox is None or np.size(vox) != 3:
        raise ValueError("Voxel size unavailable; pass --vox-size SX SY SZ")
    labels = LabelConfig(args.background_label, args.femur_label, args.fc_label,
                         args.tibia_label, args.mtc_label, args.ltc_label)
    device = choose_device(require_cuda=True, allow_cpu_fallback=args.allow_cpu_fallback)
    template = None
    template_path = args.template_seg or args.template_atlas
    if template_path:
        template, _ = load_volume(template_path)
        template = np.asarray(template).squeeze()

    debug_surface_data = {} if args.debug_surfaces else None
    final_export_data: dict = {}
    result, subregions = morphometrics_gpu(
        seg.astype(np.int16), reg, vox, labels, args.knee_side, device,
        template_seg=template, cc_percentage=args.cc_percentage, depth=args.max_thickness,
        legacy_matlab_volume_norm=args.legacy_matlab_volume_norm,
        legacy_fcl_fraction=args.legacy_fcl_fraction,
        profile=args.profile,
        debug_surface_data=debug_surface_data,
        physical_scb_mapping=args.physical_scb_mapping,
        cart_surface_finetune=args.cart_surface_finetune,
        surface_closing_iterations=args.surface_closing_iterations,
        paper_fcl_geometry=args.paper_fcl_geometry,
        balanced_scb_closing=args.balanced_scb_closing,
        matlab_native_ic=args.matlab_native_ic,
        compat_native_ic_map=args.compat_native_ic_map,
        stage_roi_debug=args.stage_roi_debug,
        interface_scb_seed=args.interface_scb_seed,
        overlap_interface_scb_seed=args.overlap_interface_scb_seed,
        constrained_interface_scb_seed=args.constrained_interface_scb_seed,
        fc_contact_augment_inner=args.fc_contact_augment_inner,
        final_export_data=final_export_data,
    )
    write_csv(args.output, result)

    out_main = Path(args.output)
    comp_out = out_main.parent / "MorphQuant_Compartments.csv"
    fcl_area_out = out_main.parent / "FCL_Areas.csv"
    if final_export_data.get("compartments"):
        write_final_compartments_csv(
            comp_out, final_export_data["compartments"]
        )
    if final_export_data.get("roi_areas"):
        write_fcl_areas_csv(
            fcl_area_out, final_export_data["roi_areas"]
        )
    if debug_surface_data is not None:
        outp = Path(args.output)
        stem = outp.with_suffix("")
        comp_path = stem.parent / f"{stem.name}_surface_debug_compartments.csv"
        roi_path = stem.parent / f"{stem.name}_surface_debug_rois.csv"
        json_path = stem.parent / f"{stem.name}_surface_debug.json"
        _write_dict_rows_csv(comp_path, debug_surface_data.get("compartments", []))
        _write_dict_rows_csv(roi_path, debug_surface_data.get("rois", []))
        json_path.write_text(json.dumps(debug_surface_data, indent=2, allow_nan=True), encoding="utf-8")
        print("SURFACE_DEBUG_COMPARTMENTS")
        for row in debug_surface_data.get("compartments", []):
            prefix = f"  {row.get('compartment')}: "
            if "eq11_or_inner_area_mm2" in row:
                prefix += (
                    f"wholeA={row.get('whole_cartilage_area_mm2', float('nan')):.3f} "
                    f"eq11orA={row.get('eq11_or_inner_area_mm2', float('nan')):.3f} "
                    f"contactA={row.get('v6_contact_inner_area_mm2_diagnostic', float('nan')):.3f} "
                    f"fineInA={row.get('fine_inner_area_mm2', float('nan')):.3f} "
                )
            print(
                prefix
                + f"splitA={row.get('split_inner_area_mm2', float('nan')):.3f} "
                f"rebasedA={row.get('rebased_inner_area_mm2', float('nan')):.3f} "
                f"seedA={row.get('seed_surface_area_mm2', float('nan')):.3f} "
                f"holeA={row.get('after_hole_fill_area_mm2', float('nan')):.3f} "
                f"curveA={row.get('after_curve_fill_area_mm2', float('nan')):.3f} "
                f"closeA={row.get('after_surface_closing_area_mm2', float('nan')):.3f} "
                f"preUnionA={row.get('pre_observed_union_area_mm2', float('nan')):.3f} "
                f"scbA={row.get('final_scb_area_mm2', float('nan')):.3f}"
                + ((f" coveredA={row.get('scb_covered_area_mm2_exact', float('nan')):.3f}"
                    f" denudedA={row.get('scb_denuded_area_mm2_exact', float('nan')):.3f}")
                   if row.get('metric_geometry_mode') == 'paper_tAB_set_difference' else "")
                + ((f" priorOnlyA={row.get('prior_only_mapped_area_mm2', float('nan')):.3f}"
                    f" priorMissA={row.get('prior_only_area_outside_observed_inner_mm2', float('nan')):.3f}"
                    f" priorRecA={row.get('prior_only_uncovered_area_recovered_in_final_scb_mm2', float('nan')):.3f}"
                    f" priorRecFrac={row.get('prior_only_uncovered_area_recovery_fraction', float('nan')):.3f}"
                    f" closeIt={row.get('scb_closing_dilation_iterations', -1)}/{row.get('scb_closing_erosion_iterations', -1)}")
                   if 'prior_only_mapped_area_mm2' in row else "")
                + ((f" seedSupportA={row.get('interface_seed_supported_area_mm2', float('nan')):.3f}"
                    f" seedSupportFrac={row.get('interface_seed_rebased_support_fraction_area', float('nan')):.3f}")
                   if 'interface_seed_supported_area_mm2' in row else "")
                + ((f" seedObsA={row.get('seed_area_supported_by_observed_iC_mm2', float('nan')):.3f}"
                    f" seedObsFrac={row.get('seed_observed_support_fraction_area', float('nan')):.3f}"
                    + (f" seedRbFrac={row.get('seed_rebased_coverage_fraction_area', float('nan')):.3f}"
                       if 'seed_rebased_coverage_fraction_area' in row else ""))
                   if 'seed_area_supported_by_observed_iC_mm2' in row else "")
            )
        print("SURFACE_DEBUG_ROIS")
        for row in debug_surface_data.get("rois", []):
            print(
                f"  {row['roi']}: cAB={row['cAB_mm2']:.3f} tAB={row['tAB_mm2']:.3f} "
                f"FCL={row['FCL_percent']:.2f}% innerV={row['inner_vertices']} scbV={row['scb_vertices']} "
                f"nzero={row['scb_not_inner_nzero']} innerNotScb={row['inner_not_scb']} "
                f"zeroFrac={row['zero_fraction_of_scb_vertices']:.3f} "
                f"posTh={row['inner_thickness_positive_mean_mm']:.3f} "
                f"finalTh={row['final_zero_padded_mean_mm']:.3f}"
                + ((f" rbV={row.get('rebased_iC_vertices_same_scb_region_diagnostic', 0)}"
                    f" rbA={row.get('rebased_iC_area_same_scb_region_mm2_diagnostic', 0.0):.3f}")
                   if row.get('metric_geometry_mode') in {'matlab_native_iC_independent_cAB_tAB','v11_native_iC_via_rebased_correspondence'} else "")
            )
        if args.stage_roi_debug:
            print("SURFACE_DEBUG_STAGE_ROIS")
            for row in debug_surface_data.get("rois", []):
                if "seed_tAB_mm2" not in row:
                    continue
                print(
                    f"  {row['roi']}: "
                    f"seed={row['seed_tAB_mm2']:.3f} "
                    f"hole={row['hole_tAB_mm2']:.3f} "
                    f"curve={row['curve_tAB_mm2']:.3f} "
                    f"close={row['close_tAB_mm2']:.3f} "
                    f"dSeed={row['seed_denuded_mm2_exact']:.3f} "
                    f"dHole={row['hole_denuded_mm2_exact']:.3f} "
                    f"dCurve={row['curve_denuded_mm2_exact']:.3f} "
                    f"dClose={row['close_denuded_mm2_exact']:.3f} "
                    f"priorMiss={row.get('prior_miss_area_mm2_stage_roi', float('nan')):.3f}"
                )
        print(f"Surface debug CSV: {comp_path}")
        print(f"Surface debug CSV: {roi_path}")
        print(f"Surface debug JSON: {json_path}")
    if args.save_subregions_npy:
        Path(args.save_subregions_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_subregions_npy, subregions)
    if args.save_meta_json:
        meta = {
            "device": str(device), "roi_names": ROI_NAMES, "row_names": ROW_NAMES,
            "voxel_size": vox.tolist(),
            "volume_scale_mode": "legacy_matlab_norm" if args.legacy_matlab_volume_norm else "physical_voxel_volume",
            "volume_scale_factor": float(np.linalg.norm(vox) if args.legacy_matlab_volume_norm else np.prod(vox)),
            "voxel_volume_mm3": float(np.prod(vox)),
            "fcl_scale_mode": "legacy_fraction_0_1" if args.legacy_fcl_fraction else "percent_0_100",
            "release": "CartiMorph-v16-final",
            "performance_mode": (
                "cartimorph_v16_final" if (
                    args.cart_surface_finetune
                    and args.paper_fcl_geometry
                    and args.balanced_scb_closing
                    and args.constrained_interface_scb_seed
                    and args.fc_contact_augment_inner
                    and not args.physical_scb_mapping
                    and not args.matlab_native_ic
                    and not args.compat_native_ic_map
                    and not args.interface_scb_seed
                    and not args.overlap_interface_scb_seed
                )
                else "exact_optimized_fp64_v16_fc_contact_augmented_inner_candidate" if args.fc_contact_augment_inner
                else "exact_optimized_fp64_v15_constrained_interface_seed_candidate" if args.constrained_interface_scb_seed
                else "exact_optimized_fp64_v14_overlap_interface_seed_candidate" if args.overlap_interface_scb_seed
                else "exact_optimized_fp64_v13_interface_seed_candidate" if args.interface_scb_seed
                else "exact_optimized_fp64_v11_native_iC_rebased_correspondence_candidate" if args.compat_native_ic_map
                else "exact_optimized_fp64_v10_matlab_native_iC_candidate" if args.matlab_native_ic
                else "exact_optimized_fp64_v9_balanced_scb_closing_candidate" if args.balanced_scb_closing
                else "exact_optimized_fp64_v8_paper_fcl_candidate" if args.paper_fcl_geometry
                else ("exact_optimized_fp64_v7_eq11or_surface_candidate" if args.cart_surface_finetune
                      else ("exact_optimized_fp64_v5_physical_scb_candidate" if args.physical_scb_mapping else "exact_optimized_fp64_v3_bbox"))
            ),
            "scb_nn_mapping_space": "physical_mm" if args.physical_scb_mapping else "legacy_voxel_index",
            "cart_surface_split_mode": ("v16_fc_eq11or_plus_contact_closing_restricted_outer" if args.fc_contact_augment_inner
                                        else "v7_eq11or_closing_restricted_outer" if args.cart_surface_finetune
                                        else "legacy_and_complement"),
            "surface_closing_iterations": int(args.surface_closing_iterations),
            "metric_inner_surface_mode": ("native_cartilage_iC_rebased_correspondence" if args.compat_native_ic_map
                                          else "matlab_native_cartilage_iC" if args.matlab_native_ic
                                          else "rebased_compatibility_iC"),
            "fcl_geometry_mode": (
                "independent_native_cAB_tAB_rebased_correspondence" if args.compat_native_ic_map
                else "matlab_independent_cAB_tAB_difference_no_observed_union" if args.matlab_native_ic
                else "paper_reconstructed_tAB_no_observed_union" if args.paper_fcl_geometry
                else "legacy_force_iC_into_tAB"
            ),
            "scb_surface_closing_mode": "balanced_dilation_erosion" if args.balanced_scb_closing else "legacy_erosion_plus4",
            "scb_seed_mode": ("v15_constrained_overlap_interface_observed_iC" if args.constrained_interface_scb_seed
                              else "v14_overlap_bone_interface_voxels" if args.overlap_interface_scb_seed
                              else "v13_observed_iC_intersect_prior_surface" if args.interface_scb_seed
                              else "legacy_overlap_voxels_to_nearest_bone"),
            "surface_debug_enabled": bool(args.debug_surfaces),
            "stage_roi_debug_enabled": bool(args.stage_roi_debug),
            "fc_contact_augment_inner_enabled": bool(args.fc_contact_augment_inner),
            "diagnostic_mode": "v12_reconstruction_stage_roi" if args.stage_roi_debug else "none",
            "metric_definitions": {
                "FCL": "100 * dAB / tAB, percent",
                "Mean Thickness": "mean cartilage thickness over final tAB vertices; denuded/uncovered tAB vertices receive zero, mm",
                "Surface Area": "cartilage-covered subchondral bone area cAB, mm^2",
                "Volume": "cartilage voxel count * physical voxel volume, mm^3",
            },
            "final_output_files": {
                "regional_20roi": str(Path(args.output).name),
                "chondrometrics_compartments": "MorphQuant_Compartments.csv",
                "fcl_area_audit": "FCL_Areas.csv",
            },
            "labels": labels.__dict__, "knee_side": args.knee_side,
        }
        Path(args.save_meta_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")
    print(f"Saved: {Path(args.output).parent / 'MorphQuant_Compartments.csv'}")
    print(f"Saved: {Path(args.output).parent / 'FCL_Areas.csv'}")
    print(f"Device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
