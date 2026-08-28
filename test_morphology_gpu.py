import numpy as np
import torch

from morphology_gpu import (
    ROI_NAMES, ROW_NAMES, LabelConfig, Mesh, matlab_round,
    volume_parcellation_fc, volume_parcellation_tc,
    atlas20_to_tissue, resolve_warped_tissue,
    cartilage_thickness_gpu, _face_components, preprocess_binary, boundary_mesh,
    split_cartilage_mesh, build_total_subchondral_mesh, triangle_area_gpu,
    regional_volumes_gpu, regional_fcl_gpu,
)


def test_matlab_round():
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    assert np.array_equal(matlab_round(x), np.array([-3, -2, -1, 1, 2, 3]))


def test_roi_order():
    assert len(ROI_NAMES) == 20
    assert ROW_NAMES == ["FCL", "Mean Thickness", "Surface Area", "Volume"]


def test_parcellation_shapes():
    s = (24, 32, 20)
    fc = np.zeros(s, bool); fc[4:20, 5:25, 8:12] = True
    mt = np.zeros(s, bool); mt[5:11, 10:22, 5:8] = True
    lt = np.zeros(s, bool); lt[14:20, 10:22, 5:8] = True
    a = volume_parcellation_fc(fc, "right", 0.6)
    b = volume_parcellation_tc(mt, lt, "right", np.array([1., 1., 1.]))
    assert a.shape == s and b.shape == s
    assert a.max() <= 10 and b.max() <= 20


def test_mpt_components_require_shared_edge():
    # Faces 0 and 1 touch only at vertex 0 -> separate in MPT.
    # Faces 1 and 2 share edge (3,4) -> same component.
    f = np.array([[0, 1, 2], [0, 3, 4], [3, 4, 5]])
    comps = _face_components(f)
    assert sorted(len(c) for c in comps) == [1, 2]


def test_registration_resolution():
    atlas = np.zeros((4, 4, 4), np.uint8)
    atlas[0] = 1; atlas[1] = 11; atlas[2] = 16
    tissue = atlas20_to_tissue(atlas, LabelConfig())
    assert set(np.unique(tissue)) == {0, 2, 4, 5}
    flow = np.zeros((4, 4, 4, 3), np.float32)
    warped = resolve_warped_tissue(flow, LabelConfig(), torch.device("cpu"), tissue)
    assert np.array_equal(warped, tissue)


def test_thickness_gpu_kernel_on_parallel_planes():
    # With voxel size 1, surfaces are 2 voxels apart. MATLAB's ellipsoid
    # uncertainty contributes +2*1, so final ThicknessMap is exactly 4 here.
    n = 5
    vi = np.array([[i + 1, j + 1, 2.] for i in range(n) for j in range(n)], float)
    vo = np.array([[i + 1, j + 1, 4.] for i in range(n) for j in range(n)], float)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j; b = a + 1; c = (i + 1) * n + j; d = c + 1
            faces.extend([[a, b, d], [a, d, c]])
    faces = np.asarray(faces, dtype=np.int64)
    th = cartilage_thickness_gpu(Mesh(vi, faces), Mesh(vo, faces),
                                 np.ones(3), 7.0, 1000, torch.device("cpu"))
    assert torch.allclose(th, torch.full_like(th, 4.0), atol=1e-8, rtol=0)


def _mesh_area(mesh, vox):
    if mesh.empty:
        return 0.0
    v = torch.as_tensor(mesh.vertices_sub * np.asarray(vox)[None, :], dtype=torch.float64)
    f = torch.as_tensor(mesh.faces, dtype=torch.long)
    return float(triangle_area_gpu(v, f).sum())


def test_fcl_reconstruction_healthy_overlap_is_zero_and_defect_is_positive():
    """Regression for the old union/expanded-bone FCL bug.

    With a healthy warped prior identical to subject cartilage, tAB must collapse
    to observed cAB (FCL ~= 0). Removing a central full-thickness patch from the
    subject while keeping the healthy prior must produce positive FCL.
    """
    shape = (36, 44, 28)
    vox = np.array([0.365, 0.365, 0.7], dtype=float)
    bone = np.zeros(shape, bool); bone[5:31, 6:38, 11:20] = True
    healthy = np.zeros(shape, bool); healthy[7:29, 8:36, 9:11] = True
    n_remove = int(np.ceil(1.0 / np.prod(vox)))
    bone = preprocess_binary(bone, n_remove)

    def reconstruct(cart):
        cart = preprocess_binary(cart, n_remove)
        whole = boundary_mesh(cart)
        inner0, _ = split_cartilage_mesh(cart, bone, whole)
        inner, scb = build_total_subchondral_mesh(
            bone, cart, healthy, inner0, vox, torch.device("cpu"), "FemoralCartilage"
        )
        cAB = _mesh_area(inner, vox); tAB = _mesh_area(scb, vox)
        return 0.0 if tAB == 0 else max(0.0, (tAB - cAB) / tAB)

    healthy_fcl = reconstruct(healthy.copy())
    defect = healthy.copy(); defect[15:21, 19:25, 9:11] = False
    defect_fcl = reconstruct(defect)
    assert healthy_fcl < 1e-10
    assert defect_fcl > 0.01



def test_fcl_output_is_percentage_by_default():
    cAB = torch.tensor([80.0, 50.0, 0.0], dtype=torch.float64)
    tAB = torch.tensor([100.0, 100.0, 0.0], dtype=torch.float64)
    pct = regional_fcl_gpu(cAB, tAB).cpu().numpy()
    assert np.isclose(pct[0], 20.0)
    assert np.isclose(pct[1], 50.0)
    assert np.isnan(pct[2])
    frac = regional_fcl_gpu(cAB, tAB, True).cpu().numpy()
    assert np.isclose(frac[0], 0.2)
    assert np.isclose(frac[1], 0.5)

def test_regional_volume_uses_physical_voxel_product():
    lab = np.zeros((3, 4, 5), np.uint8)
    lab.flat[:7] = 1
    lab.flat[7:12] = 2
    vox = np.array([0.365, 0.365, 0.7], dtype=float)
    out = regional_volumes_gpu(lab, vox, torch.device("cpu")).cpu().numpy()
    vv = float(np.prod(vox))
    assert np.isclose(out[0], 7 * vv)
    assert np.isclose(out[1], 5 * vv)
    assert np.all(out[2:] == 0)
    legacy = regional_volumes_gpu(lab, vox, torch.device("cpu"), True).cpu().numpy()
    assert np.isclose(legacy[0], 7 * np.linalg.norm(vox))


def test_stage_roi_debug_does_not_change_scb_geometry():
    shape = (30, 36, 24)
    vox = np.array([0.365, 0.365, 0.7], dtype=float)
    bone = np.zeros(shape, bool); bone[4:26, 5:31, 10:18] = True
    cart = np.zeros(shape, bool); cart[6:24, 7:29, 8:10] = True
    prior = cart.copy()
    n_remove = int(np.ceil(1.0 / np.prod(vox)))
    bone = preprocess_binary(bone, n_remove)
    cart = preprocess_binary(cart, n_remove)
    whole = boundary_mesh(cart)
    inner0, _ = split_cartilage_mesh(cart, bone, whole)
    dbg0 = {}
    i0, s0 = build_total_subchondral_mesh(
        bone, cart, prior, inner0, vox, torch.device("cpu"), "FemoralCartilage",
        debug_info=dbg0, paper_fcl_geometry=True, balanced_scb_closing=True, stage_roi_debug=False
    )
    dbg1 = {}
    i1, s1 = build_total_subchondral_mesh(
        bone, cart, prior, inner0, vox, torch.device("cpu"), "FemoralCartilage",
        debug_info=dbg1, paper_fcl_geometry=True, balanced_scb_closing=True, stage_roi_debug=True
    )
    assert np.array_equal(i0.vertices_sub, i1.vertices_sub)
    assert np.array_equal(i0.faces, i1.faces)
    assert np.array_equal(s0.vertices_sub, s1.vertices_sub)
    assert np.array_equal(s0.faces, s1.faces)
    for k in ("_stage_faces_seed", "_stage_faces_hole", "_stage_faces_curve", "_stage_faces_close"):
        assert k in dbg1
        assert isinstance(dbg1[k], np.ndarray)



def test_interface_scb_seed_is_subset_of_observed_rebased_interface():
    """v13 invariant: Eq.22 seed cannot start outside observed iC coverage.

    The old source-voxel -> global bone NN mapper could create already-denuded
    seed faces on folded bone geometry.  v13 constructs the candidate seed as
    an exact face-space intersection of rebased observed iC and warped-prior
    support, so every seed face must be an observed-interface face.
    """
    shape = (30, 36, 24)
    vox = np.array([0.365, 0.365, 0.7], dtype=float)
    bone = np.zeros(shape, bool); bone[4:26, 5:31, 10:18] = True
    cart = np.zeros(shape, bool); cart[6:24, 7:29, 8:10] = True
    prior = cart.copy()
    n_remove = int(np.ceil(1.0 / np.prod(vox)))
    bone = preprocess_binary(bone, n_remove)
    cart = preprocess_binary(cart, n_remove)
    whole = boundary_mesh(cart)
    inner0, _ = split_cartilage_mesh(cart, bone, whole)
    dbg = {}
    build_total_subchondral_mesh(
        bone, cart, prior, inner0, vox, torch.device("cpu"), "FemoralCartilage",
        debug_info=dbg, paper_fcl_geometry=True, balanced_scb_closing=True,
        stage_roi_debug=True, interface_scb_seed=True,
    )
    seed = np.asarray(dbg["_stage_faces_seed"], dtype=np.int64)
    rebased = np.asarray(dbg["_rebased_faces_full"], dtype=np.int64)
    seed_keys = {tuple(x) for x in np.sort(seed, axis=1)}
    rebased_keys = {tuple(x) for x in np.sort(rebased, axis=1)}
    assert seed_keys.issubset(rebased_keys)
    assert dbg["scb_seed_mode"] == "v13_observed_iC_intersect_prior_surface"



def test_overlap_interface_scb_seed_uses_bone_facing_overlap_only():
    """v14: Eq.22 source is the overlap/bone interface, not all overlap voxels."""
    shape = (30, 36, 24)
    vox = np.array([0.365, 0.365, 0.7], dtype=float)
    bone = np.zeros(shape, bool); bone[4:26, 5:31, 10:18] = True
    cart = np.zeros(shape, bool); cart[6:24, 7:29, 8:10] = True
    prior = cart.copy()
    n_remove = int(np.ceil(1.0 / np.prod(vox)))
    bone = preprocess_binary(bone, n_remove)
    cart = preprocess_binary(cart, n_remove)
    whole = boundary_mesh(cart)
    inner0, _ = split_cartilage_mesh(cart, bone, whole)
    dbg = {}
    build_total_subchondral_mesh(
        bone, cart, prior, inner0, vox, torch.device("cpu"), "FemoralCartilage",
        debug_info=dbg, paper_fcl_geometry=True, balanced_scb_closing=True,
        stage_roi_debug=True, overlap_interface_scb_seed=True,
    )
    assert dbg["scb_seed_mode"] == "v14_overlap_bone_interface_voxels"
    assert 0 < dbg["overlap_interface_subs_count"] < dbg["cart_prior_overlap_voxels"]
    assert dbg["overlap_interface_mapped_bone_vertices"] > 0
    assert dbg["seed_surface_faces"] > 0
    # In this healthy parallel slab the interface-only source stays entirely on
    # the observed rebased cartilage/bone interface before defect filling.
    assert dbg["seed_observed_support_fraction_area"] == 1.0



def test_constrained_interface_scb_seed_blocks_folded_bone_cross_sheet_mapping():
    """v15: overlap-interface correspondence is local to observed rebased iC.

    Two nearby folded bone sheets reproduce the failure mode seen in the femoral
    condyles: v14's global nearest-bone search sends many valid cartilage-interface
    voxels to the wrong sheet.  v15 searches only the observed-iC bone domain, keeps
    the voxel closing/component logic, then clamps faces back to observed iC.
    """
    shape = (40, 50, 32)
    vox = np.array([0.365, 0.365, 0.7], dtype=float)
    bone = np.zeros(shape, bool)
    cart = np.zeros(shape, bool)
    # U/folded femur surrogate: two close condylar sheets connected superiorly.
    left_end, right_start = 18, 22
    bone[6:left_end, 8:42, 12:24] = True
    bone[right_start:34, 8:42, 12:24] = True
    bone[6:34, 8:42, 22:27] = True
    # Observed cartilage exists only on the intended inferior articular sheets.
    cart[8:left_end - 2, 10:40, 10:12] = True
    cart[right_start + 2:32, 10:40, 10:12] = True
    prior = cart.copy()

    n_remove = int(np.ceil(1.0 / np.prod(vox)))
    bone = preprocess_binary(bone, n_remove)
    cart = preprocess_binary(cart, n_remove)
    whole = boundary_mesh(cart)
    inner0, _ = split_cartilage_mesh(
        cart, bone, whole, finetune=True, closing_iterations=4, vox=vox
    )

    dbg14 = {}
    build_total_subchondral_mesh(
        bone, cart, prior, inner0, vox, torch.device("cpu"), "FemoralCartilage",
        debug_info=dbg14, paper_fcl_geometry=True, balanced_scb_closing=True,
        stage_roi_debug=True, overlap_interface_scb_seed=True,
    )
    dbg15 = {}
    build_total_subchondral_mesh(
        bone, cart, prior, inner0, vox, torch.device("cpu"), "FemoralCartilage",
        debug_info=dbg15, paper_fcl_geometry=True, balanced_scb_closing=True,
        stage_roi_debug=True, constrained_interface_scb_seed=True,
    )

    assert dbg14["seed_observed_support_fraction_area"] < 0.5
    assert dbg15["scb_seed_mode"] == "v15_constrained_overlap_interface_observed_iC"
    assert dbg15["seed_observed_support_fraction_area"] == 1.0
    assert dbg15["seed_rebased_coverage_fraction_area"] > 0.9
    assert dbg15["seed_area_removed_by_observed_clamp_mm2"] > 0.0
    # Healthy folded geometry must not trigger a connectivity-fill explosion.
    assert np.isclose(
        dbg15["after_hole_fill_area_mm2"], dbg15["seed_surface_area_mm2"],
        rtol=0.0, atol=1e-12,
    )



def test_batch_seed_flags_are_declared_in_process_case():
    """Regression for v14 batch NameError / candidate-flag propagation."""
    import inspect
    import batch_morphology as bm
    params = inspect.signature(bm.process_case).parameters
    assert "overlap_interface_scb_seed" in params
    assert "constrained_interface_scb_seed" in params



def test_v16_contact_augmentation_recovers_half_voxel_fc_faces():
    """v16 keeps Eq.11 OR and adds only contact faces missed by rounded row matching."""
    import morphology_gpu as mg
    shape = (36, 44, 28)
    vox = np.array([0.365, 0.365, 0.7], dtype=float)
    bone = np.zeros(shape, bool); bone[5:31, 6:38, 11:20] = True
    cart = np.zeros(shape, bool); cart[7:29, 8:36, 9:11] = True
    n_remove = int(np.ceil(1.0 / np.prod(vox)))
    bone = mg.preprocess_binary(bone, n_remove)
    cart = mg.preprocess_binary(cart, n_remove)
    whole = mg.boundary_mesh(cart)
    d0 = {}; d1 = {}
    i0, _ = mg.split_cartilage_mesh(cart, bone, whole, finetune=True,
                                    closing_iterations=4, debug_info=d0, vox=vox,
                                    contact_augment_inner=False)
    i1, _ = mg.split_cartilage_mesh(cart, bone, whole, finetune=True,
                                    closing_iterations=4, debug_info=d1, vox=vox,
                                    contact_augment_inner=True)
    assert d1["eq11_or_inner_area_mm2"] == d0["eq11_or_inner_area_mm2"]
    assert d1["contact_added_inner_faces_preclosing"] > 0
    assert d1["contact_added_inner_area_mm2_preclosing"] > 0
    assert d1["contact_augmented_inner_area_mm2_preclosing"] > d1["eq11_or_inner_area_mm2"]
    assert mg._mesh_area_numpy(i1, vox) > mg._mesh_area_numpy(i0, vox)


if __name__ == "__main__":
    test_matlab_round()
    test_roi_order()
    test_parcellation_shapes()
    test_mpt_components_require_shared_edge()
    test_registration_resolution()
    test_thickness_gpu_kernel_on_parallel_planes()
    test_fcl_reconstruction_healthy_overlap_is_zero_and_defect_is_positive()
    test_fcl_output_is_percentage_by_default()
    test_regional_volume_uses_physical_voxel_product()
    test_stage_roi_debug_does_not_change_scb_geometry()
    test_interface_scb_seed_is_subset_of_observed_rebased_interface()
    test_overlap_interface_scb_seed_uses_bone_facing_overlap_only()
    test_constrained_interface_scb_seed_blocks_folded_bone_cross_sheet_mapping()
    test_batch_seed_flags_are_declared_in_process_case()
    test_v16_contact_augmentation_recovers_half_voxel_fc_faces()
    print("all tests passed")
