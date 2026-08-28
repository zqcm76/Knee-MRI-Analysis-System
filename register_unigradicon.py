#!/usr/bin/env python3
"""Bone-driven uniGradICON registration for OAIZIB / 3D-DESS knees.

Design
------
* fixed  = subject DESS
* moving = healthy KL=0 template DESS
* registration masks use ONLY femur (1) + tibia (3), dilated in physical mm
* cartilage labels (2/4/5) never drive the deformation
* template tissue segmentation is transported with nearest-neighbour interpolation
* optional 20-ROI atlas is transported for QC/visualization only
* output warped_template_seg.nii.gz is the preferred --registration input of
  morphology_gpu.py

The script intentionally wraps the official uniGradICON CLI rather than importing
private Python APIs, which keeps it resilient to internal package changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - dependency check for user env
    raise SystemExit(
        "nibabel is required. Install with: pip install nibabel scipy"
    ) from exc


DEFAULT_BONE_LABELS = (1, 3)  # OAIZIB-CM: femur, tibia


def _as_3d(img: "nib.Nifti1Image", name: str) -> np.ndarray:
    arr = np.asanyarray(img.dataobj)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D after squeeze, got {arr.shape}")
    return arr


def _check_same_grid(image_path: Path, seg_path: Path, atol: float = 1e-4) -> None:
    img = nib.load(str(image_path))
    seg = nib.load(str(seg_path))
    if img.shape[:3] != seg.shape[:3]:
        raise ValueError(
            f"Image/seg shape mismatch: {image_path.name} {img.shape[:3]} vs "
            f"{seg_path.name} {seg.shape[:3]}"
        )
    if not np.allclose(img.affine, seg.affine, atol=atol, rtol=0):
        raise ValueError(
            f"Image/seg affine mismatch: {image_path} vs {seg_path}. "
            "Resample the segmentation onto the image grid before registration."
        )


def bone_mask_from_seg(seg: np.ndarray, labels: Iterable[int] = DEFAULT_BONE_LABELS) -> np.ndarray:
    labels = tuple(int(x) for x in labels)
    return np.isin(seg.astype(np.int32, copy=False), labels)


def dilate_mask_mm(mask: np.ndarray, spacing_xyz: Sequence[float], dilation_mm: float) -> np.ndarray:
    """Euclidean dilation with physical spacing, robust to anisotropic DESS voxels."""
    mask = np.asarray(mask, dtype=bool)
    if dilation_mm <= 0:
        return mask
    # EDT on the complement returns physical distance to the nearest foreground voxel.
    dist = ndimage.distance_transform_edt(~mask, sampling=tuple(float(x) for x in spacing_xyz))
    return dist <= float(dilation_mm)


def save_mask_like(mask: np.ndarray, reference: "nib.Nifti1Image", out_path: Path) -> None:
    hdr = reference.header.copy()
    hdr.set_data_dtype(np.uint8)
    out = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header=hdr)
    out.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    out.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(out, str(out_path))


def make_bone_roi_mask(
    seg_path: Path,
    reference_image_path: Path,
    out_path: Path,
    dilation_mm: float = 4.0,
    bone_labels: Sequence[int] = DEFAULT_BONE_LABELS,
) -> Path:
    _check_same_grid(reference_image_path, seg_path)
    ref = nib.load(str(reference_image_path))
    seg_img = nib.load(str(seg_path))
    seg = _as_3d(seg_img, "segmentation")
    bone = bone_mask_from_seg(seg, bone_labels)
    if not bone.any():
        raise ValueError(f"No bone voxels {tuple(bone_labels)} found in {seg_path}")
    spacing = ref.header.get_zooms()[:3]
    roi = dilate_mask_mm(bone, spacing, dilation_mm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_mask_like(roi, ref, out_path)
    return out_path


def run_cmd(cmd: Sequence[str], *, env: Optional[dict] = None, dry_run: bool = False) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(map(str, cmd)), check=True, env=env)


def _require_cli(name: str, dry_run: bool = False) -> None:
    if dry_run:
        return
    if shutil.which(name) is None:
        raise RuntimeError(
            f"'{name}' not found in PATH. Activate the environment containing uniGradICON "
            "and run: pip install unigradicon"
        )


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    den = int(a.sum()) + int(b.sum())
    if den == 0:
        return float("nan")
    return 2.0 * float(np.count_nonzero(a & b)) / float(den)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    st = ndimage.generate_binary_structure(3, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=st, border_value=0)


def surface_distance_metrics(a: np.ndarray, b: np.ndarray, spacing: Sequence[float]) -> Tuple[float, float]:
    """Return ASSD and symmetric HD95 in mm."""
    sa = _surface(np.asarray(a, bool))
    sb = _surface(np.asarray(b, bool))
    if not sa.any() or not sb.any():
        return float("nan"), float("nan")
    dt_to_b = ndimage.distance_transform_edt(~sb, sampling=spacing)
    dt_to_a = ndimage.distance_transform_edt(~sa, sampling=spacing)
    d_ab = dt_to_b[sa]
    d_ba = dt_to_a[sb]
    assd = 0.5 * (float(np.mean(d_ab)) + float(np.mean(d_ba)))
    hd95 = float(np.percentile(np.concatenate([d_ab, d_ba]), 95))
    return assd, hd95


def _articular_surface(
    seg: np.ndarray,
    bone_label: int,
    cartilage_labels: Sequence[int],
    spacing: Sequence[float],
    cartilage_band_mm: float = 2.0,
) -> np.ndarray:
    """Bone surface voxels close to the corresponding cartilage.

    This is an operational QC proxy for the cartilage-bearing subchondral
    surface.  Cartilage labels do not drive registration; they are used here
    only to focus QC on the joint surface that matters for morphometry.
    """
    bone_surface = _surface(np.asarray(seg) == int(bone_label))
    cartilage = np.isin(np.asarray(seg), [int(x) for x in cartilage_labels])
    if not bone_surface.any() or not cartilage.any():
        return np.zeros_like(bone_surface, dtype=bool)
    dist = ndimage.distance_transform_edt(~cartilage, sampling=tuple(float(x) for x in spacing))
    return bone_surface & (dist <= float(cartilage_band_mm))


def compute_articular_qc(
    fixed_seg_path: Path,
    warped_template_seg_path: Path,
    cartilage_band_mm: float = 2.0,
) -> Dict[str, float]:
    """ASSD/HD95 on cartilage-bearing femoral and tibial bone surfaces."""
    fixed_img = nib.load(str(fixed_seg_path))
    warped_img = nib.load(str(warped_template_seg_path))
    fixed = _as_3d(fixed_img, "fixed segmentation").astype(np.int16)
    warped = _as_3d(warped_img, "warped template segmentation").astype(np.int16)
    if fixed.shape != warped.shape:
        raise ValueError(f"QC grid mismatch: {fixed.shape} vs {warped.shape}")
    spacing = tuple(float(x) for x in fixed_img.header.get_zooms()[:3])

    out: Dict[str, float] = {}
    specs = (
        ("femur_articular", 1, (2,)),
        ("tibia_articular", 3, (4, 5)),
    )
    for name, bone_lab, cart_labs in specs:
        a = _articular_surface(fixed, bone_lab, cart_labs, spacing, cartilage_band_mm)
        b = _articular_surface(warped, bone_lab, cart_labs, spacing, cartilage_band_mm)
        assd, hd95 = surface_distance_metrics(a, b, spacing)
        out[f"{name}_assd_mm"] = assd
        out[f"{name}_hd95_mm"] = hd95
        out[f"{name}_surface_voxels_fixed"] = float(np.count_nonzero(a))
        out[f"{name}_surface_voxels_warped"] = float(np.count_nonzero(b))
    vals = [out.get("femur_articular_assd_mm"), out.get("tibia_articular_assd_mm")]
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    out["articular_assd_mm"] = float(np.mean(vals)) if vals else float("nan")
    vals95 = [out.get("femur_articular_hd95_mm"), out.get("tibia_articular_hd95_mm")]
    vals95 = [float(v) for v in vals95 if v is not None and np.isfinite(v)]
    out["articular_hd95_mm"] = float(np.mean(vals95)) if vals95 else float("nan")
    out["articular_band_mm"] = float(cartilage_band_mm)
    return out


def compute_bone_qc(
    fixed_seg_path: Path,
    warped_template_seg_path: Path,
    femur_label: int = 1,
    tibia_label: int = 3,
) -> Dict[str, float]:
    fixed_img = nib.load(str(fixed_seg_path))
    warped_img = nib.load(str(warped_template_seg_path))
    fixed = _as_3d(fixed_img, "fixed segmentation").astype(np.int16)
    warped = _as_3d(warped_img, "warped template segmentation").astype(np.int16)
    if fixed.shape != warped.shape:
        raise ValueError(f"QC grid mismatch: {fixed.shape} vs {warped.shape}")
    spacing = tuple(float(x) for x in fixed_img.header.get_zooms()[:3])

    result: Dict[str, float] = {}
    for name, lab in (("femur", femur_label), ("tibia", tibia_label)):
        a = fixed == lab
        b = warped == lab
        assd, hd95 = surface_distance_metrics(a, b, spacing)
        result[f"{name}_dice"] = _dice(a, b)
        result[f"{name}_assd_mm"] = assd
        result[f"{name}_hd95_mm"] = hd95

    fixed_bone = np.isin(fixed, [femur_label, tibia_label])
    warped_bone = np.isin(warped, [femur_label, tibia_label])
    assd, hd95 = surface_distance_metrics(fixed_bone, warped_bone, spacing)
    result["bone_union_dice"] = _dice(fixed_bone, warped_bone)
    result["bone_union_assd_mm"] = assd
    result["bone_union_hd95_mm"] = hd95
    return result


def register_case(
    *,
    fixed_image: Path,
    fixed_seg: Path,
    template_image: Path,
    template_seg: Path,
    out_dir: Path,
    template_atlas: Optional[Path] = None,
    bone_dilation_mm: float = 4.0,
    io_iterations: str = "None",
    io_sim: str = "lncc2",
    model: str = "unigradicon",
    cuda_visible_devices: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Path]:
    """Register healthy template -> subject using bone-masked DESS intensity."""
    for p in (fixed_image, fixed_seg, template_image, template_seg):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    if template_atlas is not None and not Path(template_atlas).exists():
        raise FileNotFoundError(template_atlas)

    _check_same_grid(fixed_image, fixed_seg)
    _check_same_grid(template_image, template_seg)
    _require_cli("unigradicon-register", dry_run)
    _require_cli("unigradicon-warp", dry_run)

    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_bone_roi = out_dir / "fixed_bone_roi.nii.gz"
    moving_bone_roi = out_dir / "template_bone_roi.nii.gz"
    make_bone_roi_mask(fixed_seg, fixed_image, fixed_bone_roi, bone_dilation_mm)
    make_bone_roi_mask(template_seg, template_image, moving_bone_roi, bone_dilation_mm)

    transform = out_dir / "template_to_subject.hdf5"
    warped_template_image = out_dir / "warped_template_dess.nii.gz"
    warped_template_seg = out_dir / "warped_template_seg.nii.gz"
    warped_template_atlas = out_dir / "warped_template_atlas20.nii.gz"

    env = os.environ.copy()
    # Helps fragmentation on small GPUs when supported by the installed PyTorch.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    reg_cmd = [
        "unigradicon-register",
        f"--fixed={fixed_image}",
        "--fixed_modality=mri",
        f"--fixed_segmentation={fixed_bone_roi}",
        f"--moving={template_image}",
        "--moving_modality=mri",
        f"--moving_segmentation={moving_bone_roi}",
        f"--transform_out={transform}",
        f"--warped_moving_out={warped_template_image}",
        f"--io_iterations={io_iterations}",
    ]
    if model and model != "unigradicon":
        reg_cmd.append(f"--model={model}")

    io_enabled = str(io_iterations).lower() not in {"none", "0"}
    if io_enabled:
        reg_cmd += [f"--io_sim={io_sim}", "--loss_function_masking"]

    run_cmd(reg_cmd, env=env, dry_run=dry_run)

    run_cmd(
        [
            "unigradicon-warp",
            "--fixed", str(fixed_image),
            "--moving", str(template_seg),
            "--transform", str(transform),
            "--warped_moving_out", str(warped_template_seg),
            "--nearest_neighbor",
        ],
        env=env,
        dry_run=dry_run,
    )

    if template_atlas is not None:
        run_cmd(
            [
                "unigradicon-warp",
                "--fixed", str(fixed_image),
                "--moving", str(template_atlas),
                "--transform", str(transform),
                "--warped_moving_out", str(warped_template_atlas),
                "--nearest_neighbor",
            ],
            env=env,
            dry_run=dry_run,
        )

    outputs = {
        "transform": transform,
        "warped_template_image": warped_template_image,
        "warped_template_seg": warped_template_seg,
        "fixed_bone_roi": fixed_bone_roi,
        "template_bone_roi": moving_bone_roi,
    }
    if template_atlas is not None:
        outputs["warped_template_atlas"] = warped_template_atlas
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bone-driven uniGradICON template->subject registration for OAIZIB 3D-DESS"
    )
    ap.add_argument("--fixed-image", required=True, type=Path, help="subject 3D-DESS NIfTI")
    ap.add_argument("--fixed-seg", required=True, type=Path, help="subject 5-label segmentation")
    ap.add_argument("--template-image", required=True, type=Path, help="KL=0 healthy template DESS")
    ap.add_argument("--template-seg", required=True, type=Path, help="KL=0 healthy template 5-label segmentation")
    ap.add_argument("--template-atlas", type=Path, help="optional template 20-ROI atlas; warped for QC only")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bone-dilation-mm", type=float, default=4.0,
                    help="physical dilation around femur+tibia used to mask DESS (default: 4 mm)")
    ap.add_argument("--io-iterations", default="None",
                    help="default None (prediction-only). Use e.g. 10/20/50 only if GPU memory permits")
    ap.add_argument("--io-sim", default="lncc2", choices=["lncc", "lncc2", "mind"])
    ap.add_argument("--model", default="unigradicon", choices=["unigradicon", "multigradicon"])
    ap.add_argument("--cuda-visible-devices", help="e.g. 0")
    ap.add_argument("--qc-min-dice", type=float, default=0.90,
                    help="operational warning threshold; calibrate on your train/val set")
    ap.add_argument("--qc-max-hd95-mm", type=float, default=5.0,
                    help="operational warning threshold; calibrate on your train/val set")
    ap.add_argument("--fail-on-qc", action="store_true")
    ap.add_argument("--run-morphology", action="store_true",
                    help="run morphology_gpu.py after successful registration")
    ap.add_argument("--morphology-script", type=Path, default=Path(__file__).with_name("morphology_gpu.py"))
    ap.add_argument("--knee-side", choices=["left", "right"], help="required with --run-morphology")
    ap.add_argument("--morphology-output", type=Path, help="default: OUT_DIR/MorphQuant.csv")
    ap.add_argument("--allow-morphology-cpu-fallback", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print uniGradICON commands without executing them")
    args = ap.parse_args(argv)

    outputs = register_case(
        fixed_image=args.fixed_image,
        fixed_seg=args.fixed_seg,
        template_image=args.template_image,
        template_seg=args.template_seg,
        template_atlas=args.template_atlas,
        out_dir=args.out_dir,
        bone_dilation_mm=args.bone_dilation_mm,
        io_iterations=args.io_iterations,
        io_sim=args.io_sim,
        model=args.model,
        cuda_visible_devices=args.cuda_visible_devices,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry-run complete.")
        return 0

    qc = compute_bone_qc(args.fixed_seg, outputs["warped_template_seg"])
    qc["bone_dilation_mm"] = float(args.bone_dilation_mm)
    qc["io_iterations"] = args.io_iterations
    qc["model"] = args.model
    qc_pass = (
        np.isfinite(qc["femur_dice"]) and qc["femur_dice"] >= args.qc_min_dice
        and np.isfinite(qc["tibia_dice"]) and qc["tibia_dice"] >= args.qc_min_dice
        and np.isfinite(qc["femur_hd95_mm"]) and qc["femur_hd95_mm"] <= args.qc_max_hd95_mm
        and np.isfinite(qc["tibia_hd95_mm"]) and qc["tibia_hd95_mm"] <= args.qc_max_hd95_mm
    )
    qc["operational_qc_pass"] = bool(qc_pass)
    qc_path = args.out_dir / "registration_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(qc, indent=2, ensure_ascii=False))
    print(f"Warped template segmentation: {outputs['warped_template_seg']}")
    print(f"QC: {qc_path}")

    if not qc_pass:
        print(
            "WARNING: registration is below the configured operational QC threshold. "
            "Inspect overlays before morphology; consider IO only if memory permits.",
            file=sys.stderr,
        )
        if args.fail_on_qc:
            return 2

    if args.run_morphology:
        if args.knee_side is None:
            raise ValueError("--knee-side is required with --run-morphology")
        morph_out = args.morphology_output or (args.out_dir / "MorphQuant.csv")
        cmd = [
            sys.executable, str(args.morphology_script),
            "--seg", str(args.fixed_seg),
            "--registration", str(outputs["warped_template_seg"]),
            "--knee-side", args.knee_side,
            "--output", str(morph_out),
            "--save-meta-json", str(args.out_dir / "MorphQuant_meta.json"),
        ]
        if args.allow_morphology_cpu_fallback:
            cmd.append("--allow-cpu-fallback")
        run_cmd(cmd, env=os.environ.copy(), dry_run=False)
        print(f"Morphology: {morph_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
