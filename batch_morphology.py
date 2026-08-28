#!/usr/bin/env python3
"""Batch OAIZIB registration + frozen CartiMorph v16-final morphology.

Default/fixed morphology configuration:
  --cart-surface-finetune
  --paper-fcl-geometry
  --balanced-scb-closing
  --constrained-interface-scb-seed
  --fc-contact-augment-inner

The batch runner intentionally does not expose older candidate geometry modes.
It reuses registrations/<id>/warped_template_seg.nii.gz when present and writes:
  MorphQuant.csv
  MorphQuant_Compartments.csv
  FCL_Areas.csv
  MorphQuant_meta.json
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


def normalize_knee_side(value: object) -> str:
    s = str(value).strip().lower()
    if s in {"right", "r", "rt", "1", "1.0"}:
        return "right"
    if s in {"left", "l", "lt", "2", "2.0"}:
        return "left"
    raise ValueError(f"Unsupported knee_side={value!r}; expected left/right (or L/R, 1/2)")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"id", "image", "seg", "knee_side"}
    cols = set(rows[0].keys()) if rows else set()
    missing = required - cols
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return rows


def run_cmd(cmd: Sequence[object]) -> None:
    text = [str(x) for x in cmd]
    print("+", " ".join(text), flush=True)
    # Do not capture stdout/stderr: uniGradICON/morphology progress remains visible.
    subprocess.run(text, check=True)


def rerun_morphology(
    *,
    fixed_seg: Path,
    warped_template_seg: Path,
    knee_side: str,
    out_dir: Path,
    morphology_script: Path,
    allow_cpu_fallback: bool,
    profile_morphology: bool,
    debug_surfaces: bool,
    physical_scb_mapping: bool,
    cart_surface_finetune: bool,
    surface_closing_iterations: int,
    paper_fcl_geometry: bool,
    balanced_scb_closing: bool,
    matlab_native_ic: bool,
    compat_native_ic_map: bool,
    stage_roi_debug: bool,
    interface_scb_seed: bool,
    overlap_interface_scb_seed: bool,
    constrained_interface_scb_seed: bool,
    fc_contact_augment_inner: bool,
) -> None:
    morph_out = out_dir / "MorphQuant.csv"
    cmd: list[object] = [
        sys.executable,
        morphology_script,
        "--seg", fixed_seg,
        "--registration", warped_template_seg,
        "--knee-side", knee_side,
        "--output", morph_out,
        "--save-meta-json", out_dir / "MorphQuant_meta.json",
    ]
    if allow_cpu_fallback:
        cmd.append("--allow-cpu-fallback")
    if profile_morphology:
        cmd.append("--profile")
    if debug_surfaces:
        cmd.append("--debug-surfaces")
    if physical_scb_mapping:
        cmd.append("--physical-scb-mapping")
    if cart_surface_finetune:
        cmd.extend(["--cart-surface-finetune", "--surface-closing-iterations", str(surface_closing_iterations)])
    if paper_fcl_geometry:
        cmd.append("--paper-fcl-geometry")
    if balanced_scb_closing:
        cmd.append("--balanced-scb-closing")
    if matlab_native_ic:
        cmd.append("--matlab-native-ic")
    if compat_native_ic_map:
        cmd.append("--compat-native-ic-map")
    if stage_roi_debug:
        cmd.append("--stage-roi-debug")
    if interface_scb_seed:
        cmd.append("--interface-scb-seed")
    if overlap_interface_scb_seed:
        cmd.append("--overlap-interface-scb-seed")
    if constrained_interface_scb_seed:
        cmd.append("--constrained-interface-scb-seed")
    if fc_contact_augment_inner:
        cmd.append("--fc-contact-augment-inner")
    run_cmd(cmd)


def process_case(
    row: dict[str, str],
    *,
    template_root: Path,
    output_root: Path,
    register_script: Path,
    morphology_script: Path,
    bone_dilation_mm: float,
    io_iterations: str,
    cuda_visible_devices: str,
    force_registration: bool,
    only_morphology: bool,
    skip_existing_morphology: bool,
    allow_morphology_cpu_fallback: bool,
    profile_morphology: bool,
    debug_surfaces: bool,
    physical_scb_mapping: bool,
    cart_surface_finetune: bool,
    surface_closing_iterations: int,
    paper_fcl_geometry: bool,
    balanced_scb_closing: bool,
    matlab_native_ic: bool,
    compat_native_ic_map: bool,
    stage_roi_debug: bool,
    interface_scb_seed: bool,
    overlap_interface_scb_seed: bool,
    constrained_interface_scb_seed: bool,
    fc_contact_augment_inner: bool,
) -> str:
    sample_id = str(row["id"]).strip()
    if not sample_id:
        raise ValueError("empty sample id")
    fixed_img = Path(str(row["image"]).strip())
    fixed_seg = Path(str(row["seg"]).strip())
    knee_side = normalize_knee_side(row["knee_side"])

    template_folder = "KL0_R" if knee_side == "right" else "KL0_L"
    template_dir = template_root / template_folder
    template_img = template_dir / "KL0_template_dess.nii.gz"
    template_seg = template_dir / "KL0_template_seg.nii.gz"

    out_dir = output_root / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    warped_template_seg = out_dir / "warped_template_seg.nii.gz"
    morph_out = out_dir / "MorphQuant.csv"

    if skip_existing_morphology and morph_out.exists() and not force_registration:
        print(f"SKIP {sample_id}: MorphQuant.csv already exists")
        return "skipped"

    if warped_template_seg.exists() and not force_registration:
        print(
            f"REUSE {sample_id}: {warped_template_seg} already exists; "
            "rerunning morphology only."
        )
        rerun_morphology(
            fixed_seg=fixed_seg,
            warped_template_seg=warped_template_seg,
            knee_side=knee_side,
            out_dir=out_dir,
            morphology_script=morphology_script,
            allow_cpu_fallback=allow_morphology_cpu_fallback,
            profile_morphology=profile_morphology,
            debug_surfaces=debug_surfaces,
            physical_scb_mapping=physical_scb_mapping,
            cart_surface_finetune=cart_surface_finetune,
            surface_closing_iterations=surface_closing_iterations,
            paper_fcl_geometry=paper_fcl_geometry,
            balanced_scb_closing=balanced_scb_closing,
            matlab_native_ic=matlab_native_ic,
            compat_native_ic_map=compat_native_ic_map,
            stage_roi_debug=stage_roi_debug,
            interface_scb_seed=interface_scb_seed,
            overlap_interface_scb_seed=overlap_interface_scb_seed,
            constrained_interface_scb_seed=constrained_interface_scb_seed,
            fc_contact_augment_inner=fc_contact_augment_inner,
        )
        return "morphology-only"

    if only_morphology:
        raise FileNotFoundError(
            f"{sample_id}: --only-morphology requested but registration is missing: "
            f"{warped_template_seg}"
        )

    for p, label in (
        (fixed_img, "fixed image"),
        (fixed_seg, "fixed segmentation"),
        (template_img, "template image"),
        (template_seg, "template segmentation"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"{sample_id}: missing {label}: {p}")

    cmd: list[object] = [
        sys.executable,
        register_script,
        "--fixed-image", fixed_img,
        "--fixed-seg", fixed_seg,
        "--template-image", template_img,
        "--template-seg", template_seg,
        "--out-dir", out_dir,
        "--bone-dilation-mm", bone_dilation_mm,
        "--io-iterations", io_iterations,
        "--knee-side", knee_side,
        "--cuda-visible-devices", cuda_visible_devices,
    ]
    # Registration and morphology are intentionally separate here.  The old batch
    # path delegated morphology to register_unigradicon.py, which silently dropped
    # all v7-v15 morphology candidate/debug flags and always ran the default path.
    run_cmd(cmd)
    rerun_morphology(
        fixed_seg=fixed_seg,
        warped_template_seg=warped_template_seg,
        knee_side=knee_side,
        out_dir=out_dir,
        morphology_script=morphology_script,
        allow_cpu_fallback=allow_morphology_cpu_fallback,
        profile_morphology=profile_morphology,
        debug_surfaces=debug_surfaces,
        physical_scb_mapping=physical_scb_mapping,
        cart_surface_finetune=cart_surface_finetune,
        surface_closing_iterations=surface_closing_iterations,
        paper_fcl_geometry=paper_fcl_geometry,
        balanced_scb_closing=balanced_scb_closing,
        matlab_native_ic=matlab_native_ic,
        compat_native_ic_map=compat_native_ic_map,
        stage_roi_debug=stage_roi_debug,
        interface_scb_seed=interface_scb_seed,
        overlap_interface_scb_seed=overlap_interface_scb_seed,
        constrained_interface_scb_seed=constrained_interface_scb_seed,
        fc_contact_augment_inner=fc_contact_augment_inner,
    )
    return "registered+morphology"


def main(argv: Optional[Sequence[str]] = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Batch OAIZIB uniGradICON registration and frozen CartiMorph v16-final morphology"
    )
    ap.add_argument(
        "--manifest", nargs="+", type=Path, default=[Path("manifests/cmtest.csv")],
        help="CSV(s) with id,image,seg,knee_side columns",
    )
    ap.add_argument("--template-root", type=Path, default=Path("templates"))
    ap.add_argument("--output-root", type=Path, default=Path("registrations"))
    ap.add_argument("--register-script", type=Path, default=here / "register_unigradicon.py")
    ap.add_argument("--morphology-script", type=Path, default=here / "morphology_gpu.py")
    ap.add_argument("--bone-dilation-mm", type=float, default=4.0)
    ap.add_argument("--io-iterations", default="None")
    ap.add_argument("--cuda-visible-devices", default="0")
    ap.add_argument(
        "--force-registration", action="store_true",
        help="rerun uniGradICON even if warped_template_seg.nii.gz already exists",
    )
    ap.add_argument(
        "--only-morphology", action="store_true",
        help="never run registration; fail a case if warped_template_seg.nii.gz is missing",
    )
    ap.add_argument(
        "--skip-existing-morphology", action="store_true",
        help="skip a case if MorphQuant.csv already exists (off by default so bug-fix reruns overwrite it)",
    )
    ap.add_argument("--allow-morphology-cpu-fallback", action="store_true")
    ap.add_argument(
        "--profile-morphology", action="store_true",
        help="pass --profile to morphology-only reruns and print synchronized stage timings",
    )
    ap.add_argument(
        "--debug-surfaces", action="store_true",
        help="pass --debug-surfaces to morphology-only reruns and save cAB/tAB/inner/scB diagnostics",
    )
    ap.add_argument(
        "--physical-scb-mapping", action="store_true",
        help="rejected v5 A/B option retained for reproducibility only; do not combine with v6 surface test",
    )
    ap.add_argument(
        "--cart-surface-finetune", action="store_true",
        help="candidate v6: grown-bone contact inner split + CartiMorph closing/restricted outer dilation",
    )
    ap.add_argument(
        "--surface-closing-iterations", type=int, default=4,
        help="candidate v6 inner surface closing iterations (default 4)",
    )
    ap.add_argument("--paper-fcl-geometry", action="store_true")
    ap.add_argument("--balanced-scb-closing", action="store_true")
    ap.add_argument("--matlab-native-ic", action="store_true")
    ap.add_argument("--compat-native-ic-map", action="store_true")
    ap.add_argument("--interface-scb-seed", action="store_true",
                    help="v13 candidate: seed tAB from observed-iC/prior surface intersection")
    ap.add_argument("--overlap-interface-scb-seed", action="store_true",
                    help="v14 candidate: seed tAB from bone-facing interface of subject/prior overlap")
    ap.add_argument("--constrained-interface-scb-seed", action="store_true",
                    help="v15 candidate: constrain overlap-interface seed mapping to rebased observed-iC bone domain")
    ap.add_argument("--fc-contact-augment-inner", action="store_true",
                    help="v16 candidate: FC only, augment Eq.11 OR inner seed with bone-contact faces")
    ap.add_argument(
        "--stage-roi-debug", action="store_true",
        help="pass v12 reconstruction-stage ROI diagnostics; requires --debug-surfaces",
    )
    args = ap.parse_args(argv)

    # v16-final is intentionally frozen. Older A/B flags remain in argparse only
    # for command-line compatibility, but the final batch path below is fixed.
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

    if args.force_registration and args.only_morphology:
        ap.error("--force-registration and --only-morphology are mutually exclusive")
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
        ap.error("v13/v14/v15 seed flags are alternative A/B modes")

    failures: list[tuple[str, str]] = []
    counts = {"registered+morphology": 0, "morphology-only": 0, "skipped": 0}
    for csv_file in args.manifest:
        print(f"\n===== {csv_file} =====")
        rows = read_manifest(csv_file)
        for i, row in enumerate(rows, start=1):
            sample_id = str(row.get("id", "")).strip() or f"row-{i}"
            print(f"\n[{i}/{len(rows)}] sample: {sample_id}")
            try:
                state = process_case(
                    row,
                    template_root=args.template_root,
                    output_root=args.output_root,
                    register_script=args.register_script,
                    morphology_script=args.morphology_script,
                    bone_dilation_mm=args.bone_dilation_mm,
                    io_iterations=str(args.io_iterations),
                    cuda_visible_devices=str(args.cuda_visible_devices),
                    force_registration=args.force_registration,
                    only_morphology=args.only_morphology,
                    skip_existing_morphology=args.skip_existing_morphology,
                    allow_morphology_cpu_fallback=args.allow_morphology_cpu_fallback,
                    profile_morphology=args.profile_morphology,
                    debug_surfaces=args.debug_surfaces,
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
                )
                counts[state] += 1
                print(f"OK {sample_id}: {state}")
            except Exception as exc:
                failures.append((sample_id, str(exc)))
                print(f"FAIL {sample_id}: {exc}", file=sys.stderr)

    print("\n===== summary =====")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"failed: {len(failures)}")
    if failures:
        for sample_id, msg in failures:
            print(f"  - {sample_id}: {msg}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
