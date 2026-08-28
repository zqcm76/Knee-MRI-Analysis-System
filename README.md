# CartiMorph v16-final — Python/CUDA Knee Cartilage Morphometrics

An independent Python/CUDA research implementation of the CartiMorph knee cartilage morphometrics workflow, with uniGradICON-based template-to-subject registration and a frozen **v16-final** morphology configuration.

The pipeline computes regional knee cartilage morphometrics from a subject tissue segmentation and a warped healthy-template tissue segmentation:

- Full-thickness cartilage loss (**FCL**, %)
- Mean cartilage thickness (**mm**)
- Cartilage-covered subchondral bone surface area (**cAB**, mm²)
- Cartilage volume (**mm³**)
- Rule-based **20-region** cartilage parcellation
- Direct compartment-level outputs for comparison with Chondrometrics/POMA

> **Research use only.** This software is not a medical device and is not intended for diagnosis, treatment planning, or clinical decision-making.

## Status

This repository contains the frozen **v16-final** morphology path. Older v5–v15 A/B geometry modes are retained only where needed for backward-compatible error messages or reproducibility diagnostics; the normal v16 workflow forces the validated final configuration.

The frozen morphology configuration is:

- `--cart-surface-finetune`
- `--paper-fcl-geometry`
- `--balanced-scb-closing`
- `--constrained-interface-scb-seed`
- `--fc-contact-augment-inner`

Core implementation decisions retained in v16-final include physical voxel-volume scaling (`sx * sy * sz`), FCL in the `0–100%` scale, CartiMorph-style 20-ROI ordering, reconstructed total subchondral bone area (`tAB`) independent of observed cartilage coverage, and zero-padding of denuded/uncovered tAB vertices for mean thickness.

## Repository layout

```text
.
├── morphology_gpu.py                 # Core 20-ROI morphology/FCL implementation
├── batch_morphology.py               # Batch registration + frozen v16 morphology runner
├── register_unigradicon.py           # Bone-driven uniGradICON registration wrapper
├── test_morphology_gpu.py            # Synthetic/unit regression tests
├── requirements.txt                  # Python dependencies
├── validation/
│   ├── compare_chondrometrics.py      # CartiMorph-v16 vs Chondrometrics/POMA comparison
│   └── compare_fcl_paper_phr.py       # Paper-style FCL pseudo-hit-rate validation
└── docs/
    └── V16_FINAL_VALIDATION.txt       # Frozen validation summary
```

Patient/OAI images, segmentation masks, template files, Chondrometrics/POMA tables, human-rater spreadsheets, and per-subject generated outputs are **not required to be committed to this repository** and should normally be obtained or generated separately under their respective data-use terms.

## Requirements

- Python **3.10+**
- NVIDIA CUDA-capable GPU recommended
- PyTorch with a CUDA build for normal morphology execution
- `uniGradICON` for registration

Python packages used by the released code are listed in `requirements.txt`:

```text
numpy
scipy
scikit-image
torch
nibabel
unigradicon
matplotlib
```

Create an environment and install dependencies, for example:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA runtime if the default `pip` installation does not provide GPU support.

## Input label convention

The default tissue labels are:

| Label | Tissue |
|---:|---|
| 0 | Background |
| 1 | Femur |
| 2 | Femoral cartilage |
| 3 | Tibia |
| 4 | Medial tibial cartilage |
| 5 | Lateral tibial cartilage |

The labels can be overridden through the command-line options in `morphology_gpu.py`.

## 20-region output order

The released implementation follows the CartiMorph 20-ROI ordering:

```text
aMFC ecMFC ccMFC icMFC pMFC
aLFC ecLFC ccLFC icLFC pLFC
aMTC eMTC pMTC iMTC cMTC
aLTC eLTC pLTC iLTC cLTC
```

## Primary output definitions

For each ROI, v16-final reports four primary metrics:

1. **FCL (%)**

   ```text
   FCL = 100 × dAB / tAB
   ```

   where `tAB` is reconstructed total subchondral bone area and `dAB` is denuded subchondral bone area.

2. **Mean Thickness (mm)**

   Mean thickness over the final tAB domain, with denuded/uncovered tAB vertices assigned zero.

3. **Surface Area (mm²)**

   Cartilage-covered subchondral bone area (`cAB`).

4. **Volume (mm³)**

   Cartilage voxel count multiplied by physical voxel volume:

   ```text
   sx × sy × sz
   ```

## Running morphology for one case

If a warped template segmentation has already been generated:

```bash
python morphology_gpu.py \
  --seg /path/to/subject_seg.nii.gz \
  --registration /path/to/warped_template_seg.nii.gz \
  --knee-side right \
  --output /path/to/results/MorphQuant.csv \
  --save-meta-json /path/to/results/MorphQuant_meta.json
```

The v16-final script forces the validated final morphology configuration internally. Users do not need to provide the historical v7–v16 development flags for a normal run.

By default, morphology requires CUDA. `--allow-cpu-fallback` exists for debugging/testing but is not the recommended production path.

### Per-case outputs

A normal run produces:

```text
MorphQuant.csv
MorphQuant_Compartments.csv
FCL_Areas.csv
MorphQuant_meta.json        # when --save-meta-json is supplied
```

`MorphQuant.csv` contains the 20 ROI × 4 primary metrics.

`MorphQuant_Compartments.csv` contains direct union-domain metrics for:

```text
MTC / cMFC / LTC / cLFC
```

These compartment values are calculated directly from the union surface domain rather than by naively summing or averaging overlapping regional results.

`FCL_Areas.csv` records ROI-level `tAB`, `cAB`, `dAB`, and FCL audit quantities.

## Batch processing

`batch_morphology.py` expects one or more manifest CSV files with the columns:

```text
id,image,seg,knee_side
```

Example:

```csv
id,image,seg,knee_side
case001,/data/case001_dess.nii.gz,/data/case001_seg.nii.gz,right
case002,/data/case002_dess.nii.gz,/data/case002_seg.nii.gz,left
```

### Re-run morphology using existing registrations

```bash
python batch_morphology.py \
  --manifest manifests/cmtest.csv \
  --output-root registrations \
  --only-morphology
```

### Run registration when required

```bash
python batch_morphology.py \
  --manifest manifests/cmtest.csv \
  --template-root templates \
  --output-root registrations
```

The expected template layout is:

```text
templates/
├── KL0_R/
│   ├── KL0_template_dess.nii.gz
│   └── KL0_template_seg.nii.gz
└── KL0_L/
    ├── KL0_template_dess.nii.gz
    └── KL0_template_seg.nii.gz
```

The template-to-subject registration is performed by `register_unigradicon.py`, which wraps the official uniGradICON CLI and uses the subject/template bone masks to guide knee registration.

## Validation

### Code tests

```bash
python -m py_compile \
  morphology_gpu.py \
  batch_morphology.py \
  register_unigradicon.py \
  validation/compare_chondrometrics.py \
  validation/compare_fcl_paper_phr.py \
  test_morphology_gpu.py

python test_morphology_gpu.py
```

The frozen release validation records these checks as passing.

### Human FCL validation

The frozen 79-case validation summary reports:

- ALL20 median-human exact grade agreement: **76.3%**
- Within ±1 grade: **94.4%**
- Within ±2 grades: **98.4%**
- MAE: **0.3215 grades**
- QWK: **0.819**

For the paper-style pseudo hit rate (pHR) at a 10-percentage-point tolerance:

- COMMON16 paired, three-rater mean: **Python v16 = 0.8995**
- COMMON16 paired, three-rater mean: **Chondrometrics = 0.7062**
- ALL20 Python-only, three-rater mean: **0.9084**

See `docs/V16_FINAL_VALIDATION.txt` for the full frozen validation record.

### Known limitation

The release retains a known limitation in severe disease: FCL severity can be compressed downward for some high-grade lesions even though overall specificity and paper-style pHR are strong. Multiple reconstruction and post-processing variants were investigated; no tested change improved severe-case FCL without materially degrading human-zero specificity, so no additional v17 morphology rule was adopted in this release.

## Chondrometrics/POMA comparison

`validation/compare_chondrometrics.py` compares v16 outputs against an externally supplied Chondrometrics/POMA table.

```bash
python validation/compare_chondrometrics.py \
  --root registrations \
  --chondrometrics /path/to/Chondrometrics_AllMetrics_with_CMT-ID.csv \
  --out comparison_chondrometrics_v16
```

Direct regional Chondrometrics counterparts exist for 16 CartiMorph ROIs:

- Femur: `ecMFC ccMFC icMFC ecLFC ccLFC icLFC`
- Tibia: all 10 tibial ROIs

The four femoral `aMFC pMFC aLFC pLFC` values are not invented or inferred.

The comparison script converts Chondrometrics `cAB` from cm² to mm² by ×100. CartiMorph v16 outputs are not empirically rescaled to force agreement with Chondrometrics.

Chondrometrics/POMA should be treated as an independent analysis source, not as a numerical identity target for this implementation.

## Paper-style FCL pHR validation

When the required human-rater and Chondrometrics source tables are available locally:

```bash
python validation/compare_fcl_paper_phr.py \
  --root registrations \
  --mapping /path/to/Chondrometrics_AllMetrics_with_CMT-ID.csv \
  --rater1 /path/to/FCLgrading_rater1.xlsx \
  --rater2 /path/to/FCLgrading_rater2.xlsx \
  --rater3 /path/to/FCLgrading_rater3.xlsx \
  --out human_fcl_v16/paper_phr
```

This follows the CartiMorph paper-style FCL validation convention:

- each rater is evaluated independently;
- manual grade `0…10` is converted to a continuous target by `grade × 10` percentage points;
- a prediction is a hit at tolerance `R` when `|prediction − human target| <= R`.

## Data and template availability

This repository should normally contain **code, documentation, tests, and aggregate validation summaries only**.

Do not commit OAI MR images, subject segmentation files, registration outputs, or other subject-level research data unless you have explicit redistribution rights.

The original CartiMorph project documents OAIZIB-CM / OAI-ZIB resources and the CLAIR-Knee-103R template/atlas. Obtain those resources from their official release locations and follow the applicable citation and data-use requirements.

## Citation and attribution

This implementation is based on the methods and public implementation of CartiMorph. If you use this repository, cite the original CartiMorph work:

```bibtex
@article{YAO2024103035,
  title   = {CartiMorph: A framework for automated knee articular cartilage morphometrics},
  journal = {Medical Image Analysis},
  author  = {Yongcheng Yao and Junru Zhong and Liping Zhang and Sheheryar Khan and Weitian Chen},
  volume  = {91},
  pages   = {103035},
  year    = {2024},
  doi     = {10.1016/j.media.2023.103035}
}
```

Registration uses uniGradICON; please also cite the relevant uniGradICON publication when using the registration workflow.

Upstream projects:

- CartiMorph: https://github.com/YongchengYAO/CartiMorph
- CartiMorph Toolbox: https://github.com/YongchengYAO/CartiMorph-Toolbox
- uniGradICON: https://github.com/uncbiag/uniGradICON

## License

The public CartiMorph and CartiMorph-Toolbox repositories are released under **CC BY-NC 4.0**. Because this repository is a Python reimplementation/derivative research implementation of those methods and public code, do **not** assume that the complete repository can be relicensed under MIT, BSD, or Apache-2.0.

For a conservative public release, distribute this repository under **CC BY-NC 4.0**, preserve CartiMorph attribution, and document separately licensed dependencies such as uniGradICON.

If you intend commercial use or want to use a different software license, obtain appropriate permission/legal review first.

## Disclaimer

This repository is an independent research implementation and is **not an official CartiMorph release** and is not affiliated with or endorsed by the original CartiMorph or uniGradICON authors.

The software is provided for research and reproducibility purposes without clinical warranty.
