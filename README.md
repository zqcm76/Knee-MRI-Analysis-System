# CartiMorph 20-Region Cartilage Morphological Measurement

Python implementation of the CartiMorph cartilage morphological analysis pipeline.

## Features

- **Rule-based 20-region parcellation** (not atlas-based, per the CartiMorph paper)
- **Volume** (mm³) — voxel-based
- **Surface area** (mm²) — triangular mesh-based
- **Thickness** (mm) — surface normal ray-triangle intersection
- **dABp** (%) — denuded area bone percentage (full-thickness cartilage loss)

## 20 Regions

### Femoral Cartilage (FC) — ROI 1-10
| ROI | Name | Description |
|-----|------|-------------|
| 1 | aMFC | Anterior Medial FC |
| 2 | ecMFC | Exterior Central Medial FC |
| 3 | ccMFC | Central Central Medial FC |
| 4 | icMFC | Interior Central Medial FC |
| 5 | pMFC | Posterior Medial FC |
| 6 | aLFC | Anterior Lateral FC |
| 7 | ecLFC | Exterior Central Lateral FC |
| 8 | ccLFC | Central Central Lateral FC |
| 9 | icLFC | Interior Central Lateral FC |
| 10 | pLFC | Posterior Lateral FC |

### Tibial Cartilage (TC) — ROI 11-20
| ROI | Name | Description |
|-----|------|-------------|
| 11 | aMTC | Anterior Medial TC |
| 12 | eMTC | Exterior Medial TC |
| 13 | pMTC | Posterior Medial TC |
| 14 | iMTC | Interior Medial TC |
| 15 | cMTC | Central Medial TC |
| 16 | aLTC | Anterior Lateral TC |
| 17 | eLTC | Exterior Lateral TC |
| 18 | pLTC | Posterior Lateral TC |
| 19 | iLTC | Interior Lateral TC |
| 20 | cLTC | Central Lateral TC |

## Label Convention (OAIZIB Dataset)

| Label | Structure |
|-------|-----------|
| 0 | Background |
| 1 | Femoral Bone |
| 2 | Tibial Bone |
| 3 | Femoral Cartilage (FC) |
| 4 | Medial Tibial Cartilage (MTC) |
| 5 | Lateral Tibial Cartilage (LTC) |

## Usage

### Single Case
```bash
python cartimorph_measure.py case001.nii.gz --output results/ --save-atlas
```

### Skip Thickness (faster)
```bash
python cartimorph_measure.py case001.nii.gz --output results/ --no-thickness
```

### Batch Processing
```bash
python cartimorph_measure.py labelsTr/*.nii.gz --output results/ --batch
```

### Custom Labels
```bash
python cartimorph_measure.py case001.nii.gz \
    --label_fc 3 --label_mtc 4 --label_ltc 5 \
    --label_femur_bone 1 --label_tibia_bone 2
```

### Left Knee
```bash
python cartimorph_measure.py case001.nii.gz --knee_side left
```

## Output

- `*_report.csv` — Per-region morphological measurements
- `*_report.json` — Same data in JSON format
- `*_atlas.nii.gz` — 20-region parcellation atlas (if `--save-atlas`)

## Dependencies

```
numpy>=1.20
scipy>=1.7
nibabel>=3.2
scikit-image>=0.18
```

## Performance

- **Without thickness**: ~18 seconds per case
- **With thickness**: ~10 minutes per case (dominated by ray-triangle intersection on large FC regions)
- **With dABp**: additional ~30 seconds per case

## Files

| File | Description |
|------|-------------|
| `cartimorph_measure.py` | Main entry point, CLI, batch processing |
| `parcellation.py` | Rule-based 20-region parcellation (FC + TC) |
| `mesh_utils.py` | Mesh operations (marching cubes, area, boundary) |
| `thickness.py` | Surface normal estimation + ray-triangle thickness |
| `dabp.py` | dABp (denuded area bone percentage) calculation |
