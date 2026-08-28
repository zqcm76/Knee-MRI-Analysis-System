# CartiMorph v16-final — Python/CUDA 膝关节软骨形态学分析

这是一个基于 **Python / CUDA** 的 CartiMorph 膝关节软骨形态学研究实现，结合 **uniGradICON** 完成健康模板到受试者的配准，并使用已经冻结的 **v16-final** 形态学配置进行定量分析。

该流程以受试者组织分割结果和配准后的健康模板分割结果为输入，计算：

- 全层软骨缺损（Full-thickness Cartilage Loss, **FCL**，%）
- 平均软骨厚度（**mm**）
- 软骨覆盖的软骨下骨表面积（**cAB，mm²**）
- 软骨体积（**mm³**）
- 基于规则的 **20 个软骨亚区** 定量
- 用于与 Chondrometrics/POMA 比较的 compartment-level 输出

> **仅限科研使用。** 本软件不是医疗器械，不应直接用于临床诊断、治疗决策或患者管理。

---

## 项目状态

本仓库发布的是已经冻结的 **v16-final morphology pipeline**。

v5–v15 中用于算法调试和 A/B 实验的历史几何模式不再属于最终推荐工作流。正常运行时，v16-final 固定采用以下配置：

- `--cart-surface-finetune`
- `--paper-fcl-geometry`
- `--balanced-scb-closing`
- `--constrained-interface-scb-seed`
- `--fc-contact-augment-inner`

v16-final 保留的主要实现原则包括：

- 体积使用真实物理体素体积 `sx * sy * sz`
- FCL 使用 `0–100%` 百分比尺度
- 保留 CartiMorph 风格的 20 ROI 顺序
- 重建的总软骨下骨面积 `tAB` 不依赖当前观测到的 cartilage coverage
- 在平均厚度计算中，denuded / uncovered 的 tAB 顶点按 0 厚度计入
- 保留经过验证的最终 SCB seed、closing 和 FC contact augmentation 逻辑

---

## 仓库结构

```text
.
├── morphology_gpu.py
├── batch_morphology.py
├── register_unigradicon.py
├── test_morphology_gpu.py
├── requirements.txt
├── validation/
│   ├── compare_chondrometrics.py
│   └── compare_fcl_paper_phr.py
└── docs/
    └── V16_FINAL_VALIDATION.txt
```

建议 GitHub 仓库只发布：

- 核心代码
- 测试代码
- 验证脚本
- 文档
- 聚合后的验证统计

MRI、病例级 segmentation、registration 输出、人工评分表、POMA/Chondrometrics 原始表格和逐病例结果应根据各自的数据使用协议单独获取或生成。

---

## 环境要求

推荐环境：

- Python **3.10+**
- NVIDIA CUDA GPU
- 支持 CUDA 的 PyTorch
- `uniGradICON`
- NumPy / SciPy / scikit-image / nibabel / matplotlib

创建虚拟环境：

```bash
python -m venv .venv
```

Linux / macOS：

```bash
source .venv/bin/activate
```

Windows：

```bat
.venv\Scripts\activate
```

安装依赖：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

主要依赖：

```text
numpy
scipy
scikit-image
torch
nibabel
unigradicon
matplotlib
```

如果默认安装的 PyTorch 不支持你的 CUDA 环境，请根据本机 CUDA 版本安装对应的 PyTorch build。

---

## 默认 segmentation 标签

默认组织标签如下：

| Label | 组织 |
|---:|---|
| 0 | Background |
| 1 | Femur |
| 2 | Femoral cartilage |
| 3 | Tibia |
| 4 | Medial tibial cartilage |
| 5 | Lateral tibial cartilage |

如有需要，可以通过 `morphology_gpu.py` 的命令行参数修改标签定义。

---

## 20 个 ROI 的固定顺序

```text
aMFC ecMFC ccMFC icMFC pMFC
aLFC ecLFC ccLFC icLFC pLFC
aMTC eMTC pMTC iMTC cMTC
aLTC eLTC pLTC iLTC cLTC
```

其中：

- MFC：medial femoral cartilage
- LFC：lateral femoral cartilage
- MTC：medial tibial cartilage
- LTC：lateral tibial cartilage

---

## 输出指标

v16-final 对每个 ROI 固定输出 4 个主要指标。

### 1. FCL（%）

全层软骨缺损比例：

```text
FCL = 100 × dAB / tAB
```

其中：

- `tAB`：total subchondral bone area
- `cAB`：cartilage-covered subchondral bone area
- `dAB`：denuded subchondral bone area

并满足：

```text
dAB = tAB - cAB
```

---

### 2. Mean Thickness（mm）

平均软骨厚度在最终 `tAB` domain 上计算。

对于没有 cartilage coverage 的 denuded / uncovered tAB 顶点：

```text
thickness = 0
```

因此最终平均厚度反映整个目标软骨下骨区域，而不是只统计仍然存在软骨的区域。

---

### 3. Surface Area（mm²）

主输出中的 Surface Area 定义为：

```text
cAB
```

即被软骨覆盖的软骨下骨表面积。

---

### 4. Volume（mm³）

软骨物理体积按照：

```text
cartilage voxel count × sx × sy × sz
```

计算。

这里使用真实的物理体素体积，而不是 voxel spacing 的向量范数。

---

## 单病例运行

如果已经完成模板配准，并获得：

```text
warped_template_seg.nii.gz
```

可直接运行：

```bash
python morphology_gpu.py \
  --seg /path/to/subject_seg.nii.gz \
  --registration /path/to/warped_template_seg.nii.gz \
  --knee-side right \
  --output /path/to/results/MorphQuant.csv \
  --save-meta-json /path/to/results/MorphQuant_meta.json
```

Windows 示例：

```bat
python morphology_gpu.py ^
  --seg D:\data\subject_seg.nii.gz ^
  --registration D:\data\warped_template_seg.nii.gz ^
  --knee-side right ^
  --output D:\results\MorphQuant.csv ^
  --save-meta-json D:\results\MorphQuant_meta.json
```

v16-final 会在内部固定最终 morphology 配置，因此正常使用时不需要手工再指定 v5–v16 的实验性参数。

默认推荐 CUDA 运行。

`--allow-cpu-fallback` 主要用于调试和测试，不建议用于正式大规模处理。

---

## 单病例输出文件

正常运行后，每个病例目录会产生：

```text
MorphQuant.csv
MorphQuant_Compartments.csv
FCL_Areas.csv
MorphQuant_meta.json
```

### `MorphQuant.csv`

包含：

```text
20 ROI × 4 primary metrics
```

即：

- FCL
- Mean Thickness
- Surface Area
- Volume

---

### `MorphQuant_Compartments.csv`

用于和 Chondrometrics/POMA 做 compartment-level 对比。

当前输出：

```text
MTC
cMFC
LTC
cLFC
```

这些指标是在对应的 union domain 上直接重新计算，而不是把多个存在重叠关系的 ROI 简单求和或平均。

---

### `FCL_Areas.csv`

用于 FCL 审计和方法学检查，包含每个 ROI 的：

```text
tAB
cAB
dAB
FCL
```

---

### `MorphQuant_meta.json`

保存运行元数据，例如：

- voxel size
- knee side
- v16-final morphology mode
- 指标定义
- 运行参数

---

## 批量运行

`batch_morphology.py` 使用 manifest CSV。

推荐格式：

```text
id,image,seg,knee_side
```

例如：

```csv
id,image,seg,knee_side
case001,/data/case001_dess.nii.gz,/data/case001_seg.nii.gz,right
case002,/data/case002_dess.nii.gz,/data/case002_seg.nii.gz,left
```

---

### 已经有 registration，只重新计算 morphology

```bash
python batch_morphology.py \
  --manifest manifests/cmtest.csv \
  --output-root registrations \
  --only-morphology
```

Windows：

```bat
python batch_morphology.py ^
  --manifest manifests\cmtest.csv ^
  --output-root registrations ^
  --only-morphology
```

---

### 如果缺少 registration，则自动执行模板配准

```bash
python batch_morphology.py \
  --manifest manifests/cmtest.csv \
  --template-root templates \
  --output-root registrations
```

模板推荐结构：

```text
templates/
├── KL0_R/
│   ├── KL0_template_dess.nii.gz
│   └── KL0_template_seg.nii.gz
└── KL0_L/
    ├── KL0_template_dess.nii.gz
    └── KL0_template_seg.nii.gz
```

模板到受试者的配准由：

```text
register_unigradicon.py
```

负责调用官方 uniGradICON 工作流。

---

## 代码测试

建议发布前至少运行：

```bash
python -m py_compile \
  morphology_gpu.py \
  batch_morphology.py \
  register_unigradicon.py \
  validation/compare_chondrometrics.py \
  validation/compare_fcl_paper_phr.py \
  test_morphology_gpu.py
```

然后执行：

```bash
python test_morphology_gpu.py
```

v16-final 冻结版本的记录中，上述核心测试均通过。

---

## v16-final 验证结果

### 79 例人工 FCL 验证

使用 20 ROI 和三位人工评分者进行验证后，median-human grade 结果为：

| 指标 | v16-final |
|---|---:|
| Exact grade agreement | 76.3% |
| ±1 grade | 94.4% |
| ±2 grades | 98.4% |
| MAE | 0.3215 grades |
| Spearman | 0.534 |
| QWK | 0.819 |

---

## Paper-style pseudo hit rate（pHR）

为了尽量复现 CartiMorph 论文中的 FCL 验证方法：

- 三位人工 rater 分别独立作为 ground truth
- 人工 grade `0–10` 转换为：

```text
grade × 10%
```

- 在 tolerance `R` 下，如果：

```text
|algorithm FCL - human FCL| <= R
```

则记为 hit。

---

### COMMON16_PAIRED，pHR @ 10%

| Rater | Python v16 | Chondrometrics |
|---|---:|---:|
| Rater 1 | 0.9011 | 0.7136 |
| Rater 2 | 0.9074 | 0.6994 |
| Rater 3 | 0.8900 | 0.7057 |
| **3-rater mean** | **0.8995** | **0.7062** |

---

### ALL20 Python-only，pHR @ 10%

| Rater | Python v16 |
|---|---:|
| Rater 1 | 0.8987 |
| Rater 2 | 0.9234 |
| Rater 3 | 0.9032 |
| **3-rater mean** | **0.9084** |

CartiMorph 原论文报告的 10% tolerance pHR 约为：

```text
0.85
```

完整冻结验证摘要请查看：

```text
docs/V16_FINAL_VALIDATION.txt
```

---

## 已知局限

v16-final 仍保留一个已知问题：

> 在部分 severe disease 病例中，FCL severity 可能存在向低值压缩的现象。

在开发过程中已经测试过多种修改，包括：

- tAB reconstruction 调整
- SCB seed 调整
- closing 调整
- FC augmentation
- inner-surface reconstruction
- projection / rebase 检查
- thickness threshold post-processing

部分修改虽然能提高 severe FCL，但同时会明显破坏正常区域或 human-zero 病例的 specificity。

因此最终没有为了少数 severe outlier 引入新的 v17 morphology rule，而是将该现象作为当前实现的 limitation 保留和报告。

---

## 与 Chondrometrics/POMA 比较

如果本地拥有合法获得的 Chondrometrics/POMA 数据，可以运行：

```bash
python validation/compare_chondrometrics.py \
  --root registrations \
  --chondrometrics /path/to/Chondrometrics_AllMetrics_with_CMT-ID.csv \
  --out comparison_chondrometrics_v16
```

Windows：

```bat
python validation\compare_chondrometrics.py ^
  --root registrations ^
  --chondrometrics Chondrometrics_AllMetrics_with_CMT-ID.csv ^
  --out comparison_chondrometrics_v16
```

---

## COMMON16 ROI

当前 Chondrometrics 数据中可以直接与 CartiMorph ROI 对应的区域有 16 个：

```text
ecMFC ccMFC icMFC
ecLFC ccLFC icLFC

aMTC eMTC pMTC iMTC cMTC
aLTC eLTC pLTC iLTC cLTC
```

以下 4 个 femoral ROI 没有直接对应的 regional Chondrometrics dABp：

```text
aMFC
pMFC
aLFC
pLFC
```

因此比较脚本不会为了凑齐 20 ROI 人为构造或推断这 4 个区域的 Chondrometrics FCL。

---

## Chondrometrics 单位处理

Chondrometrics 中的 `cAB` 如果以 cm² 给出，比较时转换为：

```text
cm² × 100 = mm²
```

v16-final 的结果不会通过经验系数进行缩放，以强行匹配 Chondrometrics。

需要注意：

> CartiMorph-style Python 结果和 Chondrometrics/POMA 来自不同的测量流程和 annotation protocol，因此 Chondrometrics 应作为独立比较方法，而不是要求数值完全一致的目标。

---

## Paper-style FCL 验证

如果本地已经准备好：

```text
Chondrometrics_AllMetrics_with_CMT-ID.csv
FCLgrading_rater1.xlsx
FCLgrading_rater2.xlsx
FCLgrading_rater3.xlsx
```

可以运行：

```bash
python validation/compare_fcl_paper_phr.py \
  --root registrations \
  --mapping /path/to/Chondrometrics_AllMetrics_with_CMT-ID.csv \
  --rater1 /path/to/FCLgrading_rater1.xlsx \
  --rater2 /path/to/FCLgrading_rater2.xlsx \
  --rater3 /path/to/FCLgrading_rater3.xlsx \
  --out human_fcl_v16/paper_phr
```

该脚本按 CartiMorph 论文式规则分别计算三位 rater 的 pHR。

---

## 数据发布建议

本仓库推荐只发布：

```text
代码
文档
测试
聚合统计
```

不要直接提交：

```text
OAI / OAI-ZIB MRI
病例级 NIfTI 文件
segmentation masks
registrations/
warped_template_seg.nii.gz
人工评分原始 Excel
病例级 Chondrometrics/POMA 表格
逐病例预测结果
```

这些内容可能受到数据使用协议、隐私规则或第三方许可限制。

---

## 推荐开源的文件

建议 GitHub 仓库包含：

```text
morphology_gpu.py
batch_morphology.py
register_unigradicon.py
test_morphology_gpu.py
requirements.txt

validation/
    compare_chondrometrics.py
    compare_fcl_paper_phr.py

docs/
    V16_FINAL_VALIDATION.txt

README.md
LICENSE
NOTICE.md
.gitignore
```

聚合后的 validation summary / pHR curve 等文件也可以公开。

---

## 不建议直接开源的数据文件

默认不要上传：

```text
Chondrometrics_AllMetrics_with_CMT-ID.csv
FCLgrading_rater1.xlsx
FCLgrading_rater2.xlsx
FCLgrading_rater3.xlsx

comparison_chondrometrics_v16/region_pairs.csv
comparison_chondrometrics_v16/compartment_pairs.csv

human_fcl_v16/paper_phr/records_long.csv
```

特别是包含：

```text
SubjectID
MRBarCode
病例编号
逐病例人工评分
逐病例算法预测
```

的数据，建议不进入公开 GitHub 主仓库。

---

## Citation

本项目的方法学基础来自 CartiMorph：

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

如果使用了 registration pipeline，也请同时引用 uniGradICON 的相关论文。

上游项目：

```text
https://github.com/YongchengYAO/CartiMorph
https://github.com/YongchengYAO/CartiMorph-Toolbox
https://github.com/uncbiag/uniGradICON
```

---

## License 与署名

CartiMorph 和 CartiMorph-Toolbox 的公开仓库采用 **CC BY-NC 4.0**。

由于本项目属于基于 CartiMorph 方法和公开实现开发的 Python/CUDA 研究复现版本，因此不应默认认为整个项目可以直接重新授权为 MIT、BSD 或 Apache-2.0。

比较保守的公开发布方案是：

- 使用 **CC BY-NC 4.0**
- 保留 CartiMorph 作者和论文署名
- 在 `NOTICE.md` 中明确说明本项目是独立 Python/CUDA implementation
- 对 uniGradICON 等第三方依赖分别注明其原始许可
- 不将第三方源码直接复制进仓库，尽量通过 Python package dependency 安装

如果计划：

- 商业使用
- 闭源商业集成
- 改用 MIT / Apache 等软件许可证

建议先取得原始作者许可或进行正式的许可审查。

---

## Disclaimer

本项目是一个**独立的科研复现实现**：

- 不是 CartiMorph 官方发布版本
- 不代表原 CartiMorph 作者
- 不代表 uniGradICON 作者
- 未获得任何临床医疗器械批准
- 不适用于直接临床诊断

软件仅用于科研、方法复现和算法验证。
