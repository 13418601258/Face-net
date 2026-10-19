# Pipeline scripts (AST-Face) — Step 01 / 04 / 06 / 07

This repository provides a step-by-step processing pipeline for AST-Face.  
This page documents the **callable CLI scripts** for the following steps:

- **Step 01**: `01_icp.py` — ICP alignment to a canonical pose  
- **Step 04**: `04_get_bfm_flame.py` — FLAME-assisted projection to **BFM topology**  
- **Step 06**: `06_bfm_cropper.py` — Crop BFM mesh (keep a vertex subset)  
- **Step 07**: `07_deformation_fields.py` — Compute per-vertex deformation fields  

> Tip: run `python <script>.py --help` to see all arguments.

---

## 01 — ICP alignment (`01_icp.py`)

Align a **source** mesh to a **target** mesh by estimating a similarity transform  
(rotation + translation + optional scale) using ICP on mesh vertices.

### Single pair
```bash
python 01_icp.py \
  --source path/to/src.obj \
  --target path/to/tgt.obj \
  --out path/to/aligned.obj \
  --estimate_scale
```

### Batch mode (align all `.obj` under a folder to one target)
Write results to a new folder:
```bash
python 01_icp.py \
  --source_dir path/to/src_folder \
  --target path/to/tgt.obj \
  --out_dir path/to/aligned_folder \
  --estimate_scale
```

Overwrite files in-place (use with care):
```bash
python 01_icp.py \
  --source_dir path/to/src_folder \
  --target path/to/tgt.obj \
  --inplace \
  --estimate_scale
```

**Output:** aligned OBJ(s) with updated vertex positions.

---

## 04 — FLAME → BFM topology (`04_get_bfm_flame.py`)

Convert a fitted FLAME mesh to a mesh with **BFM topology** using a fixed mapping from BFM vertices
to vertices on a subdivided FLAME mesh.

This script provides two modes:

- **`reconstruct` (recommended)**: use a precomputed `bfm_to_flame_indices.txt` mapping  
- **`optimize_indices` (optional / one-time)**: estimate the mapping by optimization (GPU recommended)

### Reconstruct (recommended pipeline mode)
```bash
python 04_get_bfm_flame.py reconstruct \
  --flame_obj path/to/fitted_flame.obj \
  --bfm_faces_obj path/to/bfm_template.obj \
  --indices_txt data/bfm_to_flame_indices.txt \
  --out_obj result/flame_to_bfm.obj \
  --subdivisions 6
```

**Output:** `result/flame_to_bfm.obj` (BFM topology)

### Optimize indices (optional / one-time)
```bash
python 04_get_bfm_flame.py optimize_indices \
  --bfm_obj path/to/bfm_template.obj \
  --flame_obj path/to/fitted_flame.obj \
  --out_dir result/mapping_estimation \
  --subdivisions 2 \
  --k_neighbors 10 \
  --iterations 800 \
  --device cuda
```

**Outputs:**
- `result/mapping_estimation/bfm_to_flame_indices.txt`
- `result/mapping_estimation/optimized_mesh.obj`

---

## 06 — Crop BFM mesh (`06_bfm_cropper.py`)

Crop a BFM mesh by keeping only a subset of vertices specified in a mapping file.

### Mapping file format (expected)
First line must contain kept 1-based vertex indices:
```
kept_indices: 1,2,3,4,...
```
Optionally, the mapping file can also contain `f ...` face lines.

### Run
```bash
python 06_bfm_cropper.py \
  --input_obj path/to/bfm_full.obj \
  --mapping_txt path/to/cut_bfm.txt \
  --output_obj result/bfm_cropped.obj \
  --use_faces_from_mapping
```

- If `--use_faces_from_mapping` is set and face lines exist in the mapping file, those faces are used.
- Otherwise, faces from the input OBJ are reused and filtered.

**Output:** cropped OBJ with reindexed vertices and faces.

---

## 07 — Deformation fields (`07_deformation_fields.py`)

Compute per-vertex deformation fields relative to neutral:
\[
D = V_{expr} - V_{neutral}
\]
Meshes must have **the same topology and vertex order**.

### Trimesh backend (recommended default; CPU)
```bash
python 07_deformation_fields.py \
  --neutral path/to/neutral.obj \
  --expression path/to/expr.obj \
  --out result/expr_def.npy
```

### PyTorch3D backend (optional; GPU)
```bash
python 07_deformation_fields.py \
  --neutral path/to/neutral.obj \
  --expression path/to/expr.obj \
  --out result/expr_def.npy \
  --backend pytorch3d \
  --device cuda:0
```

**Output:** `*.npy` array with shape `(N, 3)`.

---

## Suggested directory layout (optional)

One simple layout for running these steps:

```
project_root/
  data/
    raw_meshes/
    aligned_meshes/        # step 01 outputs
    landmarks/             # step 02 outputs
    fitted_flame/          # step 03 outputs
    flame_to_bfm/          # step 04 outputs
    edge_nicp_meshes/       # step 05 outputs (topology-unified meshes after Edge-NICP)
    cropped/               # step 06 outputs
    deformation_fields/    # step 07 outputs
  scripts/
    01_icp.py
    04_get_bfm_flame.py
    06_bfm_cropper.py
    07_deformation_fieldsi.py
```

