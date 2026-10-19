# AST-Face Processing Pipeline

This repository provides the processing pipeline code for the **AST-Face dataset**, covering the complete workflow from raw 3D face scans to topology-unified meshes with consistent vertex correspondence, as well as the generation of deformation fields for downstream applications.

- **Dataset (public tier)**: [OSF Repository](https://osf.io/xk4f6/) (anonymized, non-textured derived data and annotations)
- **Controlled-access data** (raw scans; textured meshes and synchronized RGB images where available): requires signing a **Data Usage Agreement (DUA)**. See **Applying for Controlled-Access Data** below.

---

## 🚀 Pipeline Overview

The pipeline corresponds to the workflow described in the AST-Face paper. Each script/module implements one major step:

- `01_icp.py` – ICP alignment of raw scans to a canonical pose
- `02_landmarks/` – Extract 84 facial landmarks using [Deep-MVLM](https://github.com/RasmusRPaulsen/Deep-MVLM)
- `03_flame_fitting/` – Fit FLAME models using [flame-fitting](https://github.com/Rubikplayer/flame-fitting)
- `04_get_bfm_flame.py` – **FLAME-assisted Projection**: convert fitted FLAME meshes to BFM topology through subdivision & projection
- `05_edge_nicp/` – Perform non-rigid registration with [edge-nicp](https://github.com/zhaopu99/edge-nicp) to unify topology and vertex count
- `06_bfm_cropper.py` – Crop processed meshes
- `07_deformation_fields.py` – Compute per-vertex deformation fields relative to neutral scans

---

## 📦 Final Outputs (Public Tier)

The public release includes anonymized, non-textured derived data and annotations, including:

- **Topology-standardized meshes** (unified topology, identical vertex count)
- **84-point anatomical landmarks**
- **Deformation fields**
- **AU / expression metadata** (see OSF release files)

> **Note:** Raw scans are **not** part of the public tier.

---

## ⚙️ Dependencies & Environment

### Core dependencies
- Python 3.9+
- `numpy`
- `scipy`
- `pytorch3d`


> Some stages (especially FLAME fitting and Edge-NICP topology unification) require a CUDA-enabled PyTorch3D installation and a compatible PyTorch/CUDA version stack.

---

## ▶️ Running the Pipeline (Typical Order)

A typical processing sequence is:

1. `01_icp.py`
2. `02_landmarks/`
3. `03_flame_fitting/`
4. `04_get_bfm_flame.py`
5. `05_edge_nicp/`
6. `06_bfm_cropper.py`
7. `07_deformation_fields.py`

Please refer to the per-stage documentation/scripts for expected inputs, outputs, and command-line examples.

---

## 🔒 Applying for Controlled-Access Data (Raw Scans, Textures & RGB Images)

The AST-Face dataset includes a **controlled-access tier** containing:

- **Raw 3D facial scans** (controlled due to residual re-identification risk from high-resolution facial geometry)
- **Identifiable modalities**, including textured meshes and synchronized multi-view RGB images (frontal / left / right), available for the **52 participants** who consented to sharing these identifiable modalities

Controlled-access data are hosted in restricted OSF components (currently split as **Controlled_Data(1)** and **Controlled_Data(2)** due to storage limits).

### How to request access
1. **Create an OSF account** (required for permission assignment).
2. Download the **Data Usage Agreement (DUA)** from the OSF project page.  
   - The **OSF-hosted DUA is the authoritative version**.
   - A copy may also be mirrored in this GitHub repository for convenience.
3. Read, complete, and sign the DUA.
4. Send the signed DUA to **zhaoyaopu@buaa.edu.cn**  

### How access is granted
After identity verification and confirmation of the signed DUA, access is granted via **OSF’s permission mechanism** by adding the requester as a **read-only contributor** to the restricted OSF components.

- We **do not** distribute controlled data via emailed download links.
- Verification is limited to **identity confirmation** and **DUA acceptance**.
- Access is **not** screened by research topic, institution type, or nationality.

---

## 🌐 Data Access Summary

### Public tier (open download, OSF)
Includes anonymized, non-textured derived data and annotations such as:
- topology-standardized meshes
- deformation fields
- landmarks
- AU / expression metadata

### Controlled-access tier (DUA required, OSF restricted components)
Includes:
- raw scans
- textured meshes (where available; consenting subset)
- synchronized multi-view RGB images (where available; consenting subset)

---

## 📘 Quick Trial / Demo (Optional)

If a Colab notebook is provided in this repository, it is intended for **rapid exploration of the public subset**, such as:
- loading topology-unified meshes
- checking vertex correspondence consistency
- computing/visualizing deformation fields
- inspecting landmarks and AU/expression annotations

The full FLAME fitting and Edge-NICP stages are kept as documented scripts due to environment constraints (CUDA/PyTorch3D compatibility) that may be difficult to maintain reliably in Colab.

---

## 🔗 External Repositories

- **Landmarks (84 pts):** [Deep-MVLM](https://github.com/RasmusRPaulsen/Deep-MVLM)
- **FLAME fitting:** [flame-fitting](https://github.com/Rubikplayer/flame-fitting)
- **Edge-NICP (our implementation):** [edge-nicp](https://github.com/zhaopu99/edge-nicp)

---

## 🙏 Acknowledgements

We sincerely thank the authors of the following open-source projects for their contributions to the community, which were essential for building the AST-Face processing pipeline:

- **Deep-MVLM** – for multi-view landmark detection and extraction of 84 anatomical facial landmarks
- **FLAME-fitting** – for FLAME model fitting, enabling robust expression capture and geometry alignment

Their excellent work provided the foundation for key components of AST-Face.

---

## ⚠️ Disclaimer

This code is provided for **academic research** and **reproducibility** of the AST-Face paper.

Users are responsible for:
- complying with dataset access terms (including the DUA for controlled-access data)
- following applicable ethical and legal requirements
- ensuring secure handling of controlled-access data

Any misuse of the dataset or code is solely the responsibility of the user.