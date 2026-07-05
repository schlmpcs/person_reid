# System Architecture — MSMT17 Person Re-Identification for DeepStream

**Project:** Scalable, real-time person Re-Identification (ReID) deployed with NVIDIA DeepStream.
**End goal:** Multi-Target Multi-Camera (MTMC) tracking.
**Working dir:** `/home/amirizimov/MSMT17`  ·  **Last updated:** 2026-07-06

This document describes the *full architecture* end-to-end: the deep-learning model, the training
recipe, and the deployment pipeline (PyTorch → ONNX → TensorRT → DeepStream → MTMC). For task status
and reproduction commands see `HANDOFF.md`; for framework compat fixes see `PATCHES.md`.

---

## 1. What ReID is (and what the model actually produces)

Person ReID answers: *"is the person in image A the same individual as in image B, seen by a different
camera?"* The model is an **embedding extractor** — it maps a cropped person image to a fixed-length
vector (here **2048-D**) such that crops of the *same* identity land close together and *different*
identities land far apart, under a distance metric (cosine / Euclidean).

There is **no classifier at inference time.** The classification head exists only to shape the embedding
space during training. At deploy time we keep only: *image → backbone → pooling → BNNeck → 2048-D vector.*

Downstream, a tracker compares these vectors:
- **Single camera:** associate detections across frames (short-term appearance memory).
- **Cross camera (MTMC):** match tracklets between cameras by embedding similarity.

---

## 2. End-to-end pipeline (bird's-eye view)

```
┌─────────────┐   train    ┌──────────────┐  export   ┌──────────┐  build   ┌───────────────┐
│  MSMT17_V1  │ ─────────▶ │  FastReID    │ ────────▶ │  ONNX    │ ───────▶ │ TensorRT      │
│  dataset    │            │  SBS R50-IBN │           │ (opset17)│          │ engine (FP16) │
└─────────────┘            │  (PyTorch)   │           └──────────┘          └───────────────┘
                           └──────────────┘                                        │
                                                                                   │ load
                                                                                   ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │                              NVIDIA DeepStream pipeline                                 │
   │                                                                                        │
   │  camera(s) → [decode] → PGIE: person detector → tracker (NvDCF + ReID SGIE) → tracks   │
   │                                                     │                                   │
   │                                    per-object 2048-D embedding (this model)            │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                              ┌────────────────────────┐
                                              │  MTMC association       │
                                              │  (cross-camera match)   │
                                              └────────────────────────┘
```

**Status:** dataset ✅ · training ✅ · ONNX ✅ · TensorRT ⏭️ next · DeepStream ⬜ · MTMC ⬜.

---

## 3. Model architecture — FastReID "SBS" ResNet-50-IBN

Config: `fast-reid/configs/MSMT17/sbs_R50-ibn.yml` → `Base-SBS.yml` → `Base-bagtricks.yml`.
**SBS = "Stronger BaseLine"**, FastReID's high-accuracy recipe. Data flow:

```
 Input image  [N, 3, 384, 128]  (raw RGB, 0–255)
      │
      ▼  ── normalization is PART of the model (baked into ONNX) ──
 (x − mean)/std   mean=[0.485,0.456,0.406]×255   std=[0.229,0.224,0.225]×255
      │
      ▼
 ┌──────────────────────── Backbone: ResNet-50 + IBN + Non-Local ────────────────────────┐
 │  stem  (7×7 conv, s2 → maxpool)                                    → [N,  64, 96, 32]  │
 │  layer1  (3× Bottleneck, IBN)                                      → [N, 256, 96, 32]  │
 │  layer2  (4× Bottleneck, IBN)     + 2 Non-Local blocks            → [N, 512, 48, 16]  │
 │  layer3  (6× Bottleneck, IBN)     + 3 Non-Local blocks            → [N,1024, 24,  8]  │
 │  layer4  (3× Bottleneck, IBN)     LAST_STRIDE=1 (no downsample)   → [N,2048, 24,  8]  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
      │  feature map [N, 2048, 24, 8]
      ▼
 GeM pooling (GeneralizedMeanPoolingP, learnable p)                  → [N, 2048, 1, 1]
      │
      ▼
 BNNeck (BatchNorm1d, bias frozen)                                   → [N, 2048]   ← "neck_feat"
      │
      ├───────────────── INFERENCE: return neck_feat  ▶ 2048-D EMBEDDING (this is the ONNX output)
      │
      └── TRAINING ONLY: CircleSoftmax classifier head → 1041-class logits → loss
```

### 3.1 Backbone: ResNet-50 (the feature extractor)

Standard 4-stage ResNet-50 (`[3,4,6,3]` Bottleneck blocks). Two ReID-specific modifications:

- **`LAST_STRIDE = 1`** — the usual stride-2 downsample in `layer4` is removed, so the final feature
  map stays at `24×8` instead of `12×4`. Higher spatial resolution → finer appearance detail, which
  matters for distinguishing people who look similar. Doubles the last-stage compute; it's the standard
  ReID trade-off.

### 3.2 IBN — Instance-Batch Normalization (`WITH_IBN: True`)

Each early Bottleneck's first norm is an **IBN** module (`fastreid/layers/batch_norm.py`): it splits the
channels in half, sends one half through **InstanceNorm** and the other through **BatchNorm**, then
concatenates.

- **InstanceNorm** removes per-image contrast/color/illumination style → the model generalizes across
  cameras with different lighting/white-balance (crucial for MSMT17's 15 cameras and for MTMC).
- **BatchNorm** preserves discriminative content.

> ⚠️ **Export note:** InstanceNorm always uses per-instance statistics (it has no running stats), so it
> behaves identically in train/eval. During ONNX export torch prints an
> `instance_norm ... train=True` warning — this is **benign**; parity was verified (cosine 0.999999).

### 3.3 Non-Local blocks (`WITH_NL: True`)

Self-attention blocks inserted into `layer2` (×2) and `layer3` (×3) — counts `[0,2,3,0]`. Each block lets
every spatial position attend to all others, capturing long-range dependencies (e.g. relating a bag to a
shoe across the body). Improves robustness to partial occlusion and pose change.

### 3.4 GeM pooling (`GeneralizedMeanPoolingP`)

Generalized-mean pooling collapses the `24×8` map to a single vector via
`f(X) = (mean(X^p))^(1/p)` with a **learnable exponent `p`** (`fastreid/layers/pooling.py`).
- `p=1` ⇒ average pooling; `p→∞` ⇒ max pooling. Learning `p` interpolates between them, emphasizing the
  most salient regions without discarding context. Standard for strong ReID baselines.

### 3.5 BNNeck (`WITH_BNNECK: True`, `NECK_FEAT: after`)

A `BatchNorm1d` layer (bias frozen) between the pooled feature and the classifier. It decouples the two
training objectives that pull the embedding in different directions:
- **Triplet loss** wants features good for *metric distance* (measured **before** BNNeck).
- **Classification loss** wants features good for *linear separation* (measured **after** BNNeck).

`NECK_FEAT: after` ⇒ **the deployed embedding is the post-BNNeck vector.** This is what the ONNX graph
outputs.

### 3.6 Classification head — **training only** (`CircleSoftmax`)

`neck_feat → F.linear → CircleSoftmax` over MSMT17's **1041 training identities** (`scale=64,
margin=0.35`). Circle loss is an angular-margin softmax that enforces a large angular gap between
identities on the unit hypersphere. **Stripped at export** — hence the benign
`Skip loading parameter 'heads.weight'` warning (the `1041×2048` classifier isn't used for embeddings).

---

## 4. Training recipe (completed)

| Aspect | Setting |
|---|---|
| Input size | 384×128 (H×W) |
| Losses | CrossEntropy (label smooth ε=0.1) + **Triplet** (hard mining, margin 0) + Circle head |
| Optimizer | Adam, base LR 3.5e-4, weight decay 5e-4 |
| Schedule | Cosine annealing, 60 epochs, decay starts epoch 30, 2000-iter warmup |
| Freeze | backbone frozen for first 1000 iters |
| Precision | **AMP** (mixed FP16/FP32) — required to fit batch 64 in 8 GB |
| Sampler | 16 instances per identity (needed for triplet mining) |

**Result (MSMT17 test):** Rank-1 **84.56** · Rank-5 91.78 · Rank-10 93.52 · mAP **61.83** · mINP 15.95.
Matches published FastReID SBS-R50-IBN numbers.

*(Two loss types work together: triplet shapes the metric space directly; classification provides a
stable global-structure signal. This complementary pairing is why "bag of tricks" baselines are strong.)*

---

## 5. Deployment pipeline

### 5.1 PyTorch → ONNX ✅ (done 2026-07-06)

- **Exporter:** `fast-reid/tools/deploy/onnx_export_v2.py` (custom, torch-2.11 compatible; the stock
  `onnx_export.py` is broken on torch 2.11 — `operator_export_type` was removed from `torch.onnx.export`).
- **Artifact:** `fast-reid/logs/msmt17/sbs_R50-ibn/onnx/sbs_R50-ibn.onnx` (94 MB, fp32).
- **Graph I/O** — opset 17, IR 8, **dynamic batch**:

  | Tensor | Shape | dtype | Notes |
  |---|---|---|---|
  | `input` | `[batch, 3, 384, 128]` | fp32 | raw RGB **0–255**, NCHW |
  | `output` | `[batch, 2048]` | fp32 | embedding, **not** L2-normalized |

- **Validation:** onnxruntime vs PyTorch cosine similarity **0.999999** on realistic input → PARITY OK.

### 5.2 ONNX → TensorRT ⏭️ (next)

- Build with `trtexec` or `fast-reid/tools/deploy/trt_export.py`.
- **FP16** first (safe on Blackwell sm_120); INT8 later needs calibration (`trt_calibrator.py`).
- Use a **dynamic-batch optimization profile** (min/opt/max) since the ONNX has a dynamic batch axis.
- Confirm the installed TensorRT version on the box before building.

### 5.3 TensorRT → DeepStream ⬜

- Person **detector** as PGIE (primary GIE); this ReID engine as an **SGIE / NvDCF ReID** feature
  extractor on each detected person.
- Tracker: **NvDCF + ReID** (appearance-aware) for robust single-camera tracking.

### 5.4 DeepStream → MTMC ⬜

- Cross-camera association of per-camera tracklets using the 2048-D embeddings (cosine similarity +
  spatio-temporal constraints).

---

## 6. ⚠️ The preprocessing/postprocessing contract (deployment-critical)

The ONNX/TensorRT graph is **not** a bare backbone — it embeds the pixel normalization. Getting this
wrong silently destroys accuracy (embeddings look plausible but match poorly).

**INPUT — feed raw RGB in [0, 255], size 384×128, NCHW. Do NOT pre-scale or re-normalize.**
The graph itself does `(x − mean)/std` with ImageNet mean/std (×255). In DeepStream `nvinfer`:
- `net-scale-factor=1.0`  (no `1/255` scaling)
- no channel offsets / mean file
- `model-color-format=0` (RGB)
- `infer-dims=3;384;128`

**OUTPUT — the 2048-D embedding is NOT L2-normalized.** The FastReID evaluator normalizes separately
(which is why the reported mAP is correct). For cosine matching in the tracker you must **add an
L2-normalize** step — either in the tracker's ReID config, or by baking a `Normalize` op into the graph
before the TensorRT build.

---

## 7. Hardware & environment (constraints that shaped the design)

- **GPU:** RTX 5060, **8 GB VRAM**, Blackwell **sm_120**, driver 595 / CUDA 13.2.
  - 8 GB is the binding constraint → **R50-IBN chosen over TransReID**; AMP required for batch 64.
  - sm_120 requires **cu128** PyTorch wheels (older builds won't run on Blackwell).
- **Software:** `uv` venv at `.venv`, Python 3.11.15, `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`.
  ONNX stack: `onnx 1.22`, `onnxruntime 1.27`, `onnxsim 0.6.5`.
- **Box quirks:** no `git`, no `make` → FastReID needed 6 compat patches (see `PATCHES.md`), including a
  pure-Python eval fallback and a ContiguousParams fix for the AMP GradScaler crash.

---

## 8. Repository map (architecture-relevant)

| Path | Role |
|---|---|
| `MSMT17_V1/` | Dataset: 4101 IDs / 126441 imgs / 15 cameras (train 1041 IDs). |
| `fast-reid/` | FastReID framework (JDAI-CV, tarball, no git). |
| `fast-reid/configs/MSMT17/sbs_R50-ibn.yml` | Model + training config (this architecture). |
| `fast-reid/fastreid/modeling/backbones/resnet.py` | ResNet + IBN + Non-Local assembly. |
| `fast-reid/fastreid/layers/batch_norm.py` | `IBN` module (IN/BN split). |
| `fast-reid/fastreid/layers/pooling.py` | `GeneralizedMeanPoolingP` (GeM). |
| `fast-reid/fastreid/modeling/heads/embedding_head.py` | BNNeck + CircleSoftmax head; eval returns embedding. |
| `fast-reid/fastreid/modeling/meta_arch/baseline.py` | Top-level model; **in-graph normalization** in `preprocess_image`. |
| `fast-reid/tools/deploy/onnx_export_v2.py` | torch-2.11 ONNX exporter (custom). |
| `fast-reid/logs/msmt17/sbs_R50-ibn/model_best.pth` | Trained weights. |
| `fast-reid/logs/msmt17/sbs_R50-ibn/onnx/sbs_R50-ibn.onnx` | Exported embedding model. |
| `HANDOFF.md` · `PATCHES.md` · `ARCHITECTURE.md` | Status/repro · compat fixes · this file. |
