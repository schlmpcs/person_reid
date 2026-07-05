# MSMT17 Person ReID — Project Handoff

**Last updated:** 2026-07-06
**Working dir:** `/home/amirizimov/MSMT17`
**Owner:** royalkaasdf@gmail.com

---

## 1. Goal

Build a **scalable, real-time person Re-Identification system deployed with NVIDIA DeepStream**
(TensorRT engines, tracker-integrated ReID) — not just an academic retrieval benchmark.
End target: **MTMC** (multi-target multi-camera) tracking.

**Pipeline plan:** PyTorch training → ONNX export → TensorRT engine → DeepStream (NvDCF+ReID tracker) → MTMC.

---

## 2. Current status

| Stage | State |
|-------|-------|
| Dataset acquired | ✅ done |
| Environment (uv + cu128) | ✅ done |
| FastReID torch-2.11 compat patches | ✅ done (6 patches) |
| **Training (SBS R50-IBN, 60 epochs)** | ✅ **done** |
| **ONNX export** | ✅ **done** (dynamic batch, parity cosine 0.999999) |
| TensorRT engine | ⏭️ **NEXT** |
| DeepStream integration | ⬜ pending |
| MTMC | ⬜ pending |

### Final trained-model results (MSMT17 test)

| Rank-1 | Rank-5 | Rank-10 | mAP | mINP | metric |
|:------:|:------:|:-------:|:---:|:----:|:------:|
| 84.56 | 91.78 | 93.52 | 61.83 | 15.95 | 73.20 |

Matches published FastReID SBS-R50-IBN numbers. Progression: ep10 mAP42.7/R1 69.7 → ep20 51.5/77.2 → ep60 61.8/84.6.

---

## 3. Hardware / environment

- **GPU:** RTX 5060, 8 GB VRAM, **Blackwell sm_120**, driver 595 / CUDA 13.2.
  - sm_120 requires **cu128 PyTorch wheels** (older builds won't run on Blackwell).
  - 8 GB is the binding constraint → chose R50-IBN over TransReID; AMP required to fit batch 64.
- **OS:** Linux. No `git`, no `make` on the box (matters for FastReID's cython eval — see patches).
- **Python/env:** `uv`, Python 3.11.15, venv at `/home/amirizimov/MSMT17/.venv`.
  - `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128` (torch reports `cuda 12.8`).
  - Always invoke as `../.venv/bin/python` from inside `fast-reid/`.

---

## 4. Dataset

- **MSMT17_V1** (unblurred). License-restricted; obtained from HF mirror `xianpeijie/MSMT17_V1`.
- Extracted to `/home/amirizimov/MSMT17/MSMT17_V1/` — 4101 IDs, 126441 imgs, 15 cameras.
  - test split: query 3060 IDs / 11659 imgs, gallery 3060 IDs / 82161 imgs.
- Loader env var: **`FASTREID_DATASETS=/home/amirizimov/MSMT17`** (finds `MSMT17_V1/` under it).
- `MSMT17_V1.zip` (2.4 G) is redundant and can be deleted.

---

## 5. Framework & model

- **FastReID** (JDAI-CV), downloaded as a tarball to `/home/amirizimov/MSMT17/fast-reid/` (no git).
- **Backbone:** ResNet-50-IBN (`MODEL.BACKBONE.WITH_IBN: True`).
- **Config:** `fast-reid/configs/MSMT17/sbs_R50-ibn.yml` → inherits `Base-SBS.yml` → `Base-bagtricks.yml`.
  - **SBS ("Stronger BaseLine"):** input 384×128, GeM pooling, CircleSoftmax head, Non-Local blocks,
    CrossEntropy + Triplet loss.
  - Solver: Adam, BASE_LR 0.00035, CosineAnnealingLR, MAX_EPOCH 60, DELAY_EPOCHS 30 (decay starts ep30),
    FREEZE_ITERS 1000, FREEZE_LAYERS [backbone], AMP enabled, CHECKPOINT_PERIOD 20, EVAL_PERIOD 10.

---

## 6. Key files

| Path | What |
|------|------|
| `train_sbs.sh` | Reproducible launch script (sets PYTHONPATH, FASTREID_DATASETS, overrides batch/workers). |
| `train_full.log` | Full training log of the completed 60-epoch run. |
| `PATCHES.md` | Documents all 6 torch-2.11 / py3.11 / no-make compat patches to FastReID. |
| `HANDOFF.md` | This file. |
| `fast-reid/logs/msmt17/sbs_R50-ibn/model_best.pth` | **Trained weights** (295 MB; == `model_final.pth`). |
| `fast-reid/logs/msmt17/sbs_R50-ibn/model_00{19,39}.pth` | Periodic checkpoints. |
| `fast-reid/tools/deploy/` | Export tooling: `onnx_export.py`, `trt_export.py`, `*_inference.py`, `trt_calibrator.py`. |
| memory: `.claude/projects/-home-amirizimov-MSMT17/memory/msmt17-reid-project.md` | Persistent project memory. |

---

## 7. FastReID patches applied (torch 2.11 / py3.11 / no-make)

Full detail in `PATCHES.md`. Summary:

1. `fastreid/data/build.py` — `collections.Mapping` → `collections.abc.Mapping`.
2. `fastreid/evaluation/testing.py` — same `Mapping` split.
3. `fastreid/evaluation/rank.py` — `np.bool` → `bool` (removed in NumPy 1.24+).
4. `fastreid/engine/train_loop.py` (AMPTrainer) — modernized AMP API to `torch.amp.GradScaler("cuda")` /
   `torch.amp.autocast("cuda")`.
5. `fastreid/engine/defaults.py` (`build_optimizer`) — pass **`contiguous=False`**. ★ Root-cause fix for the
   AMP "No inf checks were recorded for this optimizer" crash: ContiguousParams pre-wires each param's `.grad`
   as a view into one buffer, but torch≥2.0 `zero_grad(set_to_none=True)` nulls those views → scaler sees zero
   grads. Disabling the (pure-speed) ContiguousParams fixes it.
6. `fastreid/evaluation/reid_evaluation.py` (`_compile_dependencies`) — wrapped cython `compile_helper()`
   (shells out to `make`, not installed) in try/except → falls back to pure-python ranking in `rank.py`
   (`evaluate_py`, auto-selected via `IS_CYTHON_AVAI`). Correct, just slower.

---

## 8. How to reproduce / re-run

**Train:**
```bash
bash /home/amirizimov/MSMT17/train_sbs.sh
```

**Evaluate an existing checkpoint** (from `fast-reid/`, with env exported as in `train_sbs.sh`):
```bash
../.venv/bin/python tools/train_net.py \
  --config-file configs/MSMT17/sbs_R50-ibn.yml --eval-only \
  MODEL.WEIGHTS logs/msmt17/sbs_R50-ibn/model_best.pth \
  TEST.IMS_PER_BATCH 128 OUTPUT_DIR logs/msmt17/sbs_R50-ibn
```

Required env for any run: `cd fast-reid && export PYTHONPATH=. FASTREID_DATASETS=/home/amirizimov/MSMT17`,
use `../.venv/bin/python`.

---

## 9. ONNX export — DONE (2026-07-06)

- **Artifact:** `fast-reid/logs/msmt17/sbs_R50-ibn/onnx/sbs_R50-ibn.onnx` (94.1 MB, fp32).
- **Exporter:** `fast-reid/tools/deploy/onnx_export_v2.py` (new, torch-2.11-compatible). The stock
  `onnx_export.py` is **broken** on torch 2.11 — `torch.onnx.export` dropped `operator_export_type`
  (`ONNX_ATEN_FALLBACK`), and onnxoptimizer/onnxsim weren't installed. v2 uses the legacy TorchScript
  exporter (`dynamo=False`), named I/O, dynamic batch, onnxsim, and an onnxruntime parity check.
- **Deps installed** into venv: `onnx 1.22`, `onnxruntime 1.27`, `onnxsim 0.6.5`.
- **Graph I/O:** input `input` `[batch,3,384,128]` fp32 (dynamic batch); output `output` `[batch,2048]`
  fp32 (dynamic batch). opset 17, ir 8.
- **Parity vs PyTorch:** cosine_min **0.999999** on realistic [0,255] input → PARITY OK.
- Re-run: `../.venv/bin/python tools/deploy/onnx_export_v2.py --config-file configs/MSMT17/sbs_R50-ibn.yml
  --name sbs_R50-ibn --output logs/msmt17/sbs_R50-ibn/onnx --batch-size 8 --opts MODEL.WEIGHTS
  logs/msmt17/sbs_R50-ibn/model_best.pth` (with the standard env exported).

### ⚠️ Preprocessing contract (must match in TensorRT/DeepStream)

- **Input = raw RGB in [0, 255], NCHW, 384×128 (H×W).** ImageNet mean/std normalization is **baked into
  the ONNX graph** (`preprocess_image` does `sub_(mean).div_(std)` with mean `[0.485,0.456,0.406]*255`,
  std `[0.229,0.224,0.225]*255`). So the DeepStream preprocess must feed 0–255 RGB and **NOT** pre-scale
  to [0,1] or re-normalize. (In `nvinfer`: `net-scale-factor=1.0`, no offsets, `model-color-format=0` RGB.)
- **Output embedding is NOT L2-normalized** (the eval head returns raw `neck_feat`). The FastReID
  evaluator normalizes separately, which is why mAP was correct. For DeepStream ReID / cosine matching,
  add an L2-normalize step (tracker config, or bake a `Normalize` op into the graph before TRT).
- `heads.weight` "skip loading" warning during export is **expected/benign** (classifier head, unused for
  embeddings).

## 9b. Remaining steps (in order)

1. **TensorRT engine** — `trt_export.py` or `trtexec`. FP16 is the safe first target on Blackwell sm_120.
   Use dynamic-batch optimization profile (min/opt/max). INT8 later (needs calibration via
   `trt_calibrator.py`). Confirm TensorRT version available on the box first.
2. **DeepStream** — wire the engine into an **NvDCF + ReID** tracker config (or standalone SGIE ReID).
   Honor the preprocessing contract above (0–255 RGB, no rescale) and add L2-norm on the embedding.
3. **MTMC** — cross-camera association on top of per-camera tracks.

---

## 10. Metric glossary (for reading eval tables)

- **Rank-1/5/10 (CMC):** % of queries whose top-1/5/10 ranked gallery image is the same person (different
  camera). Rank-1 is what the DeepStream tracker cares about (closest embedding = right identity).
- **mAP:** mean Average Precision — how well *all* correct matches for each query are ranked high. Stricter
  than Rank-1; the field's headline number.
- **mINP:** stricter "hardest correct match" metric; small values are normal.
- **metric:** FastReID's convenience blend `(mAP + Rank-1)/2`, used to pick `model_best.pth`.
