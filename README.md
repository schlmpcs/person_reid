# Person Re-Identification — MSMT17 → DeepStream

Training and deployment of a person **Re-Identification (ReID)** model on **MSMT17**, targeting a
scalable, real-time **NVIDIA DeepStream** pipeline (TensorRT + tracker-integrated ReID), with
**MTMC** (multi-target multi-camera) tracking as the end goal.

The model is a FastReID **SBS ResNet-50-IBN** embedding extractor: an image crop → a 2048-D appearance
vector used for cross-camera identity matching.

**Trained result (MSMT17 test):** Rank-1 **84.56** · mAP **61.83** (matches published FastReID numbers).

> Full design write-up: [`ARCHITECTURE.md`](ARCHITECTURE.md) ·
> current status & repro commands: [`HANDOFF.md`](HANDOFF.md) ·
> framework compatibility fixes: [`PATCHES.md`](PATCHES.md).

---

## Pipeline

```
MSMT17 → FastReID SBS R50-IBN (PyTorch) → ONNX → TensorRT (FP16) → DeepStream (NvDCF+ReID) → MTMC
   ✅            ✅ trained                   ✅        ⏭️ next          ⬜               ⬜
```

## What's in this repo

| Path | What |
|---|---|
| `ARCHITECTURE.md` | Full end-to-end architecture (model + pipeline + preprocessing contract). |
| `HANDOFF.md` | Status, results, and exact reproduction commands. |
| `PATCHES.md` | The 6 FastReID edits needed for torch 2.11 / py3.11 / no-`make`. |
| `configs/MSMT17/sbs_R50-ibn.yml` | The training/model config. |
| `scripts/train_sbs.sh` | Reproducible training launch script. |
| `tools/deploy/onnx_export_v2.py` | torch-2.11-compatible ONNX exporter (dynamic batch + parity check). |
| `patches/fastreid/**` | The 6 modified FastReID source files (Apache-2.0), as drop-in replacements. |
| `requirements-freeze.txt` | Full pinned environment (`pip freeze`). |

**Not included** (obtain separately — see below): the MSMT17 dataset (license-restricted), the FastReID
framework itself, the `.venv`, and binary artifacts (trained `.pth` weights ~295 MB, exported `.onnx`
~94 MB — both exceed GitHub's file limits; regenerate them with the scripts here).

## Setup / reproduce

1. **GPU/driver:** built for RTX 5060 (Blackwell **sm_120**), driver 595 / CUDA 13.2. sm_120 requires
   **cu128** PyTorch wheels.
2. **Environment** (`uv`, Python 3.11):
   ```bash
   uv venv .venv --python 3.11
   uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
   uv pip install --python .venv/bin/python onnx onnxruntime onnxsim
   # plus FastReID's own deps; see requirements-freeze.txt for exact pins
   ```
3. **FastReID:** download the [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid) framework into
   `fast-reid/`, then overlay the files in `patches/fastreid/**` onto `fast-reid/fastreid/**`
   (drop-in replacements — see `PATCHES.md` for what each changes and why). Copy `scripts/`,
   `tools/deploy/onnx_export_v2.py`, and `configs/MSMT17/sbs_R50-ibn.yml` into the corresponding
   FastReID locations.
4. **Dataset:** obtain **MSMT17_V1** (license-restricted) and extract to `MSMT17_V1/`. Set
   `FASTREID_DATASETS` to the parent directory.
5. **Train:** `bash scripts/train_sbs.sh`
6. **Export to ONNX:**
   ```bash
   cd fast-reid && export PYTHONPATH=. FASTREID_DATASETS=/path/to/parent
   ../.venv/bin/python tools/deploy/onnx_export_v2.py \
     --config-file configs/MSMT17/sbs_R50-ibn.yml \
     --name sbs_R50-ibn --output logs/msmt17/sbs_R50-ibn/onnx --batch-size 8 \
     --opts MODEL.WEIGHTS logs/msmt17/sbs_R50-ibn/model_best.pth
   ```

## Deployment contract (critical)

The ONNX/TensorRT graph **includes pixel normalization**. Downstream:
- **Input:** raw RGB in **[0, 255]**, NCHW, **384×128** — do *not* pre-scale to [0,1] or re-normalize
  (`nvinfer`: `net-scale-factor=1.0`, RGB, no offsets).
- **Output:** 2048-D embedding, **not L2-normalized** — add an L2-normalize step before cosine matching.

See `ARCHITECTURE.md` §6 for the full contract.

## Licensing

FastReID is Apache-2.0; the files under `patches/` are modified copies redistributed under that license.
The MSMT17 dataset is license-restricted and is **not** distributed here.
