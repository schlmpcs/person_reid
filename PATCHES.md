# FastReID patches for PyTorch 2.11 / Python 3.11 / Blackwell (sm_120)

FastReID was written for torch 1.x / py3.7. These edits make it run on this box
(torch 2.11.0+cu128, Python 3.11.15, RTX 5060). Applied under `fast-reid/`.

1. **`fastreid/data/build.py`** — `from collections import Mapping` → `from collections.abc import Mapping` (moved in py3.10).

2. **`fastreid/evaluation/testing.py`** — split `from collections import Mapping, OrderedDict`
   into `OrderedDict` (from `collections`) + `Mapping` (from `collections.abc`).

3. **`fastreid/evaluation/rank.py`** — `np.bool` → `bool` (removed in NumPy 1.24+).

4. **`fastreid/engine/train_loop.py`** (AMPTrainer) — modernized AMP API:
   `torch.cuda.amp.GradScaler()` → `torch.amp.GradScaler("cuda")`,
   `torch.cuda.amp.autocast()` → `torch.amp.autocast("cuda")`.

5. **`fastreid/engine/defaults.py`** (`build_optimizer`) — pass `contiguous=False`.
   ROOT CAUSE of the AMP "No inf checks were recorded for this optimizer" crash:
   ContiguousParams pre-wires each param's `.grad` as a view into one contiguous
   buffer, but torch>=2.0 `zero_grad(set_to_none=True)` nulls those views, so
   backward allocates fresh per-param grads and the buffer's `.grad` stays None →
   the AMP scaler sees zero grads. Disabling ContiguousParams (a pure speed
   optimization) fixes it. Verified: opt_params_with_grad 0/2 → 238/238.

6. **`fastreid/evaluation/reid_evaluation.py`** (`_compile_dependencies`) — wrapped
   the cython `compile_helper()` (which shells out to `make`, not installed) in
   try/except so it falls back to the pure-python ranking in `rank.py` (`evaluate_py`,
   selected automatically via `IS_CYTHON_AVAI`). Correct results, just slower ranking.

## Environment
- venv: `/home/amirizimov/MSMT17/.venv` (uv, Python 3.11)
- `FASTREID_DATASETS=/home/amirizimov/MSMT17` (loader finds `MSMT17_V1/`)
- dataset: `/home/amirizimov/MSMT17/MSMT17_V1/`
