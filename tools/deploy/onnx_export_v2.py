# encoding: utf-8
"""
Torch-2.11-compatible ONNX exporter for FastReID SBS models.

Differences from the stock tools/deploy/onnx_export.py:
  * Uses the legacy TorchScript exporter (dynamo=False). The stock script passed
    operator_export_type=ONNX_ATEN_FALLBACK, which was removed from
    torch.onnx.export in torch 2.9+.
  * Exports with named I/O ("input"/"output") and a dynamic batch axis, which is
    what TensorRT / DeepStream want.
  * Simplifies with onnx-simplifier (optional) and validates numerical parity
    against PyTorch using onnxruntime.

Notes for downstream (DeepStream/TensorRT):
  * Input is raw RGB in [0, 255], NCHW, size = cfg.INPUT.SIZE_TEST (H, W).
    Pixel mean/std normalization is baked INTO the graph.
  * Output is the eval-time embedding (neck_feat), NOT L2-normalized.
"""

import argparse
import logging
import os
import sys

import numpy as np
import onnx
import torch

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.file_io import PathManager
from fastreid.utils.logger import setup_logger

setup_logger(name="fastreid")
logger = logging.getLogger("fastreid.onnx_export")


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Convert a FastReID model to ONNX (torch 2.11 compatible)")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--name", default="baseline", help="name for the converted model")
    parser.add_argument("--output", default="onnx_model", help="directory to save the onnx model")
    parser.add_argument("--opset", default=17, type=int, help="onnx opset version")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size used for the trace / parity check")
    parser.add_argument("--no-simplify", action="store_true", help="skip onnx-simplifier")
    parser.add_argument("--static-batch", action="store_true", help="export a fixed batch size (no dynamic axis)")
    parser.add_argument(
        "--opts", default=[], nargs=argparse.REMAINDER,
        help="Modify config options using the command-line 'KEY VALUE' pairs",
    )
    return parser


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        return model
    inputs = model.graph.input
    name_to_input = {inp.name: inp for inp in inputs}
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    return model


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    if cfg.MODEL.HEADS.POOL_LAYER == 'FastGlobalAvgPool':
        cfg.MODEL.HEADS.POOL_LAYER = 'GlobalAvgPool'

    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    if hasattr(model.backbone, 'deploy'):
        model.backbone.deploy(True)
    model.eval()

    H, W = cfg.INPUT.SIZE_TEST
    logger.info(f"Input (H, W) = ({H}, {W}); expects raw RGB in [0,255], NCHW.")

    device = model.device
    # NOTE: preprocess_image() normalizes in-place (sub_/div_), so every consumer
    # gets its OWN clone and we keep a pristine copy for the onnxruntime parity check.
    # Use a realistic raw-RGB [0,255] input: a randn input is wildly OOD (post-norm
    # ~-2.1 everywhere) and amplifies harmless per-op FP noise into a false alarm.
    g = torch.Generator(device="cpu").manual_seed(0)
    dummy = (torch.rand(args.batch_size, 3, H, W, generator=g) * 255.0).to(device)
    input_np = dummy.clone().cpu().numpy()  # pristine, un-normalized input

    # Reference PyTorch output for parity check.
    with torch.no_grad():
        torch_out = model(dummy.clone()).cpu().numpy()
    logger.info(f"Output embedding shape = {torch_out.shape}")

    dynamic_axes = None if args.static_batch else {"input": {0: "batch"}, "output": {0: "batch"}}

    PathManager.mkdirs(args.output)
    save_path = os.path.join(args.output, args.name + ".onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy.clone(),
            save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,
        )
    logger.info(f"Raw ONNX written to {save_path}")

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    if not args.no_simplify:
        try:
            from onnxsim import simplify
            onnx_model, ok = simplify(onnx_model)
            assert ok, "onnxsim could not validate the simplified model"
            logger.info("Simplified ONNX graph with onnx-simplifier")
        except Exception as e:
            logger.warning(f"Skipping simplify ({type(e).__name__}: {e})")

    onnx_model = remove_initializer_from_input(onnx_model)
    onnx.save_model(onnx_model, save_path)
    logger.info(f"Final ONNX model saved to {save_path}")

    # ---- Parity check against PyTorch via onnxruntime ----
    import onnxruntime as ort
    sess = ort.InferenceSession(save_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["output"], {"input": input_np})[0]

    max_abs = float(np.abs(torch_out - ort_out).max())
    denom = float(np.abs(torch_out).max()) + 1e-9
    max_rel = max_abs / denom
    tn = torch_out / (np.linalg.norm(torch_out, axis=1, keepdims=True) + 1e-9)
    on = ort_out / (np.linalg.norm(ort_out, axis=1, keepdims=True) + 1e-9)
    cos_min = float((tn * on).sum(1).min())
    logger.info(f"Parity vs PyTorch: max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}  "
                f"cosine_min={cos_min:.6f}")
    # Cosine similarity is what matters for ReID matching; require it to be ~1.
    if cos_min > 0.9999:
        logger.info("PARITY OK")
    else:
        logger.warning("PARITY WARNING: cosine below threshold; inspect before deploying.")
