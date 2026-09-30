# -*- coding: utf-8 -*-
"""检测层：PP-OCRv4 DB 检测器 + 分辨率策略。"""
import os
import sys

import cv2
import numpy as np

from .. import config

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "vendor")

if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

_COMPAT = os.path.join(_VENDOR, "compat")
if _COMPAT not in sys.path:
    sys.path.insert(0, _COMPAT)

DEFAULT_DET_MODEL_DIR = config.PPOCR_DET_DIR

MIN_LONG_SIDE = 1280
MAX_LONG_SIDE = 1920


def _register_cudnn():
    """让 paddle 找得到 `cudnn64_9.dll` —— 手段是**先 import torch**。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        pass


def build_ocr(det_model_dir=DEFAULT_DET_MODEL_DIR, use_gpu=True, show_log=False):
    """建检测器（PP-OCRv4 + 同一组参数）。"""
    if use_gpu:
        _register_cudnn()
    from ocr_util.paddleocr import PaddleOCR
    if not os.path.isdir(det_model_dir):
        raise FileNotFoundError(f"找不到检测模型目录 {det_model_dir}")
    if use_gpu:
        import paddle
        if paddle.device.cuda.device_count() == 0:
            print("[warn] build_ocr(use_gpu=True) 但本机看不到可用 GPU，"
                  "本次退回 CPU 推理（约慢 2x）", file=sys.stderr, flush=True)
            use_gpu = False
    return PaddleOCR(
        det_model_dir=det_model_dir,
        det_db_score_mode="slow",
        det_limit_side_len=1920,
        det_db_thresh=0.1,
        use_gpu=use_gpu,
        show_log=show_log,
    )


def work_scale(w, h, max_long_side=MAX_LONG_SIDE, min_long_side=MIN_LONG_SIDE):
    """给定帧尺寸，返回检测时的缩放系数。"""
    long_side = max(w, h)
    target = max(min_long_side or 0, long_side // 2)
    if max_long_side:
        target = min(target, max_long_side)
    target = min(target, long_side)
    return target / float(long_side)


def detect_boxes(ocr_model, frame, scale=1.0, quads_out=None):
    """检测，返回**原分辨率**坐标下的框 [(x0, y0, x1, y1), ...]。

    `quads_out` 不为 None 时，额外把**未取包络之前的 4 角四边形**塞进去（同样乘回原分辨率）。
    """
    if scale < 1.0:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        small = frame
    result = ocr_model.ocr(small, cls=False, rec=False, poly=False)
    inv = 1.0 / scale if scale > 0 else 1.0
    boxes = []
    if result and result[0]:
        for points in result[0]:
            a = np.asarray(points, np.float32) * inv
            boxes.append([float(a[:, 0].min()), float(a[:, 1].min()),
                          float(a[:, 0].max()), float(a[:, 1].max())])
            if quads_out is not None:
                quads_out.append(a if a.shape[0] == 4 else None)
    return boxes
