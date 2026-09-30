# -*- coding: utf-8 -*-
"""掩码生成：框**已经是最终形状**，这里只负责填充 + 轻膨胀。"""
import cv2
import numpy as np

from .boxes import analyse, refine_boxes, tight_box   # noqa: F401

DILATE_PX = 2


def _odd(v, lo=1):
    v = max(lo, int(round(v)))
    return v if v % 2 == 1 else v + 1


def build_mask(shape, boxes, dilate=DILATE_PX):
    """由**已经定好的**框生成掩码。框是轴对齐矩形，坐标可为浮点（会四舍五入）。"""
    mask = np.zeros(shape[:2], np.uint8)
    h, w = shape[:2]
    for x0, y0, x1, y1 in boxes:
        quad = np.array([[max(0, min(w, int(round(x0)))), max(0, min(h, int(round(y0))))],
                         [max(0, min(w, int(round(x1)))), max(0, min(h, int(round(y0))))],
                         [max(0, min(w, int(round(x1)))), max(0, min(h, int(round(y1))))],
                         [max(0, min(w, int(round(x0)))), max(0, min(h, int(round(y1))))]],
                        np.int32)
        if quad[:, 0].max() - quad[:, 0].min() < 1 or \
           quad[:, 1].max() - quad[:, 1].min() < 1:
            continue
        cv2.fillPoly(mask, [quad], 255)
    if dilate > 0 and mask.any():
        k = _odd(2 * dilate + 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k)),
                          iterations=1)
    return mask


def build_mask_for_frame(bgr, boxes, dilate=DILATE_PX):
    """先按像素判决框，再出掩码。返回 (掩码, 定好的框, 判决明细)。"""
    refined, infos = refine_boxes(bgr, boxes)
    return build_mask(bgr.shape, refined, dilate=dilate), refined, infos
