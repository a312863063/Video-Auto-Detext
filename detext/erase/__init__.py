# -*- coding: utf-8 -*-
"""字幕擦除（DiffuEraser + ProPainter 先验）。

把 vendor/ 注入 sys.path，使 vendored 的 dfr_libs / propainter 按原样解析。
"""
import os
import sys

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
