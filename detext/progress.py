# -*- coding: utf-8 -*-
"""推理阶段的进度条：按已知总数报进度，带已用时与预计剩余时间。

环境变量 `DETEXT_NO_PROGRESS=1` 可整体关掉（写日志、跑批时用）。
"""
import os
import sys

_ON = os.environ.get("DETEXT_NO_PROGRESS", "") not in ("1", "true", "True")


class _Null:
    """进度条的空实现：总数未知、tqdm 缺失或被关掉时用它顶替。"""

    def update(self, n=1):
        pass

    def set_postfix(self, *a, **k):
        pass

    def set_description(self, *a, **k):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def enabled():
    """进度条总开关（`DETEXT_NO_PROGRESS=1` 关掉）。"""
    return _ON


def bar(total, desc, unit="it", leave=False):
    """建一个进度条。`total <= 0`、被关掉或没装 tqdm 时返回空操作对象。"""
    if not _ON or not total or total <= 0:
        return _Null()
    try:
        from tqdm import tqdm
    except ImportError:
        return _Null()
    return tqdm(total=total, desc=desc, unit=unit, leave=leave,
                dynamic_ncols=True, file=sys.stdout)


def pulse(desc):
    """总数未知时用：只报已用时与次数，不报百分比。"""
    if not _ON:
        return _Null()
    try:
        from tqdm import tqdm
    except ImportError:
        return _Null()
    return tqdm(desc=desc, unit="it", leave=False, dynamic_ncols=True,
                file=sys.stdout)
