# -*- coding: utf-8 -*-
"""DiffuEraser 加速开关。全部默认关闭，即默认行为与原始代码逐位一致，各项通过环境变量单独开启。"""
import os


def _int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


_ALL = os.environ.get("DFR_FAST", "0") not in ("0", "", "false", "False")


def _flag(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "", "false", "False")


FAST = dict(
    blend_sync=_flag("DFR_BLEND_SYNC", _ALL),
    no_empty_cache=_flag("DFR_NO_EMPTY_CACHE", _ALL),
    load_list=_flag("DFR_LOAD_LIST", _ALL),
    vae_decode_chunk=_int("DFR_VAE_DEC_CHUNK", 1),
    vae_encode_chunk=_int("DFR_VAE_ENC_CHUNK", 4),
    win_overlap=_int("DFR_WIN_OVERLAP", -1),
    channels_last=_flag("DFR_CHANNELS_LAST", False),
    cudnn_bench=_flag("DFR_CUDNN_BENCH", False),
    compile=_flag("DFR_COMPILE", False),
    compile_mode=os.environ.get("DFR_COMPILE_MODE", "reduce-overhead"),
)


def describe():
    return " ".join(f"{k}={v}" for k, v in FAST.items())
