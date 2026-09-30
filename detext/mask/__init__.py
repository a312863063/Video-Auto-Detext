# -*- coding: utf-8 -*-
"""字幕掩码提取。对外只暴露一个入口 `runner.extract`。"""
from .runner import extract, describe   # noqa: F401

__all__ = ["extract", "describe"]
