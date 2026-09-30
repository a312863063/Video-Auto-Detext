# -*- coding: utf-8 -*-
"""见上层 imgaug/__init__.py 的说明。这个文件只是让 `import imgaug.augmenters as iaa` 通过。"""

_MSG = ("imgaug 在本工程里只有训练期数据增强（IaaAugment/copy_paste）会用到，"
        "当前是 numpy>=2 环境，imgaug 0.4.0 无法加载。"
        "如果你确实需要它，请换 numpy<2 的环境，不要依赖这个 stub。")


def __getattr__(name):
    raise RuntimeError(_MSG)
