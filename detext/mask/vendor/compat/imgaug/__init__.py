# -*- coding: utf-8 -*-
"""imgaug 的惰性 stub。

为什么需要：
    Codes_v3 里 `ppocr/data/imaug/__init__.py` 会 eager import `iaa_augment`，
    后者 `import imgaug`。但 imgaug 0.4.0（2020 年最后一个版本）在
    numpy>=2 上直接崩：`AttributeError: np.sctypes was removed in the NumPy 2.0 release`。

为什么可以 stub：
    imgaug 在整个仓库里只被 `iaa_augment.IaaAugment` 使用，而 `IaaAugment`
    只被 `copy_paste.py`（训练期数据增强）引用。字幕掩码提取走的是
    `tools/infer/predict_det.py` 的推理路径 —— 用的是 `operators.py` 里的
    DetResizeForTest / NormalizeImage / ToCHWImage / KeepKeys，全是 numpy+cv2，
    完全不碰 imgaug。

所以这里只提供「能 import 通过」的壳子；一旦真的被实例化/调用会立刻大声报错，
不会静默给出错误结果。
"""

_MSG = ("imgaug 在本工程里只有训练期数据增强（IaaAugment/copy_paste）会用到，"
        "当前是 numpy>=2 环境，imgaug 0.4.0 无法加载。"
        "如果你确实需要它，请换 numpy<2 的环境，不要依赖这个 stub。")


def __getattr__(name):
    raise RuntimeError(_MSG)
