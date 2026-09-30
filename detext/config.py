# -*- coding: utf-8 -*-
"""项目级配置：路径、阶段开关、两条流水线的运行参数。"""
import os
from dataclasses import dataclass, field, asdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(ROOT, "models")

PPOCR_DET_DIR = os.path.join(MODELS_DIR, "ppocr_det", "ch_PP-OCRv4_det_infer")
DFR_UNET_DIR = os.path.join(MODELS_DIR, "diffuEraser")
DFR_BASE_MODEL_DIR = os.path.join(MODELS_DIR, "stable-diffusion-v1-5")
DFR_VAE_DIR = os.path.join(MODELS_DIR, "sd-vae-ft-mse")
DFR_PCM_DIR = os.path.join(MODELS_DIR, "PCM_Weights")
PROPAINTER_DIR = os.path.join(MODELS_DIR, "propainter")

DEFAULT_OUT_DIR = os.path.join(ROOT, "outputs")

MASK_RESULT_DIRNAME = "mask_result"
FINAL_RESULT_DIRNAME = "final_result"
WORK_DIRNAME = "work"

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

COMPARE_CRF = 0
COMPARE_PRESET = "ultrafast"

STAGES = ("mask", "erase", "all")


@dataclass
class MaskOptions:
    """掩码阶段的运行选项。"""

    max_long_side: int = 1920

    device: str = "auto"

    use_bg: bool = True
    use_sticky: bool = True
    use_static: bool = True
    use_tilt: bool = True
    use_panel: bool = True
    use_occupy: bool = True
    static_margin: float = 0.0
    sticky_share: float = 0.15

    crf: int = 0
    preset: str = "medium"
    x264_threads: int = 4


@dataclass
class EraseOptions:
    """擦除阶段的运行选项。"""

    max_img_size: int = 800
    mask_dilation_iter: int = 8
    ckpt: str = "2-Step"
    mode: str = "sd15"
    guidance_scale: float = None
    seed: int = None

    ref_stride: int = 10
    neighbor_length: int = 10
    subvideo_length: int = 50

    long_side: int = 960

    ram_budget_gb: float = 1.5
    max_seg_frames: int = 450
    max_px_frames: float = 150e6

    crf: int = 0
    preset: str = "medium"


@dataclass
class RunOptions:
    """一次运行的整体选项。"""

    input: str = ""
    out_dir: str = DEFAULT_OUT_DIR
    stage: str = "all"
    jobs: int = 1
    only: str = None
    compare: bool = False
    out_exact: bool = False
    keep_pure_mask: bool = False

    mask: MaskOptions = field(default_factory=MaskOptions)
    erase: EraseOptions = field(default_factory=EraseOptions)

    def to_dict(self):
        return asdict(self)


def apply_runtime_env():
    """必须在 import torch 之前调用，设置运行时环境变量。"""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
