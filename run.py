# -*- coding: utf-8 -*-
"""Detext 入口：视频 → 字幕掩码 → 擦除 → 新视频。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detext                                  # noqa: E402
from detext import config                      # noqa: E402
from detext.config import EraseOptions, MaskOptions, RunOptions   # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="Detext —— 从视频里提取字幕掩码并擦除，输出新视频",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = ap.add_argument_group("输入 / 输出")
    g.add_argument("-i", "--input", required=True,
                   help="单个视频文件，或一个装着视频的目录（目录 = 批量）")
    g.add_argument("-o", "--out", default=config.DEFAULT_OUT_DIR,
                   help="输出根目录。输入是目录时会再套一层 <目录名>/")
    g.add_argument("--stage", choices=config.STAGES, default="all",
                   help="mask=只出掩码；erase / all=掩码+擦除（两者等价：擦除阶段不再"
                        "读取已有掩码，需要就当场重算，因此同样产出 mask_result/）")
    g.add_argument("--compare", action="store_true",
                   help="把保存的两个视频做成「左原片 | 右结果」的对照拼接；"
                        "默认是各自单独的视频")
    g.add_argument("--only", default=None,
                   help="只处理这些视频（逗号分隔的**文件名词干**，精确匹配）")
    g.add_argument("--out-exact", action="store_true",
                   help="把 --out 当成最终产物目录本身；默认（关）输入是目录时会再套一层 "
                        "<目录名>/")
    g.add_argument("--jobs", type=int, default=1,
                   help="掩码阶段的并发进程数。擦除阶段显存独占，始终串行")

    m = ap.add_argument_group("掩码阶段（原 Codes_v6）")
    m.add_argument("--max-long-side", type=int, default=MaskOptions.max_long_side,
                   help="检测前把长边封顶到这个值；别低于 1280，否则小字会漏")
    m.add_argument("--device", choices=("auto", "cpu", "gpu"),
                   default=MaskOptions.device, help="检测设备")
    m.add_argument("--no-bg", action="store_true",
                   help="关掉背景判决，全部走紧贴模式（做 A/B 用）")
    m.add_argument("--no-sticky", action="store_true",
                   help="关掉轨迹级粘住，逐帧独立出判决（做 A/B 用）")
    m.add_argument("--no-static", action="store_true",
                   help="关掉静态外观剔除，退回「只走时域判据」（做 A/B 用）")
    m.add_argument("--no-tilt", action="store_true",
                   help="关掉倾角剔除（斜排文字判为画面原生文字），做 A/B 用")
    m.add_argument("--no-panel", action="store_true",
                   help="关掉静帧面板剔除（证书/销量牌/商品图里的字判为画面原生文字），"
                        "做 A/B 用")
    m.add_argument("--no-occupy", action="store_true",
                   help="关掉「全片驻留的屏幕叠加物」豁免与补全（水印/角标不再被救回来、"
                        "漏检的帧也不补），做 A/B 用")
    m.add_argument("--static-margin", type=float, default=MaskOptions.static_margin,
                   help="静态判据的余量，越大越保守（少剔）")
    m.add_argument("--sticky-share", type=float, default=MaskOptions.sticky_share,
                   help="轨迹里判出带背景的帧占比达到此值就整条粘住；>=2 等于关掉")
    m.add_argument("--mask-crf", type=int, default=MaskOptions.crf,
                   help="**中间**纯掩码视频的 x264 CRF（有损，擦除阶段还会重新二值化）；"
                        "交付的 `mask_result/<名字>` 在对照模式下走无损，不受这个影响")
    m.add_argument("--preset", default=MaskOptions.preset, help="x264 preset")
    m.add_argument("--x264-threads", type=int, default=MaskOptions.x264_threads)

    e = ap.add_argument_group("擦除阶段（原 DiffuEraser）")
    e.add_argument("--ckpt", default=EraseOptions.ckpt,
                   choices=("2-Step", "4-Step", "8-Step", "16-Step",
                            "Normal CFG 4-Step", "Normal CFG 8-Step",
                            "Normal CFG 16-Step", "LCM-Like LoRA"),
                   help="PCM 加速档位")
    e.add_argument("--max-img-size", type=int, default=EraseOptions.max_img_size,
                   help="模型侧工作分辨率上限（长边）。越大越清晰也越吃显存")
    e.add_argument("--mask-dilation", type=int, default=EraseOptions.mask_dilation_iter,
                   help="掩码膨胀次数。调大擦得更宽，掩码偏紧时用来补边")
    e.add_argument("--long-side", type=int, default=EraseOptions.long_side,
                   help="预处理降采样长边。模型内部还会压到 --max-img-size，"
                        "所以只要 ≥ 它就不会损失细节，纯省内存")
    e.add_argument("--guidance-scale", type=float, default=EraseOptions.guidance_scale,
                   help="扩散引导强度；不给就用档位自带值")
    e.add_argument("--seed", type=int, default=EraseOptions.seed, help="固定噪声种子")
    e.add_argument("--ref-stride", type=int, default=EraseOptions.ref_stride,
                   help="ProPainter 参数")
    e.add_argument("--neighbor-length", type=int, default=EraseOptions.neighbor_length,
                   help="ProPainter 参数")
    e.add_argument("--subvideo-length", type=int, default=EraseOptions.subvideo_length,
                   help="ProPainter 参数")
    e.add_argument("--max-px-frames", type=float, default=EraseOptions.max_px_frames,
                   help="单段的「帧数 × 模型工作像素」预算。**这是决定显存与内存峰值的"
                        "那个旋钮**：调小切得更碎、峰值更低、总耗时略增。方屏素材"
                        "（如 720×720）的模型工作分辨率最大，是峰值最高的一类")
    e.add_argument("--ram-budget-gb", type=float, default=EraseOptions.ram_budget_gb,
                   help="单段解码后的帧张量上限。注意流水线里同时存在 4~5 份整段帧"
                        "（原帧/掩码/被掩码帧/先验），所以**真实内存峰值约为它的 5 倍**")
    e.add_argument("--crf", type=int, default=EraseOptions.crf,
                   help="成品视频的 x264 CRF")

    ap.add_argument("--version", action="version",
                    version=f"Detext {detext.__version__}")
    return ap


def build_options(args):
    mask = MaskOptions(
        max_long_side=args.max_long_side, device=args.device,
        use_bg=not args.no_bg, use_sticky=not args.no_sticky,
        use_static=not args.no_static, static_margin=args.static_margin,
        use_tilt=not args.no_tilt, use_panel=not args.no_panel,
        use_occupy=not args.no_occupy,
        sticky_share=args.sticky_share, crf=args.mask_crf, preset=args.preset,
        x264_threads=args.x264_threads)
    erase = EraseOptions(
        max_img_size=args.max_img_size, mask_dilation_iter=args.mask_dilation,
        ckpt=args.ckpt, guidance_scale=args.guidance_scale, seed=args.seed,
        ref_stride=args.ref_stride, neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length, long_side=args.long_side,
        max_px_frames=args.max_px_frames, ram_budget_gb=args.ram_budget_gb,
        crf=args.crf, preset=args.preset)
    if erase.long_side < erase.max_img_size:
        print(f"[warn] --long-side({erase.long_side}) < --max-img-size"
              f"({erase.max_img_size})：预降采样会先丢细节，模型再放大也补不回来",
              file=sys.stderr)
    return RunOptions(input=args.input, out_dir=args.out, stage=args.stage,
                      jobs=max(1, args.jobs),
                      only=args.only, compare=args.compare,
                      out_exact=args.out_exact,
                      mask=mask, erase=erase)


def check_models(stage):
    """开工前先确认要用的模型都在。"""
    need = [("PP-OCRv4 检测模型", config.PPOCR_DET_DIR)]
    if stage in ("erase", "all"):
        need += [("DiffuEraser 主权重", config.DFR_UNET_DIR),
                 ("SD1.5 底座", config.DFR_BASE_MODEL_DIR),
                 ("VAE (sd-vae-ft-mse)", config.DFR_VAE_DIR),
                 ("PCM LoRA", config.DFR_PCM_DIR),
                 ("ProPainter 先验权重", config.PROPAINTER_DIR)]
    missing = [(n, p) for n, p in need if not os.path.isdir(p)]
    if missing:
        msg = "\n".join(f"  · {n}: {p}" for n, p in missing)
        raise SystemExit(f"缺少模型目录：\n{msg}\n\n"
                         f"本工程是自包含的，不会联网下载。请把它们放到 models/ 下。")
    return need


def main(argv=None):
    args = build_parser().parse_args(argv)
    opts = build_options(args)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    check_models(opts.stage)
    from detext import pipeline
    pipeline.run(opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
