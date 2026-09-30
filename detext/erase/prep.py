# -*- coding: utf-8 -*-
"""擦除阶段的输入预处理：把「源视频 + 掩码视频」降采样并切段，产出安全、可分段的模型输入。"""
import os
import subprocess

import numpy as np

from .. import ffmpeg
from ..config import EraseOptions


def plan_segments(w, h, n_frames, opts: EraseOptions, max_img_size=None):
    """给一个视频算「降采样到多少 + 切成几段」。返回 plan dict。"""
    max_img_size = max_img_size or opts.max_img_size
    long_side = max(w, h)
    s = min(1.0, opts.long_side / long_side)
    w2 = max(2, int(w * s) // 16 * 16)
    h2 = max(2, int(h * s) // 16 * 16)

    by_ram = int(opts.ram_budget_gb * 1e9 / (w2 * h2 * 3))

    L = max(w2, h2)
    if L > max_img_size:
        r = L / float(max_img_size)
        mw = int(w2 / r) // 8 * 8
        mh = int(h2 / r) // 8 * 8
    else:
        mw, mh = w2 // 8 * 8, h2 // 8 * 8
    model_px = max(1, mw * mh)
    by_compute = int(opts.max_px_frames // model_px)

    seg_len = max(1, min(by_compute, by_ram, opts.max_seg_frames))
    n_seg = max(1, int(np.ceil(n_frames / seg_len)))
    seg_len = int(np.ceil(n_frames / n_seg))
    driver = "算力" if by_compute <= min(by_ram, opts.max_seg_frames) else \
             ("内存" if by_ram <= opts.max_seg_frames else "帧数硬上限")
    return dict(w2=w2, h2=h2, scale=s, model_w=mw, model_h=mh, model_px=model_px,
                seg_len=seg_len, n_seg=n_seg, driver=driver, by_ram=by_ram)


def _write_mask_segment(src_mask, ss, dur, w2, h2, fps, path):
    """切段 + 降采样 + 127 二值化，一趟做完，无损写出纯 {0,255} 的掩码。"""
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{ss:.6f}", "-t", f"{dur:.6f}",
         "-i", src_mask, "-vf", f"scale={w2}:{h2}:flags=neighbor",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{w2}x{h2}", "-r", f"{fps}", "-i", "pipe:0",
         "-c:v", "libx264", "-qp", "0", "-preset", "fast",
         "-pix_fmt", "gray", path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = w2 * h2
    n = 0
    try:
        while True:
            buf = dec.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            arr = np.frombuffer(buf, np.uint8)
            enc.stdin.write(((arr > 127).astype(np.uint8) * 255).tobytes())
            n += 1
    finally:
        dec.stdout.close()
        dec.wait()
        enc.stdin.close()
        enc.wait()
    if n == 0:
        raise RuntimeError(f"掩码解码出 0 帧: {src_mask} @{ss:.2f}s")
    return n


def prepare(src, mask, out_dir, opts: EraseOptions, info=None):
    """把 (源视频, 掩码视频) 切好段落到 `out_dir`，返回 manifest 条目。

    `src` 与 `mask` 必须同尺寸、同 fps。
    """
    info = info or ffmpeg.probe(src)
    w, h, n, fps = info["w"], info["h"], info["nframes"], info["fps"]
    if not n:
        raise RuntimeError(f"{src} 读不到帧数（容器头没有 nb_frames）")

    p = plan_segments(w, h, n, opts)
    w2, h2, seg_len = p["w2"], p["h2"], p["seg_len"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[prep] {w}x{h} {n}帧 → 预降采样{w2}x{h2} → "
          f"模型工作{p['model_w']}x{p['model_h']}({p['model_px'] / 1e3:.0f}k px)\n"
          f"       原生全解{n * w * h * 3 / 1e9:.2f}GB → "
          f"降采样后{n * w2 * h2 * 3 / 1e9:.2f}GB\n"
          f"       切 {p['n_seg']} 段（每段≤{seg_len}帧，受限于{p['driver']}；"
          f"帧×像素={seg_len * p['model_px'] / 1e6:.0f}e6 / "
          f"预算{opts.max_px_frames / 1e6:.0f}e6）", flush=True)

    vf = f"scale={w2}:{h2}:flags=lanczos"
    segs = []
    for k in range(p["n_seg"]):
        a = k * seg_len
        b = min(n, a + seg_len)
        if b - a <= 0:
            break
        raw_p = os.path.join(out_dir, f"seg{k:02d}_raw.mp4")
        mask_p = os.path.join(out_dir, f"seg{k:02d}_mask.mp4")
        ss = a / fps
        dur = (b - a) / fps
        ffmpeg.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.6f}",
                    "-t", f"{dur:.6f}", "-i", src, "-vf", vf, "-an",
                    "-c:v", "libx264", "-crf", "12", "-preset", "fast",
                    "-pix_fmt", "yuv420p", raw_p])
        _write_mask_segment(mask, ss, dur, w2, h2, fps, mask_p)
        segs.append(dict(idx=k, start=a, end=b, raw=raw_p, mask=mask_p, n=b - a))

    return dict(ext=os.path.splitext(src)[1], orig_w=w, orig_h=h,
                fps=fps, n_frames=n, has_audio=info["has_audio"],
                work_w=w2, work_h=h2, scale=p["scale"], segments=segs)
