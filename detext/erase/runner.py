# -*- coding: utf-8 -*-
"""擦除阶段的编排：加载一次模型 → 逐段推理 → 拼回整条 → 按掩码合成成品。"""
import os
import subprocess
import time

import cv2
import numpy as np

from .. import config, ffmpeg
from ..config import EraseOptions
from . import prep as _prep


class Eraser:
    """持有加载好的 DiffuEraser + ProPainter，可反复调用做擦除。"""

    def __init__(self, opts: EraseOptions, device=None):
        import torch
        from propainter.inference import Propainter, get_device
        from .model import DiffuEraser

        from .. import config
        self.opts = opts
        self.device = device or get_device()
        self._torch = torch

        t = time.time()
        self.diffusion = DiffuEraser(
            self.device,
            config.DFR_BASE_MODEL_DIR,
            config.DFR_VAE_DIR,
            config.DFR_UNET_DIR,
            ckpt=opts.ckpt,
            mode=opts.mode,
            pcm_path=config.DFR_PCM_DIR,
        )
        self.propainter = Propainter(config.PROPAINTER_DIR, device=self.device)
        self.load_seconds = time.time() - t

    def erase_video(self, src, mask, dst, info=None, work_dir=None,
                    compare_path=None, log=print):
        """`src` + `mask` → 擦除后的 `dst`（原分辨率、音频 copy）
        + 可选的「左原片 | 右成品」对照 `compare_path`。返回统计 dict。"""
        t0 = time.time()
        src, mask, dst = (os.path.abspath(p) for p in (src, mask, dst))
        info = info or ffmpeg.probe(src)
        work_dir = work_dir or (os.path.splitext(dst)[0] + "_work")
        os.makedirs(work_dir, exist_ok=True)

        man = _prep.prepare(src, mask, work_dir, self.opts, info=info)
        fps, n_frames = man["fps"], man["n_frames"]

        seg_paths = []
        n_prior = n_diff = 0.0
        seg_stats = []
        for seg in man["segments"]:
            out = os.path.join(work_dir, f"seg{seg['idx']:02d}.mp4")
            priori = os.path.join(work_dir, f"_priori{seg['idx']:02d}.mp4")
            vl = seg["n"] / fps + 1.0

            t = time.time()
            log(f"    [段{seg['idx']}] 先验推理（{seg['n']}帧）…")
            self.propainter.forward(
                seg["raw"], seg["mask"], priori, video_length=vl,
                ref_stride=self.opts.ref_stride,
                neighbor_length=self.opts.neighbor_length,
                subvideo_length=self.opts.subvideo_length,
                mask_dilation=self.opts.mask_dilation_iter)
            dt_prior = time.time() - t

            t = time.time()
            self.diffusion.forward(
                seg["raw"], seg["mask"], priori, out,
                max_img_size=self.opts.max_img_size, video_length=vl,
                mask_dilation_iter=self.opts.mask_dilation_iter,
                guidance_scale=self.opts.guidance_scale, seed=self.opts.seed)
            dt_diff = time.time() - t
            self._torch.cuda.empty_cache()

            n_prior += dt_prior
            n_diff += dt_diff
            seg_stats.append(dict(idx=seg["idx"], frames=seg["n"],
                                  prior_s=round(dt_prior, 1),
                                  diffusion_s=round(dt_diff, 1),
                                  s_per_frame=round((dt_prior + dt_diff) / max(seg["n"], 1), 3),
                                  work_px=man["work_w"] * man["work_h"]))
            log(f"    [段{seg['idx']}] {seg['n']}帧 先验{dt_prior:.0f}s "
                f"扩散{dt_diff:.0f}s = {(dt_prior + dt_diff) / max(seg['n'], 1):.2f} s/帧")
            seg_paths.append(out)

        joined = self._join(man, seg_paths, work_dir)
        self._compose(src, mask, joined, dst, info, compare_path=compare_path)
        t_compose = time.time() - t0 - n_prior - n_diff

        return dict(frames=man["n_frames"], segments=len(man["segments"]),
                    prior_seconds=round(n_prior, 1), diffusion_seconds=round(n_diff, 1),
                    compose_seconds=round(t_compose, 1),
                    seconds=time.time() - t0, work_dir=work_dir,
                    work_w=man["work_w"], work_h=man["work_h"],
                    seg_stats=seg_stats)

    @staticmethod
    def _join(man, seg_paths, work_dir):
        """把分段拼回整条。只有一段就用原文件。"""
        if len(seg_paths) == 1:
            return seg_paths[0]
        work_dir = os.path.abspath(work_dir)
        lst = os.path.join(work_dir, "_concat.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            for p in seg_paths:
                fh.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/"))
        joined = os.path.join(work_dir, "_joined.mp4")
        ffmpeg.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", "14",
                    "-preset", "fast", "-pix_fmt", "yuv420p", joined])
        os.remove(lst)
        return joined

    def _compose(self, src, mask_video, res_video, dst, info, compare_path=None):
        """按掩码合成，音频从源 copy。返回写出的帧数。"""
        w, h, fps = info["w"], info["h"], info["fps"]
        part = ffmpeg.part_path(dst)
        if os.path.exists(part):
            os.remove(part)

        writer = ffmpeg.FrameWriter(
            ffmpeg.frame_encoder(part, (w, h), fps, audio_from=src,
                                 crf=self.opts.crf, preset=self.opts.preset))
        writer.start()

        cw = None
        if compare_path:
            cpart = ffmpeg.part_path(compare_path)
            if os.path.exists(cpart):
                os.remove(cpart)
            cw = ffmpeg.FrameWriter(
                ffmpeg.frame_encoder(cpart, (2 * w, h), fps, audio_from=src,
                                     crf=config.COMPARE_CRF,
                                     preset=config.COMPARE_PRESET))
            cw.start()

        cap_s = cv2.VideoCapture(src)
        cap_m = cv2.VideoCapture(mask_video)
        cap_r = cv2.VideoCapture(res_video)
        if not (cap_s.isOpened() and cap_m.isOpened() and cap_r.isOpened()):
            bad = [f"{n}={p}（{os.path.getsize(p) if os.path.exists(p) else '文件不存在'}）"
                   for n, p, c in (("原片", src, cap_s), ("掩码", mask_video, cap_m),
                                   ("擦除结果", res_video, cap_r)) if not c.isOpened()]
            cap_s.release()
            cap_m.release()
            cap_r.release()
            writer.abort()
            if cw:
                cw.abort()
            raise RuntimeError("合成时打不开：\n  " + "\n  ".join(bad))

        rw = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
        rh = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        compare = np.zeros((h, 2 * w, 3), np.uint8) if cw else None
        n = 0
        try:
            while True:
                ok1, orig = cap_s.read()
                ok2, m_raw = cap_m.read()
                ok3, r = cap_r.read()
                if not (ok1 and ok2 and ok3):
                    break
                if m_raw.ndim == 3:
                    m_raw = cv2.cvtColor(m_raw, cv2.COLOR_BGR2GRAY)
                m_work = cv2.resize((m_raw > 127).astype(np.uint8), (rw, rh),
                                    interpolation=cv2.INTER_NEAREST)
                m_work = cv2.erode(m_work, k3, iterations=1)
                m_work = cv2.dilate(m_work, k3, iterations=self.opts.mask_dilation_iter)
                m_up = cv2.resize(m_work * 255, (w, h),
                                  interpolation=cv2.INTER_LINEAR) > 127
                r_up = cv2.resize(r, (w, h), interpolation=cv2.INTER_LANCZOS4)
                comp = np.where(m_up[:, :, None], r_up, orig)
                writer.write(comp)
                if cw:
                    compare[:, :w] = orig
                    compare[:, w:] = comp
                    cw.write(compare)
                n += 1
        finally:
            cap_s.release()
            cap_m.release()
            cap_r.release()
            if n == 0:
                writer.abort()
                if cw:
                    cw.abort()
                if os.path.exists(part):
                    os.remove(part)
                raise RuntimeError("一帧都没合成出来")
        writer.finish()
        ffmpeg.replace_atomic(part, dst)
        if cw:
            cw.finish()
            ffmpeg.replace_atomic(ffmpeg.part_path(compare_path), compare_path)
        return n
