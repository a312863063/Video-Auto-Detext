# -*- coding: utf-8 -*-
"""掩码阶段的流水线：视频 → 字幕掩码视频。"""
import os
import time

import cv2
import numpy as np

from .. import config, ffmpeg
from ..config import MaskOptions
from .boxes import (grow, nearest_good, sticky_exts, tight_box, STICKY_NEAR,
                    analyse)
from .detect import build_ocr, detect_boxes, work_scale
from .filter import MotionWindow, burned_in_features, link_and_decide, tilt_of
from .filter import MOTION_MIN, is_panel_block
from . import filter as FILTER
from .render import build_mask

WIN = 8

_OCR = None
_DEVICE = "auto"


def get_ocr():
    """按 `_DEVICE` 建检测器（每个进程只建一次）。"""
    global _OCR
    if _OCR is None:
        _OCR = build_ocr(use_gpu=(_DEVICE != "cpu"))
    return _OCR


def pass1_detect(src, scale, static=True, tilt=True, panel=True):
    """逐帧检测 + 时域统计 + 静态外观特征。返回 per_frame 列表（框用 work 坐标）。"""
    ocr = get_ocr()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 {src}")
    mw = MotionWindow(WIN + 1)
    per_frame = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1.0 else bgr
            mw.push(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
            quads = [] if tilt else None
            boxes = detect_boxes(ocr, bgr, scale, quads_out=quads)
            wboxes = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]
                      for b in boxes]
            if mw.ready():
                ratios, raws = mw.ratios(wboxes, with_raw=True)
                motion = mw.motion()
            else:
                ratios, raws, motion = [1.0] * len(boxes), [0.0] * len(boxes), None

            cut = mw.has_cut()
            if cut and motion is not None:
                vote_ratios, vote_raws = mw.ratios(wboxes, with_raw=True, clean=True)
            else:
                vote_ratios, vote_raws = ratios, raws
            stats = None
            if static:
                stats = []
                for b in boxes:
                    f = burned_in_features(bgr, b)
                    stats.append(None if f is None else
                                 (f["fill_purity"], f["edge_w"], f["reliable"]))
            tilts = None
            if quads is not None:
                tilts = [tilt_of(None if q is None else q * scale) for q in quads]
            blocks = None
            if panel:
                if motion is not None and motion >= MOTION_MIN:
                    blocks = [is_panel_block(analyse(bgr, b, return_info=True)[1], b)
                              for b in boxes]
                else:
                    blocks = [False] * len(boxes)
            per_frame.append(dict(boxes=wboxes, ratios=ratios, raws=raws,
                                  motion=motion, stats=stats, tilt=tilts,
                                  cut=cut, blocks=blocks,
                                  vote_ratios=vote_ratios, vote_raws=vote_raws))
    finally:
        cap.release()
    return per_frame


def decide_frame(bgr, boxes, opts):
    """单帧、一串**原分辨率**的框 → [(box, rect, has_bg, ext), ...]。"""
    rec = []
    for b in boxes:
        if not opts.use_bg:
            rec.append((b, tight_box(b), False, None))
            continue
        rect, inf = analyse(bgr, b, return_info=True)
        rec.append((b, rect, bool(inf["has_bg"]), inf["ext"]))
    return rec


def sticky_tracks(plan, keep_sets, tracks, share, report=None, near=None):
    """按轨迹补判决（原地改写 plan）。

    只动自己没判出带背景的帧，借用同一条轨迹里时间上最近的那个判得出的帧的外扩量。
    """
    near = STICKY_NEAR if near is None else near
    n_sticky = n_fix = 0
    if tracks and share < 2.0:
        slots = [{j: s for s, j in enumerate(sorted(keep_sets[t]))}
                 for t in range(len(plan))]
        for members in tracks:
            members = [(t, j) for t, j in members
                       if t < len(slots) and j in slots[t]]
            if len(members) < 2:
                continue
            ms = [(t, slots[t][j]) for t, j in members]
            infos = [dict(has_bg=plan[t][s][2], ext=plan[t][s][3]) for t, s in ms]
            good = sticky_exts(infos, share=share)
            if good is None:
                continue
            donor = {ms[k][0]: infos[k]["ext"] for k in good}
            n_sticky += 1
            for t, s in ms:
                b, _rect, hb, e = plan[t][s]
                if hb and e is not None:
                    continue
                ext = nearest_good(donor, t, near)
                if ext is None:
                    continue
                plan[t][s] = (b, grow(b, ext), True, ext)
                n_fix += 1
    if report is not None:
        report["n_sticky"] = n_sticky
        report["sticky_boxes"] = n_fix
    return plan


def plan_boxes(src, opts, per_frame, keep_sets, scale, tracks=None, report=None,
               anchors=None):
    """2a + 2b 的**流式**版本：自己解一遍 src。

    返回 plan[t] = [(box, rect, has_bg, ext), ...]，顺序与 sorted(keep_sets[t]) 一致。
    """
    inv = 1.0 / scale
    anchors = list(anchors or [])
    plan = []
    n_filled = 0
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 {src}")
    try:
        for i in range(len(per_frame)):
            ok, bgr = cap.read()
            if not ok:
                break

            boxes = [[v * inv for v in per_frame[i]["boxes"][j]]
                     for j in sorted(keep_sets[i])]
            rec = decide_frame(bgr, boxes, opts)
            if anchors:
                rects = [r for _, r, _, _ in rec]
                for a in FILTER.anchor_gap(rects, anchors):
                    rec.append((a, list(a), False, None))
                    n_filled += 1
            plan.append(rec)
    finally:
        cap.release()
    if opts.use_bg and opts.use_sticky:
        sticky_tracks(plan, keep_sets, tracks, opts.sticky_share, report=report)
    elif report is not None:
        report["n_sticky"] = report["sticky_boxes"] = 0
    if report is not None:
        report["n_anchor_rects"] = len(anchors)
        report["anchor_fills"] = n_filled
    return plan


def encode(src, dst, opts, info, plan, compare_path=None):
    """按最终矩形出掩码并写两份视频。返回 (帧数, 保留框总数, 带背景框数)。"""
    w, h, fps = info["w"], info["h"], info["fps"]
    part = ffmpeg.part_path(dst)
    if os.path.exists(part):
        os.remove(part)

    writer = ffmpeg.FrameWriter(
        ffmpeg.frame_encoder(part, (w, h), fps, audio_from=None,
                             crf=opts.crf, preset=opts.preset,
                             threads=opts.x264_threads, out_pix_fmt="gray"))
    writer.start()

    cw = None
    if compare_path:
        cpart = ffmpeg.part_path(compare_path)
        if os.path.exists(cpart):
            os.remove(cpart)
        cw = ffmpeg.FrameWriter(
            ffmpeg.frame_encoder(cpart, (2 * w, h), fps, audio_from=src,
                                 crf=config.COMPARE_CRF,
                                 preset=config.COMPARE_PRESET,
                                 threads=opts.x264_threads))
        cw.start()

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        writer.abort()
        if cw:
            cw.abort()
        raise RuntimeError(f"无法打开 {src}")

    mask_buf = np.zeros((h, w, 3), np.uint8)
    compare = np.zeros((h, 2 * w, 3), np.uint8) if cw else None
    n = kept_total = n_bg = 0
    try:
        for rec in plan:
            ok, bgr = cap.read()
            if not ok:
                break
            kept_total += len(rec)
            rects = []
            for _, rect, has_bg, _ext in rec:
                if has_bg:
                    n_bg += 1
                rects.append(rect)
            mask = build_mask(bgr.shape, rects)
            mask_buf[:] = np.where(mask[:, :, None] > 0, 255, 0)
            writer.write(mask_buf)
            if cw:
                compare[:, :w] = bgr
                compare[:, w:] = mask_buf
                cw.write(compare)
            n += 1
    except Exception:
        writer.abort()
        if cw:
            cw.abort()
        cap.release()
        for p in (part, ffmpeg.part_path(compare_path) if compare_path else None):
            if p and os.path.exists(p):
                os.remove(p)
        raise
    finally:
        cap.release()

    writer.finish()
    ffmpeg.replace_atomic(part, dst)
    if cw:
        cw.finish()
        ffmpeg.replace_atomic(ffmpeg.part_path(compare_path), compare_path)
    return n, kept_total, n_bg


def extract(src, dst, opts: MaskOptions, compare_path=None, info=None):
    """一个视频 → 一份纯掩码视频 + 一份左右对照视频。返回统计 dict。"""
    global _DEVICE
    _DEVICE = opts.device
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    info = info or ffmpeg.probe(src)
    if info.get("rotation"):
        print(f"[warn] {os.path.basename(src)} 含旋转元数据 "
              f"rotation={info['rotation']}（本流水线不处理）", flush=True)

    t0 = time.time()
    scale = work_scale(info["w"], info["h"], opts.max_long_side)
    per_frame = pass1_detect(src, scale, static=opts.use_static, tilt=opts.use_tilt,
                             panel=opts.use_panel)

    rep = {}
    tracks = []
    anchors = []
    keep_sets = link_and_decide(per_frame, report=rep, tracks_out=tracks,
                                static_enable=opts.use_static,
                                static_margin=opts.static_margin,
                                tilt_enable=opts.use_tilt,
                                panel_enable=opts.use_panel,
                                occupy_enable=opts.use_occupy,
                                anchors_out=anchors)

    anchors = [[v / scale for v in a] for a in anchors]
    plan = plan_boxes(src, opts, per_frame, keep_sets, scale, tracks=tracks,
                      report=rep, anchors=anchors)
    n, n_kept, n_bg = encode(src, dst, opts, info, plan, compare_path=compare_path)

    rep.update(frames=n, kept_boxes=n_kept, bg_boxes=n_bg,
               seconds=time.time() - t0, scale=scale)
    return rep


def describe(rep):
    """把统计 dict 压成一行日志。"""
    n, nk = rep.get("frames", 0), rep.get("kept_boxes", 0)
    dt = rep.get("seconds", 0.0)
    pct = (100.0 * rep.get("bg_boxes", 0) / nk) if nk else 0.0
    why = []
    if rep.get("video_motion", 9) < FILTER.MOTION_MIN_VIDEO:
        why.append(f"相机静止{rep['video_motion']:.3f}px")
    if rep.get("scene_density", 9) < FILTER.MIN_SCENE_DENSITY:
        why.append(f"背景字密度{rep.get('scene_density', 0):.2f}框/帧")
    gate = "整片保留(" + "，".join(why) + ")" if why else \
        (f"执行剔除({rep.get('n_candidates', 0)}条候选/"
         f"{rep.get('candidate_boxes', 0)}框"
         f" = 时域{rep.get('n_cand_motion', 0)}条 + "
         f"静态{rep.get('n_cand_static', 0)}条/{rep.get('static_boxes', 0)}框)")
    if rep.get("n_cand_tilt"):
        gate += f" + 倾角{rep['n_cand_tilt']}条/{rep.get('tilt_boxes', 0)}框"
    if rep.get("n_cand_panel"):
        gate += f" + 静帧面板{rep['n_cand_panel']}条/{rep.get('panel_boxes', 0)}框"
    if rep.get("n_occupy_saved"):
        gate += (f"；驻留豁免{rep['n_occupy_saved']}条/"
                 f"{rep.get('occupy_boxes', 0)}框")
    if rep.get("n_anchor_rects"):
        gate += (f"；驻留补全{rep['n_anchor_rects']}片/"
                 f"补{rep.get('anchor_fills', 0)}框")
    if not rep.get("allow_drop", True) and (rep.get("allow_tilt")
                                            or rep.get("allow_panel")):
        gate += "；倾角/面板路径另开闸"
    st = rep.get("n_sticky", 0)
    stx = f" 粘住{st}条/{rep.get('sticky_boxes', 0)}框" if st else ""
    return (f"{n}帧 保留框{nk} 其中带背景{rep.get('bg_boxes', 0)}({pct:.0f}%){stx}"
            f" {dt:.0f}s = {n / dt:.2f} fps | 缩放{rep.get('scale', 1.0):.2f} | {gate}")
