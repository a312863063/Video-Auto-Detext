# -*- coding: utf-8 -*-
"""判别 OCR 框里的文字是后期叠加的字幕、还是画面里本来就有的字。

结果用于避免擦除字幕时把画面中的原生文字一起擦掉。
"""
import cv2
import numpy as np
from collections import deque

PURITY_MIN = 0.30
EDGE_W_MAX = 0.32
MIN_FG_PX = 150
STATIC_ENABLE = True
TILT_ENABLE = True
PANEL_ENABLE = True
STATIC_MARGIN = 0.0


def _foreground(L, rng):
    """大核中值当背景，偏离足够大的算笔画。返回布尔掩码。"""
    bg = cv2.medianBlur(L.astype(np.uint8), 21).astype(np.float32)
    return np.abs(L - bg) > max(0.22 * rng, 18)


def burned_in_features(bgr, box):
    """算一个框的判别特征。返回 dict 或 None（框非法）。"""
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    L = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rng = float(np.percentile(L, 99) - np.percentile(L, 1)) or 1.0

    fg = _foreground(L, rng)
    n_fg = int(fg.sum())
    if n_fg < MIN_FG_PX:
        return dict(n_fg=n_fg, edge_w=0.0, fill_purity=0.0, reliable=False)

    core = cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    if int(core.sum()) < 40:
        core = fg
    vals = L[core]
    bg_vals = cv2.medianBlur(L.astype(np.uint8), 21).astype(np.float32)[core]
    if vals.mean() > bg_vals.mean():
        fill = vals[vals > np.percentile(vals, 60)]
    else:
        fill = vals[vals < np.percentile(vals, 40)]
    if fill.size < 20:
        return dict(n_fg=n_fg, edge_w=0.0, fill_purity=0.0, reliable=False)

    hist = np.bincount(np.clip(fill, 0, 255).astype(np.uint8), minlength=256)
    fill_purity = float(hist.max() / hist.sum())

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    strong = gm >= max(np.percentile(gm, 97), 1e-6)
    edge_w = float(rng / max(np.mean(gm[strong]), 1e-6)) if strong.any() else 99.0

    return dict(n_fg=n_fg, edge_w=edge_w, fill_purity=fill_purity, reliable=True)


def is_burned_in(bgr, box, purity_min=PURITY_MIN, edge_w_max=EDGE_W_MAX):
    """判定单个 OCR 框。返回 (是否保留, 特征 dict)。"""
    f = burned_in_features(bgr, box)
    if f is None:
        return False, {}
    if not f["reliable"]:
        return False, f
    return (f["fill_purity"] >= purity_min and f["edge_w"] < edge_w_max), f


def filter_boxes(bgr, boxes, **kw):
    """批量过滤，返回 (保留的框列表, 每一个的判定明细)。"""
    kept, detail = [], []
    for b in boxes:
        ok, feat = is_burned_in(bgr, b, **kw)
        detail.append(dict(box=list(b), keep=bool(ok), **feat))
        if ok:
            kept.append(b)
    return kept, detail


MOTION_LO = 1.02
MOTION_MIN = 0.15
RAW_MIN = 0.30


def _phase_correlate(a, b, win):
    """返回 (dx, dy)；传副本调用，以免 phaseCorrelate 就地改写输入帧。"""
    (dx, dy), _ = cv2.phaseCorrelate(a.copy(), b.copy(), win.copy())
    return dx, dy


CUT_FACTOR = 6.0
CUT_FLOOR = 0.5
CUT_REL_N = 24
CUT_MIN_HIST = 4


class MotionWindow:
    """滑动窗口的运动统计 —— 跨帧缓存每个「帧对」的结果。"""

    def __init__(self, size=9):
        self.frames = deque(maxlen=size)
        self.steps = deque(maxlen=max(1, size - 1))
        self._level = deque(maxlen=CUT_REL_N)
        self._hann = {}

    def push(self, gray):
        """压入一帧灰度图（float32）。只对新增的那一对算相位相关与残差。"""
        if self.frames:
            prev = self.frames[-1]
            H, W = gray.shape
            win = self._hann.get((H, W))
            if win is None:
                win = cv2.createHanningWindow((W, H), cv2.CV_32F)
                self._hann[(H, W)] = win
            dx, dy = _phase_correlate(prev, gray, win)
            d_raw = np.abs(gray - prev)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(prev, M, (W, H), flags=cv2.INTER_LINEAR)
            dm = float(d_raw.mean())
            cut = False
            if len(self._level) >= CUT_MIN_HIST:
                ref = max(float(np.median(self._level)), CUT_FLOOR)
                cut = dm > CUT_FACTOR * ref
            self._level.append(dm)
            self.steps.append((dx, dy, d_raw, np.abs(gray - aligned), cut))
        self.frames.append(gray)

    def has_cut(self):
        """当前窗口里是否含有切点。有 → 这一帧的比值不可采信。"""
        return any(s[4] for s in self.steps)

    def ready(self):
        """窗口里至少要有 2 个帧对（= 3 帧）才够算统计。"""
        return len(self.steps) >= 2

    def motion(self):
        """窗口内的帧间位移中位数（像素）。"""
        if not self.steps:
            return 0.0
        return float(np.median([float(np.hypot(dx, dy))
                                for dx, dy, _, _, _ in self.steps]))

    def ratios(self, boxes, with_raw=False, clean=False):
        """每个框的「运动补偿残差比」= comp / raw。

        `clean=True` 时只累加未被判成切点的帧对，且只给投票用。
        """
        steps = [s for s in self.steps if not s[4]] if clean else list(self.steps)
        if len(steps) < 2 or not boxes:
            return ([1.0] * len(boxes), [0.0] * len(boxes)) if with_raw \
                else [1.0] * len(boxes)
        H, W = self.frames[0].shape
        raw = np.zeros((H, W), np.float32)
        comp = np.zeros((H, W), np.float32)
        for _, _, d_raw, d_comp, _ in steps:
            raw += d_raw
            comp += d_comp
        n = float(len(steps))
        raw /= n
        comp /= n

        out, raws = [], []
        for b in boxes:
            x0, y0, x1, y1 = [int(v) for v in b]
            r = float(raw[y0:y1, x0:x1].mean())
            c = float(comp[y0:y1, x0:x1].mean())
            raws.append(r)
            out.append(c / r if r > 1e-6 else 1.0)
        return (out, raws) if with_raw else out


def global_motion(grays):
    """一次性版本，给调试脚本用。"""
    if len(grays) < 3:
        return 0.0
    mw = MotionWindow(len(grays))
    for g in grays:
        mw.push(g)
    return mw.motion()


def scene_locked_ratios(grays, boxes, with_raw=False):
    """一次性版本，给调试脚本用。

    with_raw=True 时返回 (比值列表, 未补偿残差列表)。
    """
    if len(grays) < 3 or not boxes:
        return ([1.0] * len(boxes), [0.0] * len(boxes)) if with_raw else [1.0] * len(boxes)
    mw = MotionWindow(len(grays))
    for g in grays:
        mw.push(g)
    return mw.ratios(boxes, with_raw=with_raw)


def decide_final(bgr, box, ratio=None, raw=None, motion=None, **kw):
    """最终判定，返回 (是否保留, 依据, 静态特征)。"""
    feat = burned_in_features(bgr, box)
    if motion is None or motion < MOTION_MIN:
        return True, "静止域→保召回", feat or {}
    if raw is None or raw < RAW_MIN:
        return True, "信号弱→保召回", feat or {}
    if ratio is None:
        return True, "无比值→保召回", feat or {}
    if ratio <= MOTION_LO:
        return False, f"跟随场景(r={ratio:.2f})", feat or {}
    return True, f"屏幕锁定(r={ratio:.2f})", feat or {}


TRACK_IOU = 0.40
TRACK_GAP = 2
MIN_VOTES = 4
RATIO_LO = 0.95

MOTION_MIN_VIDEO = 0.15
MIN_SCENE_DENSITY = 1.13


def _static_says_scene(rows, margin, min_votes):
    """轨迹级的静态外观判决：两项**同时**不像字幕才返回 True。"""
    if len(rows) < min_votes:
        return False
    pur = float(np.median([r[0] for r in rows]))
    edg = float(np.median([r[1] for r in rows]))
    return pur < PURITY_MIN - margin and edg >= EDGE_W_MAX + margin


TILT_MIN_W = 40.0
TILT_MIN = 8.0


FREEZE_RAW = 0.5
PANEL_GROW = 1.3
PANEL_AREA = 5.0
PANEL_SHARE = 0.15


def is_panel_block(info, box):
    """`boxes.analyse` 的 info → 这个框是不是压在一块**远大于自己**的硬边色块上。"""
    if not info or not info.get("has_bg"):
        return False
    rect = info.get("grown")
    if not rect:
        return False
    bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
    if bw < 1 or bh < 1:
        return False
    gw, gh = float(rect[2] - rect[0]), float(rect[3] - rect[1])
    return (gw >= PANEL_GROW * bw and gh >= PANEL_GROW * bh
            and gw * gh >= PANEL_AREA * bw * bh)


def tilt_of(quad, min_w=TILT_MIN_W):
    """DB 四边形 → 上边离水平的夹角，[0, 90]；量不准返回 None。"""
    if quad is None:
        return None
    a = np.asarray(quad, np.float64)
    if a.shape[0] != 4:
        return None
    tl, tr, bl = a[0], a[1], a[3]
    d = tr - tl
    w = float(np.hypot(d[0], d[1]))
    h = float(np.hypot(bl[0] - tl[0], bl[1] - tl[1]))
    if w < min_w or h < 0.3 * min_w:
        return None
    ang = abs(np.degrees(np.arctan2(d[1], d[0]))) % 180.0
    return float(min(ang, 180.0 - ang))


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


OCCUPANCY_IOU = 0.30
OCCUPANCY_MIN = 0.60

ANCHOR_IOU = 0.30
ANCHOR_MERGE = 0.50


def occupancy(per_frame, boxes, iou_thr=OCCUPANCY_IOU):
    """每个框的「全片驻留率」：多少帧在同一屏幕位置也检出了框（IoU ≥ iou_thr）。"""
    arrs = [np.asarray(f["boxes"], np.float32).reshape(-1, 4)
            for f in per_frame]
    n = float(max(1, len(per_frame)))
    out = []
    for q in boxes:
        q = np.asarray(q, np.float32)
        qa = (q[2] - q[0]) * (q[3] - q[1])
        hit = 0
        for a in arrs:
            if a.shape[0] == 0:
                continue
            ix = np.minimum(q[2], a[:, 2]) - np.maximum(q[0], a[:, 0])
            iy = np.minimum(q[3], a[:, 3]) - np.maximum(q[1], a[:, 1])
            inter = np.clip(ix, 0.0, None) * np.clip(iy, 0.0, None)
            ua = qa + (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) - inter
            if np.any(inter / np.maximum(ua, 1e-6) >= iou_thr):
                hit += 1
        out.append(hit / n)
    return out


def _median_box(pf, members):
    """一条轨迹成员框的逐坐标中位数 —— 代表「它钉在哪片屏幕上」。"""
    a = np.asarray([pf[t]["boxes"][i] for t, i in members], np.float32)
    return [float(v) for v in np.median(a, axis=0)]


def merge_anchors(anchors, iou_thr=ANCHOR_MERGE):
    """把彼此几乎重合的锚定矩形并成一条（保留先出现的那条）。"""
    out = []
    for a in anchors:
        if any(_iou(a, b) >= iou_thr for b in out):
            continue
        out.append(a)
    return out


def anchor_gap(rects, anchors, iou_thr=ANCHOR_IOU):
    """这一帧**还缺哪些**锚定矩形：已有的最终矩形没盖住那些。

    返回要补的锚定矩形列表。
    """
    return [a for a in anchors
            if not any(_iou(a, r) >= iou_thr for r in rects)]


def link_and_decide(per_frame, iou_thr=TRACK_IOU, gap=TRACK_GAP,
                    min_votes=MIN_VOTES, ratio_lo=RATIO_LO,
                    motion_min_video=MOTION_MIN_VIDEO,
                    min_scene_density=MIN_SCENE_DENSITY, report=None,
                    tracks_out=None, static_enable=STATIC_ENABLE,
                    static_margin=STATIC_MARGIN, tilt_enable=TILT_ENABLE,
                    tilt_min=TILT_MIN, panel_enable=PANEL_ENABLE,
                    panel_share=PANEL_SHARE,
                    occupy_enable=True, occupy_min=OCCUPANCY_MIN,
                    occupy_iou=OCCUPANCY_IOU, anchors_out=None):
    """逐帧检测结果 → 逐帧的保留框下标集合。

    返回 [set(保留的框下标), ...]，与 per_frame 等长。
    """
    use_static = bool(static_enable) and any(
        f.get("stats") is not None for f in per_frame)
    use_tilt = bool(tilt_enable) and any(
        f.get("tilt") is not None for f in per_frame)
    use_panel = bool(panel_enable) and any(
        f.get("blocks") is not None for f in per_frame)
    use_occupy = bool(occupy_enable) and use_static
    tracks = []
    for t, fr in enumerate(per_frame):
        used = set()
        stats = fr.get("stats")
        tilts = fr.get("tilt")
        blocks = fr.get("blocks")
        for i, b in enumerate(fr["boxes"]):
            best, bj = 0.0, -1
            for j, tr in enumerate(tracks):
                if j in used or t - tr["last"] > gap:
                    continue
                v = _iou(b, tr["box"])
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best >= iou_thr:
                tr = tracks[bj]
            else:
                tr = dict(members=[], votes=[], static=[], tilt=[], panel=[])
                tracks.append(tr)
                bj = len(tracks) - 1
            used.add(bj)
            tr["last"], tr["box"] = t, b
            tr["members"].append((t, i))
            is_panel = bool(use_panel and blocks is not None and blocks[i]
                            and fr["raws"][i] <= FREEZE_RAW)
            if is_panel:
                tr["panel"].append(t)
            else:
                m = fr["motion"]
                vr = fr.get("vote_ratios") or fr["ratios"]
                vw = fr.get("vote_raws") or fr["raws"]
                if m is not None and m >= MOTION_MIN and vw[i] >= RAW_MIN:
                    tr["votes"].append(vr[i])
                if use_tilt and tilts is not None and tilts[i] is not None:
                    tr["tilt"].append(tilts[i])
            if use_static and stats is not None and stats[i] is not None:
                pur, edg, ok = stats[i]
                if ok:
                    tr["static"].append((pur, edg))

    cand = [tr for tr in tracks
            if len(tr["votes"]) >= min_votes
            and float(np.median(tr["votes"])) <= ratio_lo]
    n_motion = len(cand)
    n_occupy_saved = n_occupy_boxes = 0
    if use_static:
        seen = set(id(tr) for tr in cand)
        static_cands = [tr for tr in tracks
                        if id(tr) not in seen and tr["static"]
                        and _static_says_scene(tr["static"], static_margin,
                                               min_votes)]
        if use_occupy and static_cands:
            occ = occupancy(per_frame, [tr["box"] for tr in static_cands],
                            occupy_iou)
            kept = [tr for tr, o in zip(static_cands, occ) if o < occupy_min]
            saved = [tr for tr, o in zip(static_cands, occ) if o >= occupy_min]
            n_occupy_saved = len(saved)
            n_occupy_boxes = sum(len(tr["members"]) for tr in saved)
            static_cands = kept
            if anchors_out is not None:
                anchors_out.extend(merge_anchors(
                    [_median_box(per_frame, tr["members"]) for tr in saved]))
        for tr in static_cands:
            seen.add(id(tr))
            cand.append(tr)

    tilt_cands = []
    if use_tilt:
        for tr in cand[:n_motion]:
            if tr["tilt"] and len(tr["tilt"]) >= min_votes \
                    and float(np.median(tr["tilt"])) >= tilt_min:
                tilt_cands.append(tr)

    panel_cands = []
    if use_panel:
        for tr in tracks:
            if len(tr["panel"]) >= max(min_votes,
                                       panel_share * len(tr["members"])):
                panel_cands.append(tr)

    motions = [f["motion"] for f in per_frame if f["motion"] is not None]
    video_motion = float(np.median(motions)) if motions else 0.0
    n_frames = max(1, len(per_frame))
    density = sum(len(tr["members"]) for tr in cand[:n_motion]) / n_frames
    allow_drop = (video_motion >= motion_min_video) and (density >= min_scene_density)
    allow_tilt = video_motion >= motion_min_video
    allow_panel = video_motion >= motion_min_video

    if report is not None:
        report.update(video_motion=video_motion, scene_density=density,
                      n_candidates=len(cand) + len(tilt_cands) + len(panel_cands),
                      candidate_boxes=sum(len(tr["members"]) for tr in cand)
                      + sum(len(tr["members"]) for tr in tilt_cands)
                      + sum(len(tr["members"]) for tr in panel_cands),
                      n_cand_motion=n_motion,
                      n_cand_static=len(cand) - n_motion,
                      static_boxes=sum(len(tr["members"]) for tr in cand[n_motion:]),
                      n_cand_tilt=len(tilt_cands),
                      tilt_boxes=sum(len(tr["members"]) for tr in tilt_cands),
                      n_cand_panel=len(panel_cands),
                      panel_boxes=sum(len(tr["members"]) for tr in panel_cands),
                      n_occupy_saved=n_occupy_saved,
                      occupy_boxes=n_occupy_boxes,
                      allow_drop=bool(allow_drop), allow_tilt=bool(allow_tilt),
                      allow_panel=bool(allow_panel))

    out = [set() for _ in per_frame]
    drop = set(id(tr) for tr in cand) if allow_drop else set()
    if allow_tilt:
        drop |= set(id(tr) for tr in tilt_cands)
    if allow_panel:
        drop |= set(id(tr) for tr in panel_cands)
    for tr in tracks:
        if id(tr) in drop:
            continue
        for t, i in tr["members"]:
            out[t].add(i)
    if tracks_out is not None:
        tracks_out.extend(tr["members"] for tr in tracks if id(tr) not in drop)
    return out
