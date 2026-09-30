# -*- coding: utf-8 -*-
"""由框内像素分布判断这条字幕是不是「带背景」的，
并据此给出最终矩形 —— 带背景就外扩到盖住背景，不带背景就紧贴字幕。
"""
import cv2
import numpy as np

TOL = 40.0
RING_T = 2
BUCKET = 8

COV_MIN = 0.45
SIDE_MIN = 0.60
SIDES_MIN = 2
LINE_MIN = 0.70
LINE_TOL = 3
EXT_MIN = 2
GROW_FRAC = 2.5
RECT_COV_MIN = 0.85

STEP_BAND = 3
STEP_MIN = 60.0
STEP_FRAC = 0.50
STEP_SIDES_MIN = 2

TIGHT_PAD_FRAC = 0.10
TIGHT_PAD_MIN = 3
TIGHT_PAD_MAX = 8

STICKY_SHARE = 0.15
STICKY_MIN_N = 2
STICKY_NEAR = 8


def _dom_colour(pts, tol=TOL):
    """量化直方图的众数桶 → 用该桶内像素的均值细化 → 返回 (颜色, 占比)。"""
    pts = pts.astype(np.float32)
    q = (pts // BUCKET).astype(np.int32)
    key = (q[:, 0] << 10) + (q[:, 1] << 5) + q[:, 2]
    vals, cnt = np.unique(key, return_counts=True)
    if vals.size == 0:
        return None, 0.0
    k = vals[int(cnt.argmax())]
    sel = key == k
    col = pts[sel].mean(0)
    d = np.linalg.norm(pts - col, axis=1)
    return col, float((d <= tol).mean()), float(cnt.max() / cnt.sum())


def border_colour(bgr, box, t=RING_T):
    """框**轮廓带**上的众数颜色 —— 沿四条边各取 t 像素在内、t 像素在外。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = bgr.shape[:2]
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, 0.0, []
    xa, xb = max(0, x0 - t), min(w, x1 + t)
    ya, yb = max(0, y0 - t), min(h, y1 + t)
    c = bgr[ya:yb, xa:xb]
    ry0, ry1 = y0 - ya, y1 - ya
    rx0, rx1 = x0 - xa, x1 - xa
    hy0, hy1 = max(0, ry0 - t), min(c.shape[0], ry0 + t)
    by0, by1 = max(0, ry1 - t), min(c.shape[0], ry1 + t)
    lx0, lx1 = max(0, rx0 - t), min(c.shape[1], rx0 + t)
    rx0b, rx1b = max(0, rx1 - t), min(c.shape[1], rx1 + t)
    strips = [c[hy0:hy1, lx0:rx1b].reshape(-1, 3),
              c[by0:by1, lx0:rx1b].reshape(-1, 3),
              c[hy0:by1, lx0:lx1].reshape(-1, 3),
              c[hy0:by1, rx0b:rx1b].reshape(-1, 3)]
    strips = [s for s in strips if s.size]
    if not strips:
        return None, 0.0, []
    col, cov, _ = _dom_colour(np.concatenate(strips, 0))
    if col is None:
        return None, 0.0, []
    sides = []
    for s in strips:
        d = np.linalg.norm(s.astype(np.float32) - col, axis=1)
        sides.append(float((d <= TOL).mean()))
    return col, cov, sides


def _line_ok(seg, col, frac=LINE_MIN):
    if seg.size == 0:
        return False
    d = np.linalg.norm(seg.astype(np.float32) - col, axis=1)
    return float((d <= TOL).mean()) >= frac


def scan_outward(bgr, box, col, cap_x, cap_y):
    """从框的四边往外扫，直到某一行/列不再是该颜色为止。

    返回 (ext, capped)：ext 是 {up,down,left,right} 的像素数，
    capped 是被上限截断的方向集合。
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = bgr.shape[:2]
    xa, xb = max(0, x0), min(W, x1)
    ya, yb = max(0, y0), min(H, y1)
    ext, capped = {}, set()

    def walk(name, start, step, limit, line_fn, hi):
        n, v, fails, last_ok = 0, start, 0, None
        while n < limit and 0 <= v < hi:
            if line_fn(v):
                last_ok, fails = v, 0
            else:
                fails += 1
                if fails > LINE_TOL:
                    break
            n += 1
            v += step

        ext[name] = 0 if last_ok is None else abs(last_ok - start) + 1
        if limit > 0 and n >= limit:
            capped.add(name)

    def row_fn(y):
        return _line_ok(bgr[y, xa:xb], col)

    def col_fn(x):
        return _line_ok(bgr[ya:yb, x], col)

    walk("up", y0 - 1, -1, cap_y, row_fn, H)
    walk("down", y1, +1, cap_y, row_fn, H)
    walk("left", x0 - 1, -1, cap_x, col_fn, W)
    walk("right", x1, +1, cap_x, col_fn, W)
    return ext, capped


def _grow(box, ext):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return [x0 - ext["left"], y0 - ext["up"], x1 + ext["right"], y1 + ext["down"]]


def _rect_cover(bgr, rect, col, skip):
    """外扩出来的那一圈里，该颜色占多少（跳过原框，原框里是字）。"""
    x0, y0, x1, y1 = rect
    H, W = bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    c = bgr[y0:y1, x0:x1]
    d = np.linalg.norm(c.astype(np.float32) - col, axis=2) <= TOL
    sx0, sy0, sx1, sy1 = [int(round(v)) for v in skip]
    m = np.ones(d.shape, bool)
    m[max(0, sy0 - y0):max(0, sy1 - y0), max(0, sx0 - x0):max(0, sx1 - x0)] = False
    if not m.any():
        return 0.0
    return float(d[m].mean())


def _has_step(bgr, rect, box, col, direction, band=STEP_BAND):
    """块朝 direction 延伸到头之后，紧贴边界再往外是不是**换了颜色**。

    返回带里「明显不是块颜色」的像素比例。
    """
    x0, y0, x1, y1 = rect
    bx0, by0, bx1, by1 = [int(round(v)) for v in box]
    H, W = bgr.shape[:2]
    if direction == "up":
        a = bgr[max(0, y0 - band):max(0, y0), max(0, x0):min(W, x1)]
    elif direction == "down":
        a = bgr[min(H, y1):min(H, y1 + band), max(0, x0):min(W, x1)]
    elif direction == "left":
        a = bgr[max(0, y0):min(H, y1), max(0, x0 - band):max(0, x0)]
    else:
        a = bgr[max(0, y0):min(H, y1), min(W, x1):min(W, x1 + band)]
    if a.size == 0:
        return 0.0
    d = np.linalg.norm(a.reshape(-1, 3).astype(np.float32) - col, axis=1)
    return float((d > STEP_MIN).mean())


def tight_box(box, pad_frac=TIGHT_PAD_FRAC):
    """紧贴模式：DB 框 + 一点描边/抗锯齿余量。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    pad = int(round(pad_frac * (y1 - y0)))
    pad = max(TIGHT_PAD_MIN, min(TIGHT_PAD_MAX, pad))
    return [x0 - pad, y0 - pad, x1 + pad, y1 + pad]


def analyse(bgr, box, return_info=False):
    """一个框 → 最终矩形。返回 (矩形, info) 或只返回矩形。

    info["has_bg"] 是判决，info["why"] 是依据（便于排查）。
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    info = dict(box=[x0, y0, x1, y1], has_bg=False, why="", ring=None,
                cov=0.0, sides=[], ext=None, grown=None, steps=None)

    def tight(why):
        info["why"] = why
        return ([tight_box(box), info] if return_info else tight_box(box))

    bw, bh = x1 - x0, y1 - y0
    if bw < 3 or bh < 3:
        return tight("框太小")

    col, cov, sides = border_colour(bgr, box)
    info["ring"] = None if col is None else col.astype(int).tolist()
    info["cov"] = round(cov, 3)
    info["sides"] = [round(s, 2) for s in sides]
    if col is None:
        return tight("取不到 ring")
    n_sides = sum(1 for s in sides if s >= SIDE_MIN)
    if cov < COV_MIN or n_sides < SIDES_MIN:
        return tight("框边颜色不统一(cov=%.2f,n=%d)" % (cov, n_sides))

    cap_x = max(EXT_MIN + 1, int(round(GROW_FRAC * bw)))
    cap_y = max(EXT_MIN + 1, int(round(GROW_FRAC * bh)))
    ext, capped = scan_outward(bgr, box, col, cap_x, cap_y)
    info["ext"] = ext
    if capped:
        return tight("颜色无界(%s撞上限) → 是场景不是块" % ",".join(sorted(capped)))

    got = [k for k, v in ext.items() if v >= EXT_MIN]

    vert = [k for k in got if k in ("up", "down")]
    horz = [k for k in got if k in ("left", "right")]
    if not vert or not horz:
        return tight("外扩不足(%s)" % (",".join(got) or "none"))

    rect = _grow(box, ext)
    cover = _rect_cover(bgr, rect, col, box)
    info["grown"] = rect
    info["cover"] = round(cover, 3)
    if cover < RECT_COV_MIN:
        return tight("外扩圈不纯(%.2f)" % cover)

    steps = {k: round(_has_step(bgr, rect, box, col, k), 2) for k in got}
    info["steps"] = steps
    n_step = sum(1 for v in steps.values() if v >= STEP_FRAC)
    if n_step < STEP_SIDES_MIN:
        return tight("延伸出去不是硬边(%s) → 是场景不是块" % steps)

    info["has_bg"] = True
    info["why"] = "带背景 cov=%.2f 外扩=%s 硬边=%s" % (cov, ext, steps)
    return ([rect, info] if return_info else rect)


def refine_boxes(bgr, boxes):
    """一串框 → (最终矩形列表, info 列表)。"""
    out, infos = [], []
    for b in boxes:
        r, i = analyse(bgr, b, return_info=True)
        out.append(r)
        infos.append(i)
    return out, infos


grow = _grow


def sticky_exts(infos, share=STICKY_SHARE, min_n=STICKY_MIN_N):
    """一条轨迹逐帧的 info → 「自己判出带背景、且量到了 ext」的帧下标；不足则 None。

    返回 None 表示「证据不足，别粘」，调用方应当保持逐帧判决不动。
    """
    good = [k for k, i in enumerate(infos)
            if i.get("has_bg") and i.get("ext") is not None]
    if len(good) < min_n or len(good) < share * len(infos):
        return None
    return good


def nearest_good(good, t, near=STICKY_NEAR):
    """good 是 {帧号: 外扩量}，取离 t 最近且不超过 near 帧的那个；没有则 None。"""
    best, bd = None, None
    for a, ext in good.items():
        d = abs(a - t)
        if d <= near and (bd is None or d < bd):
            best, bd = ext, d
    return best
