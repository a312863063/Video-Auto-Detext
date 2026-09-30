# -*- coding: utf-8 -*-
"""编排：把「掩码」和「擦除」两段串成一条流水线，并处理批量的并发。

掩码阶段可以 `--jobs` 并行，擦除阶段显存独占、只能串行。
"""
import multiprocessing as mp
import os
import shutil
import time
import traceback

from . import config, ffmpeg
from .config import RunOptions


config.apply_runtime_env()


def collect_inputs(path):
    """`path` 是文件就返回它自己；是目录就返回目录下所有视频（按名排序）。"""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [(os.path.splitext(os.path.basename(path))[0], path)]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"输入不存在：{path}")
    out = []
    for f in sorted(os.listdir(path)):
        if os.path.splitext(f)[1].lower() in config.VIDEO_EXTS:
            out.append((os.path.splitext(f)[0], os.path.join(path, f)))
    return out


def resolve_out_dir(inp, base_out):
    """输入是目录时，产物放进 `<base_out>/<目录名>/`；是文件就直接放 `<base_out>/`。"""
    inp = os.path.normpath(os.path.abspath(inp))
    base_out = os.path.abspath(base_out)
    if os.path.isdir(inp):
        return os.path.join(base_out, os.path.basename(inp))
    return base_out


def _ensure_dir(path):
    """按需建目录。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def paths_for(name, src, out_dir, opts: RunOptions):
    """一个视频的全部产物路径。扩展名跟着源走（`.mov` 进就 `.mov` 出）。"""
    ext = os.path.splitext(src)[1]
    w = os.path.join(out_dir, config.WORK_DIRNAME, name)
    return dict(
        dir=w,
        work=w,
        pure=os.path.join(w, "mask.mp4"),
        erased=os.path.join(w, "erased.mp4"),
        mask_out=os.path.join(out_dir, config.MASK_RESULT_DIRNAME, name + ext),
        final_out=os.path.join(out_dir, config.FINAL_RESULT_DIRNAME, name + ext),
    )


def run_mask(src, p, opts: RunOptions, log=print):
    """掩码阶段：出 `mask_result/`，并在 `work/` 留一份纯掩码给擦除阶段用。"""
    from .mask import runner as mask_runner

    pure, deliverable = (p["pure"], p["mask_out"]) if opts.compare \
        else (p["pure"], None)
    _ensure_dir(pure)
    _ensure_dir(p["mask_out"])
    if deliverable:
        _ensure_dir(deliverable)
    os.makedirs(p["work"], exist_ok=True)
    ffmpeg.cleanup_parts(p["work"])
    info = ffmpeg.probe(src)
    rep = mask_runner.extract(src, pure, opts.mask,
                              compare_path=deliverable, info=info)
    if not opts.compare:
        shutil.copyfile(pure, p["mask_out"])
    log(f"  [mask] {mask_runner.describe(rep)}")
    return rep


def run_erase(src, p, opts: RunOptions, eraser=None, log=print):
    """擦除阶段。`eraser` 传进来就复用（模型只加载一次），掩码只用本次运行算出来的那份。"""
    from .erase.runner import Eraser

    if not os.path.exists(p["pure"]):
        log("  [i] 本次运行还没有掩码，当场补算一遍")
        run_mask(src, p, opts, log=log)

    own = eraser is None
    if own:
        eraser = Eraser(opts.erase)
        log(f"  [i] 模型加载 {eraser.load_seconds:.0f}s")
    dst, compare_path = (p["erased"], p["final_out"]) if opts.compare \
        else (p["final_out"], None)
    _ensure_dir(dst)
    if compare_path:
        _ensure_dir(compare_path)
    info = ffmpeg.probe(src)
    rep = eraser.erase_video(src, p["pure"], dst, info=info,
                             work_dir=p["work"], compare_path=compare_path,
                             log=log)
    log(f"  [erase] {rep['frames']}帧 {rep['segments']}段 "
        f"{rep['seconds']:.0f}s = {rep['frames'] / max(rep['seconds'], 1e-6):.2f} fps")
    return rep


def cleanup_work(p, opts: RunOptions, log=print):
    """删掉一个视频的中间产物；`opts.keep_pure_mask` 打开时只留 `work/<名字>/mask.mp4`。"""
    if not os.path.isdir(p["work"]):
        return
    if opts.keep_pure_mask:
        keep = os.path.basename(p["pure"])
        for f in os.listdir(p["work"]):
            if f == keep:
                continue
            q = os.path.join(p["work"], f)
            shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) \
                else os.remove(q)
        log("  [clean] 中间产物已清理（保留 mask.mp4 供后续 erase 复用）")
    else:
        shutil.rmtree(p["work"], ignore_errors=True)
        log("  [clean] 中间产物已清理")


def _cleanup_work_root(out_dir, log=print):
    """收掉 `work/` 目录。"""
    root = os.path.join(out_dir, config.WORK_DIRNAME)
    if not os.path.isdir(root):
        return
    left = os.listdir(root)
    if left:
        log(f"[i] work/ 暂留（{len(left)} 项，供后续 erase 复用），整批结束后删除")
        return
    try:
        os.rmdir(root)
    except OSError:
        pass


def run_one(name, src, out_dir, opts: RunOptions, eraser=None, log=print):
    """一个视频跑完 `opts.stage` 指定的阶段。"""
    p = paths_for(name, src, out_dir, opts)
    os.makedirs(p["dir"], exist_ok=True)
    t0 = time.time()
    log(f"[{name}] {os.path.basename(src)}")
    rep = dict(name=name, src=src, ok=True, error=None)
    try:
        rep["mask"] = run_mask(src, p, opts, log=log)
        if opts.stage in ("erase", "all"):
            rep["erase"] = run_erase(src, p, opts, eraser=eraser, log=log)
    except Exception as e:
        rep["ok"] = False
        rep["error"] = f"{type(e).__name__}: {e}"
        log(f"  [FAIL] {rep['error']}\n{traceback.format_exc()[-1500:]}")
    rep["seconds"] = time.time() - t0
    if rep["ok"]:
        cleanup_work(p, opts, log=log)
        log(f"[{name}] 完成，用时 {rep['seconds']:.0f}s")
    return rep


def _mask_worker(job):
    """掩码阶段的多进程 worker（Windows 用 spawn，必须是顶层可 pickle 的函数）。"""
    name, src, out_dir, opts = job
    return run_one(name, src, out_dir, opts, log=lambda *a, **k: None)


def run(payload: RunOptions, log=print):
    """一次运行的入口，返回每个视频的统计列表。"""
    inputs = collect_inputs(payload.input)
    if payload.only:
        want = {w.strip() for w in payload.only.split(",") if w.strip()}
        inputs = [x for x in inputs if x[0] in want]
    if not inputs:
        raise SystemExit("没有匹配的视频")

    out_dir = os.path.abspath(payload.out_dir) if payload.out_exact \
        else resolve_out_dir(payload.input, payload.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    log(f"[i] 输入      {os.path.abspath(payload.input)}  （{len(inputs)} 个视频）")
    log(f"[i] 输出      {out_dir}")
    log(f"[i] 产物      {config.MASK_RESULT_DIRNAME}/ 与 "
        f"{config.FINAL_RESULT_DIRNAME}/（文件名用原名，不带后缀）")
    log(f"[i] 阶段      {payload.stage}"
        + ("   ← erase 与 all 等价，都会重算掩码并同时产出 mask_result/"
           if payload.stage in ("erase", "all") else ""))
    log(f"[i] 保存形式  {'左右对照拼接' if payload.compare else '单独视频'}")
    if payload.stage in ("mask", "all"):
        log(f"[i] 掩码      长边封顶{payload.mask.max_long_side}，"
            f"背景判决={'开' if payload.mask.use_bg else '关'}，"
            f"粘住={'开' if payload.mask.use_sticky else '关'}，"
            f"静态剔除={'开' if payload.mask.use_static else '关'}，"
            f"倾角剔除={'开' if payload.mask.use_tilt else '关'}")
    if payload.stage in ("erase", "all"):
        e = payload.erase
        log(f"[i] 擦除      ckpt={e.ckpt} max_img_size={e.max_img_size} "
            f"mask_dilation={e.mask_dilation_iter} 预降采样长边={e.long_side}")

    t0 = time.time()
    results = []

    if payload.stage == "mask":
        if payload.jobs <= 1 or len(inputs) == 1:
            for i, (name, src) in enumerate(inputs, 1):
                log(f"\n[{i}/{len(inputs)}]")
                results.append(run_one(name, src, out_dir, payload, log=log))
        else:
            jobs = [(n, s, out_dir, payload) for n, s in inputs]
            log(f"\n[i] 并发 {payload.jobs} 个进程")
            with mp.get_context("spawn").Pool(processes=payload.jobs) as pool:
                for i, r in enumerate(pool.imap_unordered(_mask_worker, jobs), 1):
                    log(f"[{i}/{len(jobs)}] {r['name']} "
                        f"{'完成' if r['ok'] else 'FAIL: ' + str(r['error'])}")
                    results.append(r)
    else:
        eraser = None
        if payload.stage in ("erase", "all"):
            from .erase.runner import Eraser
            log("")
            log("[i] 加载模型（只加载一次，后续视频复用）…")
            eraser = Eraser(payload.erase)
            log(f"[i] 模型就绪，用时 {eraser.load_seconds:.0f}s")
        for i, (name, src) in enumerate(inputs, 1):
            log(f"\n[{i}/{len(inputs)}]")
            results.append(run_one(name, src, out_dir, payload,
                                   eraser=eraser, log=log))

    ok = sum(1 for r in results if r["ok"])
    log(f"\n[OK] {ok}/{len(results)} 个完成，总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    for r in results:
        if not r["ok"]:
            log(f"  [FAIL] {r['name']}: {r['error']}")
    _cleanup_work_root(out_dir, log=log)
    return results
