# -*- coding: utf-8 -*-
"""资源监控 + OOM 硬闸门。

后台定期采样内存 / 显存 / 磁盘，越线就杀掉本进程树。
"""
import json
import os
import shutil
import subprocess
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None

MEM_WARN_PCT = 88.0
COMMIT_KILL_RATIO = 0.90
RSS_KILL_GB = 12.0
VRAM_WARN_GB = 20.0
VRAM_KILL_GB = 21.5
DISK_KILL_GB = 5.0
DISK_WARN_GB = 12.0
KILL_STREAK = 3
INTERVAL = 5.0

EXIT_OOM = 86


def _kill_children():
    """自杀之前先收掉子进程。"""
    if not psutil:
        return
    try:
        for p in psutil.Process(os.getpid()).children(recursive=True):
            try:
                p.kill()
            except Exception:
                pass
    except Exception:
        pass


def _nvidia():
    """整卡显存用量 (used_mb, total_mb)；取不到就 (None, None)。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=10).stdout.strip().splitlines()
        if not out:
            return None, None
        used, total = out[0].split(",")
        return float(used), float(total)
    except Exception:
        return None, None


class Monitor:
    """后台线程：定期采样、写 JSONL、越线就杀进程。"""

    def __init__(self, log_path, label="", interval=INTERVAL, disk_path=None):
        self.log_path = log_path
        self.label = label
        self.interval = interval
        self.disk_path = disk_path or os.path.dirname(os.path.abspath(log_path)) or "."
        self.peak_vram_gb = 0.0
        self.peak_mem_pct = 0.0
        self.peak_commit_pct = 0.0
        self.peak_rss_gb = 0.0
        self.min_disk_gb = -1.0
        self.n_samples = 0
        self._streak = 0
        self._stop = threading.Event()
        self._thread = None
        self._fh = None

    def sample(self):
        vm = psutil.virtual_memory() if psutil else None
        sw = psutil.swap_memory() if psutil else None
        mem_pct = vm.percent if vm else 0.0
        if vm and sw:
            commit_gb = (vm.used + sw.used) / 1e9
            commit_limit_gb = (vm.total + sw.total) / 1e9
            commit_pct = 100.0 * commit_gb / commit_limit_gb
        else:
            commit_gb = commit_limit_gb = commit_pct = 0.0
        rss_gb = 0.0
        if psutil:
            try:
                rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
            except Exception:
                pass
        used_mb, total_mb = _nvidia()
        vram_gb = (used_mb / 1024.0) if used_mb is not None else 0.0
        try:
            disk_gb = shutil.disk_usage(self.disk_path).free / 1e9
        except Exception:
            disk_gb = -1.0
        rec = dict(t=time.time(), label=self.label, mem_pct=round(mem_pct, 1),
                   commit_pct=round(commit_pct, 1), commit_gb=round(commit_gb, 1),
                   commit_limit_gb=round(commit_limit_gb, 1),
                   rss_gb=round(rss_gb, 2), vram_gb=round(vram_gb, 2),
                   vram_total_gb=round(total_mb / 1024.0, 1) if total_mb else None,
                   disk_free_gb=round(disk_gb, 1))
        self.peak_mem_pct = max(self.peak_mem_pct, mem_pct)
        self.peak_commit_pct = max(self.peak_commit_pct, commit_pct)
        self.peak_rss_gb = max(self.peak_rss_gb, rss_gb)
        self.peak_vram_gb = max(self.peak_vram_gb, vram_gb)
        self.min_disk_gb = disk_gb if self.min_disk_gb < 0 else min(self.min_disk_gb, disk_gb)
        self.n_samples += 1
        return rec

    def _loop(self):
        while not self._stop.wait(self.interval):
            rec = self.sample()
            try:
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._fh.flush()
            except Exception:
                pass
            why = None
            if rec["commit_pct"] >= COMMIT_KILL_RATIO * 100:
                why = f"提交内存 {rec['commit_gb']}/{rec['commit_limit_gb']}G "\
                      f"({rec['commit_pct']}%)"
            elif rec["rss_gb"] >= RSS_KILL_GB:
                why = f"本进程 RSS {rec['rss_gb']}G"
            elif rec["vram_gb"] and rec["vram_gb"] >= VRAM_KILL_GB:
                why = f"显存 {rec['vram_gb']}G"
            elif 0 <= rec["disk_free_gb"] < DISK_KILL_GB:
                why = f"磁盘仅剩 {rec['disk_free_gb']}G"
            if why:
                self._streak += 1
                print(f"[MONITOR][CRITICAL] {why} (连续 {self._streak}/{KILL_STREAK})",
                      flush=True)
                if self._streak >= KILL_STREAK:
                    print(f"[MONITOR] 连续 {KILL_STREAK} 次越线，"
                          f"主动退出以免整机换页卡死（退出码 {EXIT_OOM}）", flush=True)
                    try:
                        self._fh.write(json.dumps(dict(killed=True, why=why, **rec),
                                                  ensure_ascii=False) + "\n")
                        self._fh.flush()
                    except Exception:
                        pass
                    _kill_children()
                    os._exit(EXIT_OOM)
            else:
                self._streak = 0
                if rec["mem_pct"] >= MEM_WARN_PCT:
                    print(f"[MONITOR][info] 物理内存 {rec['mem_pct']}%（正常，"
                          f"提交 {rec['commit_pct']}%）RSS {rec['rss_gb']}G "
                          f"显存 {rec['vram_gb']}G", flush=True)
                elif 0 <= rec["disk_free_gb"] < DISK_WARN_GB:
                    print(f"[MONITOR][warn] 磁盘仅剩 {rec['disk_free_gb']}G", flush=True)

    def start(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        self._fh = open(self.log_path, "a", encoding="utf-8")
        self._fh.write(json.dumps(dict(event="start", t=time.time(),
                                       pid=os.getpid()), ensure_ascii=False) + "\n")
        self._fh.flush()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self, event="stop"):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._fh:
            try:
                self._fh.write(json.dumps(dict(event="stop", t=time.time(),
                                               peak_mem_pct=round(self.peak_mem_pct, 1),
                                               peak_commit_pct=round(self.peak_commit_pct, 1),
                                               peak_rss_gb=round(self.peak_rss_gb, 2),
                                               peak_vram_gb=round(self.peak_vram_gb, 2),
                                               min_disk_gb=round(self.min_disk_gb, 1)),
                                          ensure_ascii=False) + "\n")
                self._fh.close()
            except Exception:
                pass

    def summary(self):
        return dict(peak_mem_pct=round(self.peak_mem_pct, 1),
                    peak_commit_pct=round(self.peak_commit_pct, 1),
                    peak_rss_gb=round(self.peak_rss_gb, 2),
                    peak_vram_gb=round(self.peak_vram_gb, 2),
                    min_disk_gb=round(self.min_disk_gb, 1),
                    samples=self.n_samples)
