# -*- coding: utf-8 -*-
"""ffmpeg / ffprobe 的最小封装。"""
import os
import queue
import subprocess
import threading

import numpy as np


def _q(args, path, timeout=60):
    """跑一次 ffprobe 取一个字段。"""
    r = subprocess.run(["ffprobe", "-v", "error"] + args + [path],
                       stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def probe(path):
    """ffprobe 取分辨率 / fps / 帧数 / 有无音频 / 旋转元数据。"""
    v = _q(["-select_streams", "v:0", "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames", "-of", "csv=p=0"], path).split(",")
    if len(v) < 3:
        raise RuntimeError(f"ffprobe 无法解析 {path}: {v}")
    num, den = v[2].split("/")
    return dict(w=int(v[0]), h=int(v[1]), fps=float(num) / float(den),
                nframes=int(v[3]) if len(v) > 3 and v[3].isdigit() else 0,
                has_audio=bool(_q(["-select_streams", "a:0", "-show_entries",
                                   "stream=codec_name", "-of", "csv=p=0"], path)),
                rotation=_q(["-select_streams", "v:0", "-show_entries",
                             "side_data=rotation", "-of", "csv=p=0"], path) or None)


def probe_exact(path):
    """真解一遍数帧数，比 `probe` 慢得多。"""
    out = _q(["-select_streams", "v:0", "-count_frames", "-show_entries",
              "stream=nb_read_frames", "-of", "csv=p=0"], path)
    return int(out) if out.isdigit() else 0


def run(cmd, **kw):
    """跑一条 ffmpeg/ffprobe 命令，失败就把 stderr 尾部抛出来。"""
    kw.setdefault("stdin", subprocess.DEVNULL)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} 退出码 {r.returncode}\n{(r.stderr or '')[-1200:]}")
    return r


class FrameWriter(threading.Thread):
    """后台把帧喂给 ffmpeg 的 stdin，让编码与推理重叠。"""

    def __init__(self, cmd, maxsize=6):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=maxsize)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.err_lines, self.error = [], None
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in iter(self.proc.stderr.readline, b""):
            self.err_lines.append(line)
            if len(self.err_lines) > 200:
                del self.err_lines[:100]

    def run(self):
        try:
            while True:
                frame = self.q.get()
                if frame is None:
                    break
                self.proc.stdin.write(frame.tobytes())
        except Exception as e:
            self.error = e
        finally:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    def write(self, frame):
        """把一帧拷贝后入队。"""
        try:
            self.q.put(np.ascontiguousarray(frame).copy(), timeout=60)
            return
        except queue.Full:
            pass
        if self.error is not None:
            err = b"".join(self.err_lines).decode("utf-8", "replace")
            raise RuntimeError(f"写帧失败：{self.error}\n{err[-1200:]}") from self.error
        if not self.is_alive():
            err = b"".join(self.err_lines).decode("utf-8", "replace")
            raise RuntimeError(f"编码器线程已退出（ffmpeg 大概率启动就失败了）\n{err[-1200:]}")
        raise RuntimeError("编码器 60 秒没有消费任何一帧，判定为卡死")

    def finish(self):
        self.q.put(None)
        self.join()
        rc = self.proc.wait()
        if rc != 0:
            err = b"".join(self.err_lines).decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg 退出码 {rc}\n{err[-1200:]}")
        if self.error:
            raise self.error

    def abort(self):
        try:
            self.proc.kill()
        except Exception:
            pass


def frame_encoder(dst, size, fps, audio_from=None, crf=16, preset="medium",
                  threads=4, pix_fmt="bgr24", out_pix_fmt="yuv420p"):
    """构造 ffmpeg 命令：原始帧从 stdin 进，用 x264 写出，音频从源 copy。

    `size` 是 (w, h)，指写出去的画面尺寸。
    """
    w, h = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-r", f"{fps}",
           "-i", "pipe:0"]
    if audio_from:
        cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-fps_mode", "passthrough",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-threads", str(threads), "-pix_fmt", out_pix_fmt]
    if audio_from:
        cmd += ["-c:a", "copy"]
    cmd += [dst]
    return cmd


def mux_audio(video, audio_from, dst):
    """把 `audio_from` 的音轨原样搬到目标片上（重封装，不重编码画面）。"""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio_from,
         "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", dst])
    return dst


def replace_atomic(part_path, final_path):
    """先写 `.part.<ext>` 成功后再原子改名。"""
    os.replace(part_path, final_path)


def part_path(dst):
    root, ext = os.path.splitext(dst)
    return f"{root}.part{ext}"


def cleanup_parts(directory):
    """清掉上次跑崩留下的 `.part.*`。"""
    n = 0
    if not os.path.isdir(directory):
        return 0
    for f in os.listdir(directory):
        if ".part." in f:
            try:
                os.remove(os.path.join(directory, f))
                n += 1
            except OSError:
                pass
    return n
