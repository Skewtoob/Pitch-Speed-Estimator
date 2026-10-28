"""generate_clip(): render + encode one synthetic clip and write its ground truth.

    gt = generate_clip({"camera": {"view": "behind"}, "duration_s": 8}, "out/clip.mpg")

Deterministic: the same (resolved) config gives bit-identical frames, in
serial or parallel mode (every random stream is keyed by seed and subsystem,
the tape noise by frame index).  Parallel rendering (`jobs > 1`) uses
'spawn' worker processes (Windows-safe); each worker builds its own
Renderer (background canvas ~1-5 s), so it only pays off for clips longer
than ~15 s.  As always with 'spawn', a calling *script* must guard its
entry point with `if __name__ == "__main__":` (the CLI already does).

Files: see gt.py (video, .gt.json, .calib.json [+ .calib_exact.json],
.cams.npz).  The GT is written last, so its presence marks a complete clip.
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import multiprocessing
import os
import time
from typing import Callable, Iterator, Optional

import cv2

from pitchspeed.calib.model import REQUIRED_POINTS
from pitchspeed.synth.config import n_source_frames, resolve_config
from pitchspeed.synth.encode import DEFAULT_EXT, Encoder, ffmpeg_version
from pitchspeed.synth.gt import (build_frame_map, build_gt, calib_path_for, cams_path_for, choose_reference,
                                 gt_path_for, landmarks_px, make_calibration, save_cams, video_fingerprint,
                                 write_json_atomic)
from pitchspeed.synth.render import FrameResult, Renderer

ProgressFn = Callable[[int, int], None]

# ------------------------------------------------------------------ parallel workers
_WORKER: dict = {}


def _worker_init(cfg: dict, ref_times: list) -> None:
    cv2.setNumThreads(1)
    _WORKER["renderer"] = Renderer(cfg, ref_times)


def _worker_render(frames: tuple) -> list:
    r = _WORKER["renderer"]
    return [r.render_frame(n) for n in range(frames[0], frames[1])]


def _iter_frames(cfg: dict, renderer: Optional[Renderer], ref_times: list, n_frames: int, jobs: int,
                 chunk: int = 24) -> Iterator[FrameResult]:
    """Rendered frames in order (serial, or parallel with a bounded number of chunks in flight)."""
    if jobs <= 1:
        for n in range(n_frames):
            yield renderer.render_frame(n)
        return
    ctx = multiprocessing.get_context("spawn")
    chunks = [(a, min(a + chunk, n_frames)) for a in range(0, n_frames, chunk)]
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_worker_init,
                                initargs=(cfg, ref_times)) as ex:
        pending = []
        it = iter(chunks)
        for c in it:
            pending.append(ex.submit(_worker_render, c))
            if len(pending) >= 2 * jobs:
                break
        while pending:
            fut = pending.pop(0)
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(_worker_render, nxt))
            yield from fut.result()


# ------------------------------------------------------------------ helpers
def _with_ext(path: str, codec: str) -> str:
    root, ext = os.path.splitext(path)
    return path if ext else path + DEFAULT_EXT.get(codec, ".mkv")


def _outputs(cfg: dict, out_path: str) -> list:
    """[(path, video cfg)] for the main output and video.extra_outputs (same render)."""
    V = cfg["video"]
    outs = [(_with_ext(out_path, V["codec"]), V)]
    base_dir = os.path.dirname(os.path.abspath(out_path))
    for ex in V.get("extra_outputs") or []:
        v2 = copy.deepcopy(V)
        v2.update({k: v for k, v in ex.items() if k != "out"})
        v2["extra_outputs"] = []
        p = ex.get("out") or (os.path.splitext(os.path.basename(out_path))[0] + "_" + v2["codec"])
        outs.append((_with_ext(os.path.join(base_dir, p), v2["codec"]), v2))
    return outs


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ------------------------------------------------------------------ main entry
def generate_clip(config: Optional[dict], out_path, *, jobs: int = 1, progress: Optional[ProgressFn] = None,
                  write_cams: bool = True) -> dict:
    """Render and encode a clip; write GT/calibration files next to it; return the GT dict.

    `config` is a partial config (deep-merged over config.DEFAULT_CONFIG).
    `out_path` is the video file; its extension picks the container (added
    from the codec when missing).  Raises on any encoder failure (partial
    outputs are removed).
    """
    T0 = time.perf_counter()
    cfg = resolve_config(config)
    out_path = os.fspath(out_path)
    outs = _outputs(cfg, out_path)
    for p, _ in outs:
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)
    fmap = build_frame_map(cfg)
    ref = choose_reference(cfg, fmap)
    renderer = Renderer(cfg, ref["exposure_times"])
    warnings = list(renderer.events.warnings) + list(renderer.warnings)
    if renderer.canvas.has_fence and renderer.canvas.fence_blur_px < 1.0:
        warnings.append(f"backstop fence is nearly in focus (blur {renderer.canvas.fence_blur_px:.2f} px)")
    n_src = n_source_frames(cfg)
    cap = cfg["capture"]
    drops = set(int(x) for x in cap.get("drop_frames") or [])
    dups = set(int(x) for x in cap.get("dup_frames") or [])
    tff = cfg["video"]["field_order"] == "tff"
    encoders = []
    records, cams = [], []
    t_loop = time.perf_counter()
    try:
        for p, v in outs:
            encoders.append(Encoder(v, p, tff))
        for fr in _iter_frames(cfg, renderer, ref["exposure_times"], n_src, int(jobs or 1)):
            records.extend(fr.records)
            cams.extend(fr.cams)
            if fr.n not in drops:
                buf = fr.image.tobytes()
                for _ in range(2 if fr.n in dups else 1):
                    for e in encoders:
                        e.write(buf)
            if progress is not None:
                progress(fr.n + 1, n_src)
        for e in encoders:
            e.close()
    except BaseException:
        for e in encoders:
            e.abort()
        for p, _ in outs:
            _remove_quietly(p)
        raise
    t_loop = time.perf_counter() - t_loop
    # which out frame shows each field (None: dropped or not in the output, e.g. field_discard)
    first_out = {}
    for e in fmap:
        for k in e["field_index"]:
            first_out.setdefault(k, e["out_frame"])
    for r in records:
        r["out_frame"] = first_out.get(r["field_index"])
    lm_px = landmarks_px(cfg, renderer, ref)
    ffv = ffmpeg_version()
    gt_main = None
    for p, v in outs:
        fp = video_fingerprint(p, len(fmap), renderer.W, renderer.H)
        cal_exact, cal_clicked, cinfo = make_calibration(cfg, renderer, ref, fp)
        if cinfo["omitted"] and gt_main is None:
            req = [n for n in cinfo["omitted"] if n in REQUIRED_POINTS[cinfo["view"]]]
            warnings.append(f"calibration points outside the reference frame (not written): {cinfo['omitted']}"
                            + (f"; REQUIRED points missing, the .calib.json will not validate: {req}" if req else ""))
        cpath = calib_path_for(p)
        cal_clicked.save(cpath)
        cinfo["file"] = os.path.basename(cpath)
        emit_exact = cfg["calibration"].get("emit_exact")
        if emit_exact is None:
            emit_exact = cinfo["click_noise_px"] > 0
        if emit_exact:
            epath = calib_path_for(p, exact=True)
            cal_exact.save(epath)
            cinfo["exact_file"] = os.path.basename(epath)
        if write_cams:
            save_cams(cams_path_for(p), cams, renderer)
        cfg_out = copy.deepcopy(cfg)
        cfg_out["video"] = copy.deepcopy(v)
        video = dict(path=p, file=os.path.basename(p), size=fp["size"])
        timing = dict(canvas_s=renderer.t_canvas, loop_s=t_loop, total_s=time.perf_counter() - T0,
                      jobs=int(jobs or 1), n_src_frames=n_src)
        gt = build_gt(cfg_out, renderer, records, fmap, video, cinfo, lm_px, warnings, timing, ffv)
        write_json_atomic(gt, gt_path_for(p))
        if gt_main is None:
            gt_main = gt
    return gt_main
