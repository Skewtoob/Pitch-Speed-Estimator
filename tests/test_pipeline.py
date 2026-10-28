"""Pipeline glue: field-order resolution from tracks, numbering, CLI parsing."""
from __future__ import annotations

import pytest

from pitchspeed.cli import build_parser
from pitchspeed.pipeline import renumber, resolve_field_order
from pitchspeed.types import EventTime, Pitch, PitchTrack, ScanInfo, SpeedEstimate, TrackPoint


def _scan(conf="low", source="content", mode="interlaced"):
    return ScanInfo(mode=mode, field_order="bff" if mode == "interlaced" else None,
                    confidence=conf, source=source, sample_rate=59.94)


def _track(preferred, rg, rf, n=30, conflict=False):
    return PitchTrack(track_id=0, points=[TrackPoint(s=0, t=0.0, x=0.0, y=0.0)],
                      diagnostics={"field_order": {"n": n, "preferred": preferred, "conflict": conflict,
                                                   "rms_given": rg, "rms_flipped": rf}})


def test_tracks_confirm_field_order():
    scan = _scan()
    assert resolve_field_order(scan, [_track("given", 0.45, 14.2)]) == "confirmed"
    assert scan.confidence == "high"


def test_tracks_contradict_field_order():
    scan = _scan()
    v = resolve_field_order(scan, [_track("flipped", 14.0, 0.4, conflict=True),
                                   _track("flipped", 12.0, 0.5, conflict=True)])
    assert v == "flip"
    assert scan.confidence == "low"


def test_weak_or_short_tracks_do_not_decide():
    scan = _scan()
    assert resolve_field_order(scan, [_track("given", 0.45, 0.6), _track("given", 0.4, 14.0, n=6),
                                      _track("unclear", None, None)]) == "unknown"
    assert scan.confidence == "low"


def test_mixed_evidence_is_unknown():
    scan = _scan()
    assert resolve_field_order(scan, [_track("given", 0.4, 14.0),
                                      _track("flipped", 14.0, 0.4, conflict=True)]) == "unknown"


def test_override_and_progressive_untouched():
    s1 = _scan(source="override")
    assert resolve_field_order(s1, [_track("flipped", 14.0, 0.4, conflict=True)]) == "unknown"
    s2 = _scan(mode="progressive")
    assert resolve_field_order(s2, [_track("given", 0.4, 14.0)]) == "unknown"


def _pitch(t, status="auto"):
    ev = EventTime(t=t, sigma=0.01, method="manual")
    return Pitch(number=0, status=status, release=ev, catch=EventTime(t=t + 0.5, sigma=0.01, method="manual"),
                 speed=SpeedEstimate(v_release_mph=70.0))


def test_renumber_chronological_skips_rejected():
    ps = [_pitch(9.0), _pitch(1.0, "rejected"), _pitch(5.0, "manual"), _pitch(3.0)]
    renumber(ps)
    assert [(p.release.t, p.number) for p in ps] == [(1.0, 0), (3.0, 1), (5.0, 2), (9.0, 3)]


@pytest.mark.parametrize("argv", [
    ["analyze", "v.avi", "--render", "none", "--field-order", "tff", "--start", "3", "--end", "9.5"],
    ["calibrate", "v.avi", "--view", "side", "--level", "ll", "--extension", "4.2", "--throws", "L"],
    ["run", "v.avi", "--ui-scale", "1.5"],
    ["review", "v.avi"],
    ["render", "v.avi", "--mode", "highlights", "--no-audio"],
    ["probe", "v.avi", "--par", "0.9091"],
])
def test_cli_parses(argv):
    args = build_parser().parse_args(argv)
    assert args.func is not None


class _Src:
    """Minimal SampleSource stand-in: sample s = 2*frame + k."""
    def locate(self, s):
        return (s // 2, s % 2, s % 2)


def _sim_track(kind, n=30, frame_dt=1001 / 30000, v=600.0, a=-300.0):
    """Ball at x = v t + a t^2 / 2 (px), sampled per field, timed as interlaced fields."""
    pts = []
    for s in range(n):
        f, k = s // 2, s % 2
        t_field = f * frame_dt + k * frame_dt / 2          # time the pipeline assigns
        if kind == "interlaced":
            t_true = t_field
        elif kind == "flipped":                            # real order is the other one
            t_true = f * frame_dt + (1 - k) * frame_dt / 2
        else:                                              # progressive: both fields at frame time
            t_true = f * frame_dt
        x = v * t_true + 0.5 * a * t_true ** 2
        y = 200.0 + 40.0 * t_true
        pts.append(TrackPoint(s=s, t=t_field, x=x, y=y))
    return PitchTrack(track_id=0, points=pts)


@pytest.mark.parametrize("kind,expected", [("interlaced", "confirmed"), ("flipped", "flip"),
                                           ("progressive", "progressive")])
def test_track_timing_hypotheses(kind, expected):
    scan = _scan()
    assert resolve_field_order(scan, [_sim_track(kind)], _Src(), 1001 / 30000) == expected


def test_slow_ball_is_unclear():
    scan = _scan()
    # ~0.3 px per field: no timing hypothesis can be told apart
    assert resolve_field_order(scan, [_sim_track("interlaced", v=18.0, a=0.0)], _Src(), 1001 / 30000) == "unknown"


def _tp(t, prog_last, speed=None, status="auto"):
    tr = PitchTrack(track_id=0, points=[TrackPoint(s=int(t * 60), t=t, x=0.0, y=0.0),
                                        TrackPoint(s=int(t * 60) + 20, t=t + 0.33, x=0.0, y=0.0)],
                    diagnostics={"progress_last": prog_last})
    return Pitch(number=0, status=status, release=EventTime(t=t, sigma=0.01, method="first_free_field"),
                 catch=EventTime(t=t + 0.5, sigma=0.01, method="extrapolated"),
                 speed=SpeedEstimate(v_release_mph=speed), track=tr)


def test_throwback_after_catch_is_demoted():
    from pitchspeed.pipeline import _demote_throwbacks
    rejected, missed = [], []
    ps = [_tp(2.0, 0.8, 70.0), _tp(5.0, 0.15), _tp(14.0, 0.8, 68.0)]   # catch 2.5 -> 2.5 s later
    kept = _demote_throwbacks(ps, rejected, missed)
    assert [p.release.t for p in kept] == [2.0, 14.0]
    assert rejected[0].reject_code == "THROWBACK_TIMING" and "throw back" in missed[0]["reason"]


@pytest.mark.parametrize("pitch", [
    _tp(5.0, 0.15, speed=65.0),      # has a speed: never demoted
    _tp(5.0, 0.6),                    # got far along the path
    _tp(9.0, 0.15),                   # too long after the catch
    _tp(5.0, 0.15, status="manual"),  # user-made
])
def test_throwback_rule_keeps_real_candidates(pitch):
    from pitchspeed.pipeline import _demote_throwbacks
    kept = _demote_throwbacks([_tp(2.0, 0.8, 70.0), pitch], [], [])
    assert len(kept) == 2


# =====================================================================
# Review work is never lost, capture glitches, transforms, output folders
# =====================================================================
import json                                                       # noqa: E402
import os                                                         # noqa: E402
from types import SimpleNamespace                                 # noqa: E402

import numpy as np                                                # noqa: E402

from pitchspeed import pipeline as P                              # noqa: E402
from pitchspeed.calib.model import CalibrationFile, Segment, Settings   # noqa: E402
from pitchspeed.types import AnalysisResult, VideoInfo            # noqa: E402

FDT = 1001 / 60000


def _info(n_frames=600, dup=(), drop=(), path="game.avi"):
    return VideoInfo(path=path, width=720, height=480, n_frames=n_frames, frame_rate=30000 / 1001,
                     frame_rate_str="30000/1001", duplicate_frames=list(dup), dropped_frames=list(drop))


class _FieldSrc:
    """SampleSource stand-in for interlaced fields: s = 2 * frame + k."""
    sample_dt = FDT
    interlaced = True

    def __init__(self, info=None):
        self.reader = SimpleNamespace(info=info or _info())

    def locate(self, s):
        return (int(s) // 2, int(s) % 2, int(s) % 2)

    def sample_at_time(self, t):
        return int(round(t / FDT))

    def sample_time(self, s):
        return s * FDT


def _behind_cf(ref_frame=5, end_frame=None, distance=60.5):
    seg = Segment(start_frame=0, end_frame=end_frame, ref_frame=ref_frame, view="behind",
                  points={"release": [360.0, 150.0], "catch": [362.0, 400.0]})
    return CalibrationFile(settings=Settings(distance_ft=distance), segments=[seg], image_size=(720, 480),
                           video={"name": "game.avi", "n_frames": 600, "width": 720, "height": 480})


def _ev(s, method="first_free_field", sigma=0.006):
    return EventTime(t=s * FDT, sigma=sigma, method=method, s=s)


def _trk(s0, s1, tid=1):
    pts = [TrackPoint(s=s, t=s * FDT, x=360.0, y=150.0 + 6.0 * (s - s0)) for s in range(s0, s1 + 1)]
    return PitchTrack(track_id=tid, points=pts, release=_ev(s0), catch=_ev(s1, "last_moving_field"),
                      confidence=0.9)


def _auto(s0, s1, tid=1):
    tr = _trk(s0, s1, tid)
    return Pitch(number=0, status="auto", release=tr.release, catch=tr.catch,
                 speed=SpeedEstimate(v_release_mph=70.0), frame_release=s0 // 2, frame_catch=s1 // 2,
                 track=tr, confidence=0.9)


def _result(pitches, cf=None, scan=None, info=None, missed=(), rejected=()):
    cf = cf or _behind_cf()
    return AnalysisResult(video=info or _info(), scan=scan or _scan(conf="high"), calibration=cf.to_dict(),
                          params={}, pitches=list(pitches), rejected_tracks=list(rejected),
                          possible_missed=[dict(m) for m in missed])


# ------------------------------------------------- duplicate / dropped frames
def test_frame_glitch_flags_cover_the_flight_plus_one_frame():
    assert P.frame_glitch_flags(100, 120, duplicate_frames=[110]) == ["duplicate_frame_suspected"]
    assert P.frame_glitch_flags(100, 120, duplicate_frames=[99, 121]) == ["duplicate_frame_suspected"]
    assert P.frame_glitch_flags(100, 120, duplicate_frames=[98, 122]) == []
    assert P.frame_glitch_flags(100, 120, dropped_frames=[100]) == ["dropped_frame_suspected"]
    assert P.frame_glitch_flags(100, 120, dropped_frames=[121]) == ["dropped_frame_suspected"]
    assert P.frame_glitch_flags(100, 120, dropped_frames=[99, 122]) == []
    assert P.frame_glitch_flags(-1, 120, duplicate_frames=[110]) == []
    assert P.frame_glitch_flags(100, 120, [110], [105]) == ["duplicate_frame_suspected", "dropped_frame_suspected"]


def test_pipeline_sets_glitch_flags_on_pitch_and_track():
    src = _FieldSrc(_info(dup=[55]))
    cf = _behind_cf()
    tr = _trk(100, 130)
    p = P._pitch_from_track(tr, src, cf.segment_calibs()[0], _scan(conf="high"))
    assert "duplicate_frame_suspected" in p.flags and "duplicate_frame_suspected" in tr.flags
    # the estimator consumes the track flag: wider timing, flag on the speed
    assert "dropped_frame_suspected" in p.speed.flags
    clean = P._pitch_from_track(_trk(100, 130), _FieldSrc(_info()), cf.segment_calibs()[0], _scan(conf="high"))
    assert not set(P.GLITCH_FLAGS) & set(clean.flags)
    assert p.speed.sigma_mph > clean.speed.sigma_mph


def test_recompute_moves_glitch_flags_with_the_edited_range():
    """Review edit: flags follow the new R/C, also for a hand-marked pitch without a track."""
    cf = _behind_cf()
    video = _info(dup=[210])                      # recorded during the analysis (results.json)
    src = _FieldSrc(_info())                      # a fresh reader has not seen the duplicate
    rc = P.make_recompute(cf, _scan(conf="high"), src, video=video)
    p = Pitch(number=1, status="manual", release=_ev(400, "manual", FDT / 3.5), catch=_ev(430, "manual", FDT / 3.5),
              speed=SpeedEstimate(v_release_mph=None))
    rc(p)
    assert p.frame_release == 200 and p.frame_catch == 215
    assert "duplicate_frame_suspected" in p.flags
    assert "dropped_frame_suspected" in p.speed.flags          # reached the estimator without a track
    p.release, p.catch = _ev(460, "manual", FDT / 3.5), _ev(490, "manual", FDT / 3.5)
    rc(p)
    assert not set(P.GLITCH_FLAGS) & set(p.flags)
    assert "dropped_frame_suspected" not in p.speed.flags
    assert p.track is None                                     # the stand-in track is never stored


# ------------------------------------------------------------- transforms
def test_load_transforms_prefers_homographies(tmp_path):
    d = str(tmp_path)
    A = np.tile(np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]), (4, 1, 1))
    H = np.tile(np.array([[1.0, 0.01, 2.0], [0.0, 1.0, -1.0], [1e-5, 0.0, 1.0]]), (4, 1, 1))
    s = np.array([0, 1, 2, 3])
    np.savez_compressed(P.out_paths(d)["transforms"], samples=s, transforms=A, transforms_3x3=H)
    T, ts = P.load_transforms(d)
    assert T.shape == (4, 3, 3) and np.allclose(T, H) and list(ts) == [0, 1, 2, 3]
    np.savez_compressed(P.out_paths(d)["transforms"], samples=s, transforms=A)       # older file
    T, _ = P.load_transforms(d)
    assert T.shape == (4, 2, 3)
    np.savez_compressed(P.out_paths(d)["transforms"], samples=s, transforms=A, transforms_3x3=H[:2])
    assert P.load_transforms(d)[0].shape == (4, 2, 3)                                 # inconsistent 3x3 set
    with open(P.out_paths(d)["transforms"], "wb") as f:
        f.write(b"not a zip file")
    assert P.load_transforms(d) == (None, None)
    assert P.load_transforms(str(tmp_path / "nowhere")) == (None, None)


# ---------------------------------------------------------- output folders
def _write_results_for(out_dir, video_name):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"video": {"path": video_name}, "pitches": []}, f)


def test_same_stem_other_extension_gets_its_own_folder(tmp_path):
    avi, mpg = tmp_path / "game 1.avi", tmp_path / "game 1.mpg"
    avi.write_bytes(b"x")
    mpg.write_bytes(b"y")
    _write_results_for(str(tmp_path / "game 1_pitchspeed"), str(avi))
    d, note = P.resolve_out_dir(str(mpg))
    assert d == str(tmp_path / "game 1_mpg_pitchspeed") and "game 1.avi" in note
    assert P.resolve_out_dir(str(avi)) == (str(tmp_path / "game 1_pitchspeed"), None)
    # the other file is gone (converted and deleted): keep the folder, the fingerprint check warns
    avi.unlink()
    assert P.resolve_out_dir(str(mpg))[0] == str(tmp_path / "game 1_pitchspeed")


def test_calibration_fingerprint_problems(tmp_path):
    v = tmp_path / "game.avi"
    v.write_bytes(b"0" * 100)
    cf = _behind_cf()
    cf.video = P.video_fingerprint(str(v), _info())
    assert P.calibration_video_problems(cf, str(v), _info()) == ([], [])
    errs, _ = P.calibration_video_problems(cf, str(v), VideoInfo(path="", width=640, height=480, n_frames=600,
                                                                 frame_rate=29.97, frame_rate_str="30000/1001"))
    assert errs and "640x480" in errs[0]
    _, warns = P.calibration_video_problems(cf, str(v), _info(n_frames=585))
    assert any("585" in w for w in warns)
    other = tmp_path / "game.mpg"
    other.write_bytes(b"0" * 100)
    _, warns = P.calibration_video_problems(cf, str(other), _info())
    assert any("game.avi" in w for w in warns)
    v.write_bytes(b"0" * 101)
    assert P.calibration_video_problems(cf, str(v), _info())[1] == [P.FILE_CHANGED_WARNING]
    cf.video = {}                                       # old calibration without a fingerprint
    assert P.calibration_video_problems(cf, str(v), _info()) == ([], [])


# ------------------------------------------------------------ saving / backups
def test_locked_csv_is_reported_not_fatal(tmp_path, monkeypatch):
    from pitchspeed.render import report
    d = str(tmp_path)
    r = _result([_auto(100, 130)])
    real = os.replace

    def locked(src, dst):
        if str(dst).endswith(".csv"):
            raise PermissionError(13, "locked", dst)
        return real(src, dst)
    monkeypatch.setattr(report.os, "replace", locked)
    msg = P.save_results(r, d, csv_retry_s=0.0)
    assert msg and msg.startswith(P.CSV_WARNING_PREFIX) and "results.json is up to date" in msg
    assert os.path.exists(P.out_paths(d)["results_json"]) and not os.path.exists(P.out_paths(d)["results_csv"])
    assert r.warnings[-1] == msg                                     # shown in the table ...
    saved = json.load(open(P.out_paths(d)["results_json"], encoding="utf-8"))
    assert not any(w.startswith(P.CSV_WARNING_PREFIX) for w in saved["warnings"])   # ... never saved
    monkeypatch.setattr(report.os, "replace", real)
    assert P.save_results(r, d) is None
    assert not any(w.startswith(P.CSV_WARNING_PREFIX) for w in r.warnings)
    assert os.path.exists(P.out_paths(d)["results_csv"])


def test_previous_results_backed_up_only_with_review_work(tmp_path):
    d = str(tmp_path)
    paths = P.out_paths(d)
    P.save_results(_result([_auto(100, 130)]), d)
    np.savez_compressed(paths["transforms"], samples=np.arange(2), transforms=np.zeros((2, 2, 3)))
    prev = P.PreviousResults.load(paths)
    assert prev.result is not None and prev.backup is None
    edited = _auto(100, 130)
    edited.notes = "my note"
    P.save_results(_result([edited]), d)
    prev = P.PreviousResults.load(paths)
    assert prev.backup and os.path.basename(prev.backup).startswith("results.backup-")
    assert json.load(open(prev.backup, encoding="utf-8"))["pitches"][0]["notes"] == "my note"
    assert os.path.exists(os.path.splitext(prev.backup)[0] + ".transforms.npz")
    assert P.PreviousResults.load(paths).backup != prev.backup                      # never overwritten
    with open(paths["results_json"], "w") as f:
        f.write("{ damaged")
    prev = P.PreviousResults.load(paths)
    assert prev.unreadable and prev.backup and open(prev.backup).read() == "{ damaged"


def test_user_work_detection():
    r = _result([_auto(100, 130)])
    assert not P.has_user_work(r)
    r.pitches[0].edit_history.append({"action": "accept"})
    assert P.has_user_work(r)
    r = _result([_auto(100, 130)], missed=[{"t_start": 1.0, "t_end": 2.0, "review": "no_pitch"}])
    assert P.user_work_summary(r)["windows"] == 1


# ------------------------------------------------------------- carry over
def _carry(old, new, cf=None):
    return P.carry_over_user_work(old, new, cf or _behind_cf(), _FieldSrc())


def test_carry_over_rejection_manual_and_edits():
    rej = _auto(100, 130)
    rej.status, rej.notes = "rejected", "throw back"
    rej.edit_history = [{"action": "reject"}]
    man = Pitch(number=0, status="manual", release=_ev(400, "manual"), catch=_ev(430, "manual"),
                speed=SpeedEstimate(v_release_mph=66.0), edit_history=[{"action": "create_manual"}])
    ed = _auto(700, 731)
    ed.status, ed.release = "edited", _ev(702, "manual")
    ed.edit_history = [{"action": "set_release"}]
    acc = _auto(1000, 1030)
    acc.edit_history = [{"action": "accept"}]
    untouched = _auto(1300, 1330)
    old = _result([rej, man, ed, acc, untouched],
                  missed=[{"t_start": 20.0, "t_end": 21.0, "review": "no_pitch"}])
    new = _result([_auto(101, 131), _auto(699, 730), _auto(1001, 1031), _auto(1300, 1330), _auto(1600, 1630)],
                  missed=[{"t_start": 20.2, "t_end": 20.9}, {"t_start": 30.0, "t_end": 31.0}])
    rep = _carry(old, new)
    assert rep.reason is None and rep.rejected == 1 and rep.pitches == 3 and rep.windows == 1
    by_t = sorted(new.pitches, key=lambda p: p.release.t)
    st = [(p.release.s, p.status) for p in by_t]
    assert st == [(101, "rejected"), (400, "manual"), (702, "edited"), (1001, "auto"), (1300, "auto"),
                  (1600, "auto")]
    assert by_t[0].notes == "throw back" and by_t[0].number == 0
    assert by_t[2].release.method == "manual" and by_t[2].edit_history[-1]["action"] == "carried_over"
    assert by_t[3].edit_history[0]["action"] == "accept"
    assert [p.number for p in by_t] == [0, 1, 2, 3, 4, 5]
    # the automatic pitch the edited one replaced is listed, not dropped
    assert [t.reject_code for t in new.rejected_tracks] == ["REPLACED_BY_REVIEW"]
    assert new.possible_missed[0]["review"] == "no_pitch" and "review" not in new.possible_missed[1]
    assert by_t[1].speed is not None and by_t[1].frame_release == 200      # recomputed in the new space


def test_carry_over_refused_when_indices_change_meaning():
    ed = _auto(100, 130)
    ed.status = "edited"
    old = _result([ed])
    new = _result([_auto(100, 130)], scan=_scan(conf="high", mode="progressive"))
    rep = _carry(old, new)
    assert rep.reason and "frames" in rep.reason and [p.status for p in new.pitches] == ["auto"]
    rep = _carry(_result([ed]), _result([_auto(100, 130)], info=_info(n_frames=590)))
    assert rep.reason and "590" in rep.reason
    rep = _carry(_result([ed]), _result([_auto(100, 130)]), cf=_behind_cf(ref_frame=99))
    assert rep.reason and "reference" in rep.reason


def test_recompute_all_applies_new_settings_and_keeps_edits():
    p = _auto(100, 131)
    p.status, p.notes = "edited", "keep me"
    r = _result([p], cf=_behind_cf(distance=60.5))
    cf46 = _behind_cf(distance=46.0)
    changes, clicks = P.calibration_differences(cf46, r)
    assert clicks is False and any("60.5 -> 46" in c for c in changes)
    assert P.recompute_all(r, cf46, _FieldSrc()) == 1
    assert r.calibration["settings"]["distance_ft"] == 46.0
    assert r.pitches[0].notes == "keep me" and r.pitches[0].status == "edited"
    assert r.pitches[0].speed.assumptions.get("distance_ft") == 46.0
    assert P.calibration_differences(cf46, r) == ([], False)
    assert P.calibration_differences(_behind_cf(ref_frame=7, distance=46.0), r)[1] is True


def test_undecodable_frames_count_as_timing_gaps():
    reader = SimpleNamespace(info=_info(), decode_errors=[{"frame": 7, "error": "x"},
                                                          {"frames": [210, 212], "error": "y"},
                                                          {"pts": 99, "error": "not in the index"}])
    assert P.undecodable_frames(reader) == [7, 210, 211, 212]
    src = _FieldSrc()
    src.reader = reader
    rc = P.make_recompute(_behind_cf(), _scan(conf="high"), src)
    p = _auto(400, 430)
    rc(p)
    assert "dropped_frame_suspected" in p.flags and "dropped_frame_suspected" in p.speed.flags
    q = _auto(400, 430)
    P.make_recompute(_behind_cf(), _scan(conf="high"), _FieldSrc(), undecodable=[205])(q)
    assert "dropped_frame_suspected" in q.flags
