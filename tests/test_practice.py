"""Practice video: generated with known speeds, marked by hand, scored."""
import json
import math

import pytest

pytest.importorskip("av")

from pitchspeed import cli, pipeline as P, practice as PR      # noqa: E402
from pitchspeed.review import ui                              # noqa: E402
from pitchspeed.review.state import ReviewState               # noqa: E402
from test_cli_rerun import _inputs                            # noqa: E402


def test_practice_config_is_a_behind_view_30_fps_tape_with_known_varied_speeds():
    c = PR.practice_config(seed=3, count=8)
    sp = [s["speed_mph"] for s in c["pitches"]["schedule"]]
    assert len(sp) == 8 and min(sp) >= 60 and max(sp) <= 88 and len(set(sp)) > 4
    assert c["camera"]["view"] == "behind" and c["video"]["postprocess"] == "blend"
    assert PR.practice_config(seed=3) == PR.practice_config(seed=3)


def test_practice_marks_get_scored_against_the_truth(tmp_path, monkeypatch, capsys):
    """Make a short practice video, 'mark' it like a careful user (one pitch a frame late),
    and check the score sees exactly that."""
    monkeypatch.setattr(PR, "DEFAULT_COUNT", 2)
    out = tmp_path / "practice"
    video = PR.make_practice_video(str(out), seed=5, count=2)
    gt = json.load(open(PR.answers_path(video)))
    dt = gt["video"]["out_frame_dt"]

    def window(result, source, calib_file, recompute, save, mode="review", **kw):
        st = ReviewState(result, source, recompute, save, mode=mode)
        for i, p in enumerate(gt["pitches"]):
            st.goto(math.floor(p["t_release"] / dt) + 1 + (1 if i == 1 else 0))   # 2nd release a frame late
            st.on_key("n")
            st.goto(math.ceil(p["t_catch"] / dt))
            st.on_key("c")
        st.finish()
        return result
    monkeypatch.setattr(ui, "run_review_ui", window)
    _inputs(monkeypatch, [])
    assert cli.main(["practice", "--out", str(out)]) == 0
    txt = capsys.readouterr().out
    assert "Practice score" in txt and "Pitches scored: 2 of 2" in txt
    rows, missed, extra = PR.score(__import__("pitchspeed.render.report", fromlist=["load_json"]).load_json(
        P.out_paths(P.default_out_dir(video))["results_json"]), gt)
    assert missed == 0 and extra == 0
    assert abs(rows[0].release_frames) < 0.6 and 0.4 < rows[1].release_frames < 1.6
    assert rows[1].error > rows[0].error                       # a late release reads fast
    assert (out / (PR.PRACTICE_NAME.replace(".mp4", "") + "_pitchspeed") / PR.SCORE_NAME).exists()
