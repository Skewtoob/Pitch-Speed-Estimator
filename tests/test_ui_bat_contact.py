"""Review: B marks the end of the flight at the bat (a hit ball has no catch)."""
from pitchspeed.estimate.events import manual_event
from pitchspeed.review.state import effective_end, end_text

from test_ui_review_fixes import _behind_estimator, keys, make
from ui_helpers import FakeSource


def test_b_marks_bat_contact_and_uses_the_contact_distance():
    src = FakeSource(n_frames=400)
    rc = _behind_estimator(src)
    st, _, res, _ = make(src, windows=((100, 130),), missed=(), recompute=rc)
    rc(res.pitches[0])
    glove_mph = res.pitches[0].speed.v_release_mph
    keys(st, "end", "b")
    p = st.current_pitch()
    assert p.catch.method == "manual" and p.catch.point == "contact" and p.catch.s == 130
    assert p.speed.assumptions["end_point"] == "contact"
    assert effective_end(p) == "contact" and "BAT CONTACT (B marked)" in end_text(p)
    # the flight to the bat is shorter than to the glove: same time -> lower speed
    assert p.speed.v_release_mph < glove_mph
    # C at the same field switches back to the glove; undo restores the bat contact
    keys(st, "c")
    p = st.current_pitch()
    assert p.catch.point is None and p.speed.assumptions["end_point"] == "catch"
    keys(st, "u")
    p = st.current_pitch()
    assert p.catch.point == "contact" and p.speed.assumptions["end_point"] == "contact"


def test_new_manual_pitch_ending_at_the_bat():
    src = FakeSource(n_frames=400)
    rc = _behind_estimator(src)
    st, _, res, _ = make(src, windows=(), missed=(), recompute=rc)
    st.goto(200)
    keys(st, "n")
    st.goto(228)
    keys(st, "b")
    p = [q for q in res.pitches if q.status == "manual"][0]
    assert p.catch.point == "contact" and p.catch.s == 228
    assert p.speed.assumptions["end_point"] == "contact"


def test_manual_event_point_validation():
    ev = manual_event(1.0, 1 / 60, "catch", s=60, point="contact")
    assert ev.point == "contact" and ev.method == "manual"
    import pytest
    with pytest.raises(ValueError):
        manual_event(1.0, 1 / 60, "release", point="contact")


def test_point_survives_json_round_trip(tmp_path):
    from pitchspeed.types import EventTime, pitch_from_dict, to_jsonable, Pitch, SpeedEstimate
    p = Pitch(number=1, status="manual", release=EventTime(t=1.0, sigma=0.005, method="manual", s=60),
              catch=EventTime(t=1.5, sigma=0.005, method="manual", s=90, point="contact"),
              speed=SpeedEstimate(v_release_mph=None))
    q = pitch_from_dict(to_jsonable(p))
    assert q.catch.point == "contact" and q.release.point is None
