"""Review navigation after handling a possible-missed window (user report:
R then C on the last window, then Y / Enter / PgDn 'kept showing the same window')."""
from test_ui_review_fixes import keys, make
from ui_helpers import FakeSource


def test_adding_a_pitch_on_the_last_window_then_moving_on():
    src = FakeSource(n_frames=400)
    st, _, res, _ = make(src, windows=((100, 125),), missed=((420, 440),))
    keys(st, "pgdn")
    assert st.cur.kind == "missed" and "press N" in st.message.text
    st.goto(425)
    keys(st, "r")
    st.goto(452)
    keys(st, "c")
    assert st.cur.kind == "pitch" and "added" in st.message.text
    for k in ("y", "enter", "pgdn"):
        keys(st, k)
        # never back on the handled window; a clear 'done' message instead
        assert st.cur.kind == "pitch" and "everything is checked" in st.message.text and "Q" in st.message.text
    assert res.possible_missed[0]["review"] == "pitch_added"
    assert [p.status for p in res.pitches] == ["auto", "manual"]


def test_window_after_new_pitch_is_skipped():
    """The window can start after the release the user marked: PgDn must not land on it."""
    src = FakeSource(n_frames=600)
    st, _, res, _ = make(src, windows=((100, 125), (520, 546)), missed=((420, 440),))
    keys(st, "pgdn")
    assert st.cur.kind == "missed"
    st.goto(410)                     # release marked before the window starts
    keys(st, "n")
    st.goto(436)
    keys(st, "c")
    keys(st, "pgdn")
    assert st.cur.kind == "pitch" and st.current_pitch().release.s >= 500
    assert "pitch" in st.message.text
