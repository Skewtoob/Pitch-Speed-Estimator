"""Tests for pitchspeed.render.graphics: radar card texts and states, sprites, the running
strip and the path drawing."""
from __future__ import annotations

import os

import numpy as np
import pytest

import render_helpers as H
from pitchspeed.render import graphics as G

pytest.importorskip("PIL")


def test_bundled_fonts_load_and_carry_their_licence():
    assert G.available() == (True, "")
    for f in list(G.FONT_FILES.values()) + ["OFL.txt"]:
        assert os.path.isfile(os.path.join(G.FONT_DIR, f))
    with open(os.path.join(G.FONT_DIR, "OFL.txt"), encoding="utf-8") as fh:
        text = fh.read()
    assert "SIL OPEN FONT LICENSE Version 1.1" in text and "The Inter Project Authors" in text


def test_card_texts_follow_the_honesty_rules():
    good, fair = H.speed(80.6, 1.8, "good"), H.speed(78.4, 3.1, "fair")
    poor, none = H.speed(66.2, 4.6, "poor"), H.speed(None, None, "none", reason="catch not seen")
    assert (G.card_digits(good), G.card_state(good), G.card_line(good)) == ("81", "good", "±2  90%: 78–84")
    assert G.card_state(fair) == "good" and G.card_line(fair).startswith("±3  90%: 73–83")
    assert (G.card_digits(poor), G.card_state(poor)) == ("66", "poor")
    assert G.card_line(poor) == "±4.5  90%: 59–74  rough"
    assert (G.card_digits(none), G.card_state(none), G.card_line(none)) == ("--", "none", "not measurable")
    assert G.card_digits(None) == "--" and G.card_line(None) == "not measurable"
    assert G.card_digits(H.speed(8.6, 0.5, "good")) == " 9"            # blank leading digit, like an LED
    assert G.card_digits(H.speed(101.5, 2.0, "good")) == "102"
    assert G.card_size(H.speed(101.5, 2.0, "good"))[0] == G.CARD_W + G.DIGIT_STEP
    odd = H.speed(70.0, 2.0, "good")
    odd.sigma_mph, odd.ci90_mph = None, None
    assert G.card_line(odd) == "no error estimate"


def _red(sp: G.Sprite) -> int:
    c = sp.premul
    return int(((c[..., 2] > 200) & (c[..., 1] < 100) & (c[..., 0] < 100)).sum())


def _amber(sp: G.Sprite) -> int:
    c = sp.premul
    return int(((c[..., 2] > 220) & (c[..., 1] > 130) & (c[..., 1] < 210) & (c[..., 0] < 90)).sum())


def test_radar_card_states_and_sizes():
    good = G.radar_card(3, H.speed(88.0, 1.5, "good"), 1.0)
    none = G.radar_card(3, H.speed(None, None, "none"), 1.0)
    poor = G.radar_card(3, H.speed(88.0, 6.0, "poor"), 1.0)
    assert (good.w, good.h) == (G.CARD_W, G.CARD_H)
    assert _red(good) > 3 * _red(none) > 0                           # '88' lights 14 segments, '--' two
    assert _amber(poor) > 200 and _amber(good) < 20                  # amber '~', 'EST', 'rough'
    half = G.radar_card(3, H.speed(88.0, 1.5, "good"), 0.5)
    assert (half.w, half.h) == (G.CARD_W // 2, G.CARD_H // 2)
    assert np.all(good.premul <= 255.0 * good.alpha + 1e-3) and good.alpha.max() <= 1.0


def test_blend_clips_and_respects_opacity():
    sp = G.Sprite(np.full((10, 10, 3), 255.0, np.float32), np.ones((10, 10, 1), np.float32))
    img = np.zeros((20, 20, 3), np.uint8)
    G.blend(img, sp, -5, -5)
    assert img[:5, :5].min() == 255 and img[5:, :].max() == 0 and img[:, 5:].max() == 0
    img = np.zeros((20, 20, 3), np.uint8)
    G.blend(img, sp, 15, 15, opacity=0.5)
    assert abs(int(img[19, 19, 0]) - 128) <= 1 and img[:15].max() == 0
    G.blend(img, sp, 100, 100)                                       # fully outside: nothing, no error
    G.blend(img, sp, 0, 0, opacity=0.0)
    assert img[:15].max() == 0


def test_running_strip_keeps_the_latest_items_that_fit():
    few = G.running_strip([("P1", "76"), ("P2", "~65?"), ("P3", "n/a")], 1.0, 430)
    many = G.running_strip([(f"P{i}", "77") for i in range(1, 60)], 1.0, 430)
    assert few.w < many.w <= 431 and few.h == many.h == int(G.LIST_H)
    assert G.running_strip([], 1.0, 430) is None
    assert [G.value_kind(v) for v in ("76", "~65?", "n/a")] == ["good", "poor", "none"]


def test_tag_label():
    sp = G.tag_label("P12", 1.0)
    assert 20 < sp.w < 60 and 10 < sp.h < 40 and sp.alpha.max() > 0.9


def test_thin_dots_never_closer_than_the_gap():
    pts = np.column_stack([np.arange(0, 50, 1.0), np.zeros(50)])
    out = G.thin_dots(pts, 5.0)
    assert np.all(np.diff(out[:, 0]) >= 5.0) and out[0, 0] == 0
    assert len(G.thin_dots(pts[:1], 5.0)) == 1


def _lit(img) -> np.ndarray:
    return img.astype(int).sum(axis=2) > 30


def test_draw_path_solid_where_seen_dashed_without_glow_in_gaps():
    img = np.zeros((100, 400, 3), np.uint8)
    line = np.column_stack([np.linspace(20, 380, 361), np.full(361, 50.0)])
    seen = line[:, 0] < 200
    G.draw_path(img, line, np.zeros((0, 2)), 1.0, seen=seen)
    lit = _lit(img)
    solid, gap = lit[:, 40:180].sum(), lit[:, 220:360].sum()
    assert solid > 3 * gap > 0                                       # glow + full width vs. thin dashes
    row = lit[50, 220:360]
    assert row.any() and not row.all()                               # dashes: on and off along the line
    assert not lit[:, 380 + 25:].any()                               # nothing beyond the last point (+glow)


def test_draw_path_dots_and_opacity():
    base = np.full((60, 60, 3), 40, np.uint8)
    img = base.copy()
    G.draw_path(img, np.zeros((0, 2)), np.array([[30.0, 30.0]]), 1.0)
    assert tuple(img[30, 30]) == G.DOT_BGR
    half = base.copy()
    G.draw_path(half, np.zeros((0, 2)), np.array([[30.0, 30.0]]), 1.0, opacity=0.5)
    assert 130 < int(half[30, 30, 0]) < 160
    none = base.copy()
    G.draw_path(none, np.zeros((0, 2)), np.array([[30.0, 30.0]]), 1.0, opacity=0.0)
    assert np.array_equal(none, base)
    off = base.copy()
    G.draw_path(off, np.array([[-500.0, -500.0], [-400.0, -400.0]]), np.zeros((0, 2)), 1.0)   # off screen
    assert np.array_equal(off, base)
