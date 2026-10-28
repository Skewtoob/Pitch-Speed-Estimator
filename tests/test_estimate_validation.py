"""Monte Carlo validation of the speed estimators (math.md error budgets).

Simulated 3-D pitches (drag, gravity, Magnus; random extension, release
side/height, Cd, spin, target) are seen through pinhole cameras at 59.94 Hz
with a random field phase, 1 px centroid noise and 2 px click noise; event
times follow math.md §2's noise models.  Extension, catch point and Cd are
drawn from the same priors the estimator assumes, so the 90 % intervals
should cover ~90 % of the pitches.

Fast checks run by default; the full grid is marked slow
(`pytest -m slow tests/test_estimate_validation.py -s` prints the tables).
"""
import pytest

import estimate_helpers as H

# math.md §4 predicted 1-sigma (%) for the behind view: (90@60.5, 75, 60, 45@46)
BEHIND_BUDGET = {"A": (2.8, 2.7, 2.8, 3.3), "B": (3.4, 3.1, 3.0, 3.5), "C": (2.3, 2.1, 2.3, 2.5),
                 "D": (3.5, 3.2, 3.1, 3.5)}
BEHIND_KINDS = ("90FB", "75FB", "60FB", "45FB46")
# math.md §3 E2 RMS (%) for (90, 60, 45@46); the simulation here also has
# non-square pixels (10/11) and slightly wider Cd / extension spreads.
SIDE_RMS = {"perp90": (1.2, 1.4, 1.6), "perp150": (1.0, 1.1, 1.2), "elev15ft": (1.8, 2.0, 2.3),
            "far-oblique": (2.6, 2.8, 3.3), "mound-side": (2.1, 2.5, 2.5)}
SIDE_KINDS = ("90FB", "60FB", "45FB46")


def _check(st: H.Stats, lo=0.85, hi=0.97, max_bias=0.8, allow_none=False):
    assert allow_none or st.n_none == 0
    assert abs(st.bias) < max_bias, st.line("bias")
    assert lo <= st.coverage <= hi, st.line("coverage")


# ------------------------------------------------------------------ fast
@pytest.mark.parametrize("kind", ["90FB", "45FB46"])
def test_behind_view_first_field_release(kind):
    st = H.run_behind("A", kind, 300, seed=21)
    print(st.line(f"behind A {kind}"))
    # bias: sample-mean fluctuation of extension / catch point (~0.15 %) plus the
    # simulated +1 ms catch-time bias of the vision module (~ -0.2 %)
    _check(st, 0.86, 0.96, 0.8)
    i = BEHIND_KINDS.index(kind)
    assert st.rms < BEHIND_BUDGET["A"][i] + 0.3
    assert abs(st.mean_sigma - BEHIND_BUDGET["A"][i]) < 0.3


@pytest.mark.parametrize("kind", ["90FB", "45FB46"])
def test_side_view_perpendicular(kind):
    st = H.run_side("perp90", kind, 60, seed=22)
    print(st.line(f"side perp90 {kind}"))
    assert st.methods == {"side_fit": 60}
    _check(st, 0.80, 1.0, 1.0)          # n = 60: binomial spread
    assert st.rms < 2.3


def test_side_view_very_oblique_is_not_measurable_without_manual_events():
    st = H.run_side("deep-oblique", "75FB", 20, seed=23)
    assert st.n_none == 20 and st.flags.get("oblique_view") == 20
    man = H.run_side("deep-oblique", "75FB", 20, seed=23, manual=True)
    assert man.n_none == 0 and man.methods == {"manual_tof": 20}


# ------------------------------------------------------------------ slow
@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["A", "B", "C", "D", "manual"])
@pytest.mark.parametrize("kind", BEHIND_KINDS)
def test_behind_view_grid(scenario, kind):
    st = H.run_behind(scenario, kind, 400, seed=31)
    print(st.line(f"behind {scenario} {kind}"))
    _check(st, 0.86, 0.96, 0.8)
    if scenario in BEHIND_BUDGET:
        assert st.rms < BEHIND_BUDGET[scenario][BEHIND_KINDS.index(kind)] + 0.4


@pytest.mark.slow
@pytest.mark.parametrize("scene", ["perp90", "perp150", "elev15ft", "mound-side", "far-oblique", "deep-oblique"])
@pytest.mark.parametrize("kind", SIDE_KINDS)
def test_side_view_grid(scene, kind):
    st = H.run_side(scene, kind, 150, seed=32)
    print(st.line(f"side {scene} {kind}"))
    # n = 150: +-2.5 binomial sd around 0.90-0.95.  mound-side (42 deg) may
    # over-cover: the focal length (hence phi) is only loosely determined, E2 is
    # partly rejected in favour of the conservative time of flight, and a few
    # pitches are gated as too oblique (no number).
    if scene == "deep-oblique":          # 66 deg: neither E2 nor an image-space release is trusted
        assert st.n_none == st.n and st.flags.get("oblique_view") == st.n
        return
    mound = scene == "mound-side"      # some pitches are gated as too oblique (phi known to a few deg)
    _check(st, 0.84, 1.0 if mound else 0.985, 0.8, allow_none=mound)
    assert st.n_none <= 0.25 * st.n
    if scene in SIDE_RMS:
        assert st.rms < SIDE_RMS[scene][SIDE_KINDS.index(kind)] + 0.7


@pytest.mark.slow
@pytest.mark.parametrize("variant,kw", [
    ("manual events", dict(manual=True)),
    ("last 50% visible", dict(visible_frac=0.5)),
    ("handedness known", dict(hand_known=True)),
    ("wrong field order", dict(swap_order=True)),
    ("10% outliers", dict(outlier_frac=0.1)),
    ("2.5 px noise", dict(sigma_px=2.5)),
    ("3 deg roll", dict(roll_deg=3.0)),
])
@pytest.mark.parametrize("scene", ["perp90", "elev15ft", "far-oblique"])
def test_side_view_variants(variant, kw, scene):
    st = H.run_side(scene, "75FB", 100, seed=33, **kw)
    print(st.line(f"side {scene} {variant}"))
    _check(st, 0.82, 0.99, 1.0)        # n = 100: +-2.5 binomial sd
    if variant == "wrong field order":
        assert st.flags.get("field_order_ambiguous") == 100
    if variant == "last 50% visible":
        assert st.flags.get("partial_track") == 100
