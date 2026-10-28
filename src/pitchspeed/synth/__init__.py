"""Synthetic baseball pitch videos with exact ground truth (testing aid).

Public entry points (`generate_clip` is imported lazily so `import pitchspeed.synth` is cheap):

    from pitchspeed.synth import generate_clip
    from pitchspeed.synth.evaluate import evaluate, evaluate_dir, format_summary
    gt = generate_clip({"camera": {"view": "behind"}, "duration_s": 8}, "out/clip.mpg")

Modules: config (schema + defaults), physics (3-D flight), camera (pinhole +
shake), events (pitches, distractors), scene (background canvas, actors),
fence (backstop overlay), ball (motion-blur coverage), render (fields ->
frames), tape (noise / tape chain), stamp (debug barcode), encode (ffmpeg),
gt (ground truth + calibration files), generate (generate_clip), suite,
check, evaluate, cli.  Design: docs/design-notes/synth.md.
"""
from __future__ import annotations


def generate_clip(config, out_path, **kw):
    from pitchspeed.synth.generate import generate_clip as _g
    return _g(config, out_path, **kw)


__all__ = ["generate_clip"]
