"""`python -m pitchspeed.synth ...` == `pitchspeed synth ...` (guarded for 'spawn' workers)."""
from pitchspeed.synth.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
