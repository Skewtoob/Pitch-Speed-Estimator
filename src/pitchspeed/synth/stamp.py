"""Debug barcode of the global field (sample) index, drawn per FIELD.

Drawn into each field image before interleaving, so field order and frame
alignment survive any codec: read the top rows of field parity p of decoded
frame n and compare with the GT frame map.  Bits are bright/dark blocks,
LSB first, along the top-left of the image: rows [2, 2 + BLOCK_H) of a
field (vscale=1) or [4, 4 + 2*BLOCK_H) of a progressive frame (vscale=2).
Reading takes the MEDIAN of each block's interior, so one corrupted line
(a simulated tape dropout) cannot flip a bit.
"""
from __future__ import annotations

import numpy as np

N_BITS = 14
BLOCK_H = 6            # rows of a field (doubled for a full frame)


def bit_width(width: int, nbits: int = N_BITS) -> int:
    return int(min(24, max(6, (width - 16) // nbits)))


def _rows(vscale: int) -> tuple[int, int]:
    y0 = 2 * vscale
    return y0, y0 + BLOCK_H * vscale


def stamp(img: np.ndarray, k: int, vscale: int = 1, nbits: int = N_BITS) -> None:
    """Draw the barcode of index k into img (a field: vscale=1, a full frame: vscale=2)."""
    bw = bit_width(img.shape[1], nbits)
    y0, y1 = _rows(vscale)
    for b in range(nbits):
        img[y0:y1, 8 + b * bw:8 + (b + 1) * bw - 4] = 230 if (k >> b) & 1 else 20


def read_stamp(img: np.ndarray, vscale: int = 1, nbits: int = N_BITS) -> int:
    """Read the barcode from a gray field image (vscale=1) or progressive frame (vscale=2)."""
    bw = bit_width(img.shape[1], nbits)
    y0, y1 = _rows(vscale)
    y0, y1 = y0 + vscale, y1 - vscale
    m = max(1, bw // 6)
    bits = [float(np.median(img[y0:y1, 8 + b * bw + m:8 + (b + 1) * bw - 4 - m])) > 125 for b in range(nbits)]
    return sum(int(v) << b for b, v in enumerate(bits))
