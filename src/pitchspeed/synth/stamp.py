"""Debug barcode of the global field (sample) index, drawn per FIELD.

Drawn into each field image before interleaving, so field order and frame
alignment survive any codec: read the top rows of field parity p of decoded
frame n and compare with the GT frame map.  Bits are bright/dark blocks,
LSB first, along the top-left of the image.
"""
from __future__ import annotations

import numpy as np

N_BITS = 14
BLOCK_H = 6            # rows (in field rows for interlaced)


def bit_width(width: int, nbits: int = N_BITS) -> int:
    return int(min(24, max(6, (width - 16) // nbits)))


def stamp(img: np.ndarray, k: int, vscale: int = 1, nbits: int = N_BITS) -> None:
    """Draw the barcode of index k into img (a field: vscale=1, a full frame: vscale=2)."""
    bw = bit_width(img.shape[1], nbits)
    y0, y1 = 2 * vscale // 2, (2 + BLOCK_H) * vscale // 2 + BLOCK_H * (vscale - 1) // 2
    for b in range(nbits):
        img[y0:y1, 8 + b * bw:8 + (b + 1) * bw - 4] = 230 if (k >> b) & 1 else 20


def read_stamp(img: np.ndarray, vscale: int = 1, nbits: int = N_BITS) -> int:
    """Read the barcode from a gray field image (vscale=1) or progressive frame (vscale=2)."""
    bw = bit_width(img.shape[1], nbits)
    y0 = 2 * vscale // 2 + 1
    y1 = (2 + BLOCK_H) * vscale // 2 + BLOCK_H * (vscale - 1) // 2 - 1
    m = max(1, bw // 6)
    bits = [img[y0:y1, 8 + b * bw + m:8 + (b + 1) * bw - 4 - m].mean() > 125 for b in range(nbits)]
    return sum(int(v) << b for b, v in enumerate(bits))
