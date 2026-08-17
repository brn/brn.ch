"""Turn a portrait photograph into the RGBA asset particles.js samples.

    python3 tools/photo2particles.py assets/whoami.jpg assets/whoami.png --mosaic
    python3 tools/photo2particles.py assets/skelton.png assets/skull.png

Channels, all of which the sampler reads:

    R  tone, contrast-stretched over the sitter only
    G  how much of a face this pixel is, which is where the mosaic applies
    B  tone averaged per mosaic cell
    A  the cut-out: the sitter, not the wall behind him

The mask is the part that matters. The wall is brighter than the sitter, so
sampling on tone alone lights the wall and loses the man. It is found by
flooding the bright region in from the border, so highlights *inside* the face
stay part of the subject instead of being punched out as holes.

MOSAIC_CELL must stay in step with the constant of the same name in particles.js.
"""
import sys
from collections import deque

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 340, 476
MOSAIC_CELL = 16
WALL = 0.90          # a pixel this bright, reachable from the border, is wall
# Only the head is wanted. The silhouette holds a steady width down to the chin
# and then flares where the shoulders start, so the mask is faded out across
# that band: the jaw stays, the neck dissolves, the shirt goes.
HEAD_FADE = (0.60, 0.74)


def blur(m, passes):
    for _ in range(passes):
        p = np.pad(m, 1, mode='edge')
        m = (p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1]
             + p[1:-1, :-2] + p[1:-1, 2:]) / 5
    return m


def subject_mask(lum):
    h, w = lum.shape
    bright = lum > WALL
    wall = np.zeros_like(bright)
    q = deque()

    def push(y, x):
        if bright[y, x] and not wall[y, x]:
            wall[y, x] = True
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                push(ny, nx)

    mask = (~wall).astype(np.float32)

    v = (np.arange(h, dtype=np.float32) + 0.5) / h
    t = np.clip((v - HEAD_FADE[0]) / (HEAD_FADE[1] - HEAD_FADE[0]), 0, 1)
    mask *= (1 - t * t * (3 - 2 * t))[:, None]

    return blur(mask, 3)


def face_weight(rgb, mask):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    span = rgb.max(2) - rgb.min(2)
    skin = ((r > 95) & (g > 40) & (b > 20) & (span > 15)
            & (np.abs(r - g) > 15) & (r > g) & (r > b)).astype(np.float32)
    return np.clip(blur(skin, 14) * 2.2, 0, 1) * mask


def cell_average(tone):
    out = tone.copy()
    h, w = tone.shape
    for y in range(0, h, MOSAIC_CELL):
        for x in range(0, w, MOSAIC_CELL):
            out[y:y + MOSAIC_CELL, x:x + MOSAIC_CELL] = \
                tone[y:y + MOSAIC_CELL, x:x + MOSAIC_CELL].mean()
    return out


def main(src, dst, mosaic):
    rgb = np.asarray(Image.open(src).convert('RGB')
                     .resize((WIDTH, HEIGHT), Image.LANCZOS)).astype(np.float32)
    lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
           + 0.0722 * rgb[:, :, 2]) / 255

    mask = subject_mask(lum)
    lo, hi = np.percentile(lum[mask > 0.5], [2, 99])
    tone = np.clip((lum - lo) / (hi - lo), 0, 1)
    face = face_weight(rgb, mask) if mosaic else np.zeros_like(tone)

    out = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    out[:, :, 0] = (tone * 255).astype(np.uint8)
    out[:, :, 1] = (face * 255).astype(np.uint8)
    out[:, :, 2] = (cell_average(tone) * 255).astype(np.uint8)
    out[:, :, 3] = (mask * 255).astype(np.uint8)
    Image.fromarray(out, 'RGBA').save(dst, optimize=True)

    print('%s -> %s  subject %.3f  face %.3f  tone %.3f-%.3f'
          % (src, dst, mask.mean(), face.mean(), lo, hi))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0], args[1], '--mosaic' in sys.argv)
