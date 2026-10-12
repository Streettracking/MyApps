"""Kenyon-cell sparsity, KC→MBON homeostasis, and the turn.

One place for those numbers. The dopamine step size is
``MushroomBodyRuntime.eta`` (the trainer passes 0.2) and is not copied here.
"""

from __future__ import annotations

import math

# APL / k-WTA. One frame keeps this fraction inside each visual group.
KC_ACTIVE_FRAC = 0.07
KC_FRAC_MIN = 0.05
KC_FRAC_MAX = 0.10

# Visual groups per hemisphere. The FlyWire cells of that type are repeated
# until the group reaches this size, so the file's own mix is kept.
KC_GD_PER_SIDE = 1500
KC_ABP_PER_SIDE = 650

# Depressed weights close this fraction of the gap to their birth value each frame.
RECOVERY_RATE = 0.01
# Extra gain when a Kenyon cell fires and no dopamine arrives, per frame.
IDLE_GROWTH = 0.002
# Bounds as a fraction of the birth weight. The floor stays above zero.
W_FLOOR_FRAC = 0.20
W_CEIL_FRAC = 1.50

# Ratio law kept for the gain probe: k (drive_L - drive_R) / (|drive_L| + |drive_R| + eps).
# Each drive is approach minus avoidance on that side. Left is positive.
YAW_K = 1.0
YAW_EPS = 1e-3
YAW_DEAD = 0.08

# Live yaw. Both hemispheres are born near +520 for a pattern in one hemifield,
# with the same sign. Dividing by that scale, and saturating presence, cancels
# a blank frame and a pair of occupied hemifields whose birth magnitudes differ
# by a few tens of counts. The learned term is this view's drive minus its
# birth drive, in those units. tanh keeps the command inside ±1.
TURN_SCALE = 520.0
TURN_PRESENCE = 0.35
TURN_MEMORY = 4.0
TURN_TANH = 1.05
LIVE_DEAD = 0.05


def kc_winner_count(n_kc: int) -> int:
    """How many Kenyon cells APL keeps. The count stays inside the 5–10% band."""
    n = int(n_kc)
    if n <= 1:
        return n
    k = int(round(KC_ACTIVE_FRAC * n))
    lo = int(math.ceil(KC_FRAC_MIN * n - 1e-12))
    hi = int(math.floor(KC_FRAC_MAX * n + 1e-12))
    if lo < 1:
        lo = 1
    if hi < lo:
        hi = lo
    if k < lo:
        k = lo
    if k > hi:
        k = hi
    if k > n:
        k = n
    return k
