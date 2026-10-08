"""Label-free map from a raw MBON readout to a 0–100% recognition confidence.

The absolute readout (approach minus avoid, or minus novelty) carries an
offset from the connectome. It is not a probability. Confidence asks only
whether this frame sits above the same brain's own recent quiet frames.

Quiet frames are the low-energy part of the recent sensory stream. Energy is
the mean absolute value of the 72-d camera/lidar vector. Operator keys do not
enter this file: no D, N, T, or X argument exists on ``update``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

WINDOW = 200
WARMUP = 20
QUIET_Q = 0.35
BUSY_Q = 0.65
Z_ON = 0.80
Z_OFF = 0.65


@dataclass
class Confidence:
    z: float = 0.0
    percent: float = 0.0
    recognized: bool = False
    ready: bool = False
    floor: float = 0.0
    scale: float = 1.0


class ConfidenceCalibrator:
    def __init__(self) -> None:
        self.readouts: deque[float] = deque(maxlen=WINDOW)
        self.energies: deque[float] = deque(maxlen=WINDOW)
        self.recognized = False
        self.last = Confidence()

    def reset(self) -> None:
        self.readouts.clear()
        self.energies.clear()
        self.recognized = False
        self.last = Confidence()

    def _baseline(self) -> tuple[float, float, float, bool] | None:
        if len(self.readouts) < WARMUP:
            return None
        values = np.asarray(self.readouts, dtype=np.float64)
        energies = np.asarray(self.energies, dtype=np.float64)
        lo_cut = float(np.quantile(energies, QUIET_Q))
        hi_cut = float(np.quantile(energies, BUSY_Q))
        quiet = values[energies <= lo_cut]
        busy = values[energies >= hi_cut]
        if quiet.size < 3:
            quiet = values
        if busy.size < 3:
            busy = values
        mu_lo = float(np.median(quiet))
        mu_hi = float(np.median(busy))
        mad_lo = float(np.median(np.abs(quiet - mu_lo)))
        gap = mu_hi - mu_lo
        # Busy frames must clear the quiet jitter by two MADs. A smaller gap is
        # the untrained brain's own scatter, so the indicator stays «НЕ УЗНАЮ».
        learned = gap > max(2.0 * mad_lo, 1.0)
        return mu_lo, gap, mad_lo, learned

    def _map(self, readout: float, *, latch: bool) -> Confidence:
        base = self._baseline()
        if base is None:
            return Confidence(percent=0.0, recognized=False, ready=False)
        mu_lo, gap, mad_lo, learned = base
        if learned:
            z = (float(readout) - mu_lo) / gap
            percent = float(np.clip(z, 0.0, 1.0) * 100.0)
            if latch and self.recognized:
                recognized = z >= Z_OFF
            else:
                recognized = z >= Z_ON
        else:
            z = gap / max(mad_lo, 1.0)
            percent = 0.0
            recognized = False
        return Confidence(
            z=float(z),
            percent=percent,
            recognized=recognized,
            ready=True,
            floor=mu_lo,
            scale=float(gap if learned else max(mad_lo, 1.0)),
        )

    def update(self, readout: float, energy: float) -> Confidence:
        self.readouts.append(float(readout))
        self.energies.append(float(energy))
        mapped = self._map(readout, latch=True)
        self.recognized = mapped.recognized
        self.last = mapped
        return self.last

    def score(self, readout: float) -> Confidence:
        """Map a readout through the current full-frame floor and gap.

        Used for bearing crops. Does not append to the window and does not
        move the latched full-frame «УЗНАЮ» flag.
        """
        return self._map(readout, latch=False)

    def export_arrays(self) -> dict[str, np.ndarray]:
        return {
            "cal_readout": np.asarray(self.readouts, dtype=np.float32),
            "cal_energy": np.asarray(self.energies, dtype=np.float32),
            "cal_recognized": np.int8(self.recognized),
        }

    def load_arrays(self, z) -> None:
        files = set(getattr(z, "files", []))
        if "cal_readout" not in files or "cal_energy" not in files:
            return
        reads = np.asarray(z["cal_readout"], dtype=np.float32).ravel()
        en = np.asarray(z["cal_energy"], dtype=np.float32).ravel()
        n = min(len(reads), len(en), WINDOW)
        self.readouts.clear()
        self.energies.clear()
        self.readouts.extend(float(v) for v in reads[-n:])
        self.energies.extend(float(v) for v in en[-n:])
        if "cal_recognized" in files:
            self.recognized = bool(int(np.asarray(z["cal_recognized"]).ravel()[0]))
        if len(self.readouts) >= WARMUP:
            last_r = self.readouts.pop()
            last_e = self.energies.pop()
            self.update(last_r, last_e)


class TrainProgress:
    """Session counters plus totals stored with the KC→MBON file.

    Label totals are display metrics. Nothing here is read by plasticity
    or by the calibrator.
    """

    def __init__(self) -> None:
        self.base_pam = 0
        self.base_ppl1 = 0
        self.base_novelty = 0
        self.base_time = 0.0
        self.session_time = 0.0
        self.base_dog_n = 0
        self.base_dog_sum = 0.0
        self.base_none_n = 0
        self.base_none_sum = 0.0
        self.base_correct = 0
        self.base_scored = 0
        self.ses_dog_n = 0
        self.ses_dog_sum = 0.0
        self.ses_none_n = 0
        self.ses_none_sum = 0.0
        self.ses_correct = 0
        self.ses_scored = 0

    def reset(self) -> None:
        self.__init__()

    def tick(self, dt: float, learning: bool) -> None:
        if learning and dt > 0:
            self.session_time += float(dt)

    def note_label(self, label: str | None, readout: float, recognized: bool, ready: bool) -> None:
        if not ready or label not in ("dog", "none"):
            return
        hit = (label == "dog" and recognized) or (label == "none" and not recognized)
        self.ses_scored += 1
        self.ses_correct += int(hit)
        if label == "dog":
            self.ses_dog_n += 1
            self.ses_dog_sum += float(readout)
        else:
            self.ses_none_n += 1
            self.ses_none_sum += float(readout)

    @staticmethod
    def _sep(dog_n: int, dog_sum: float, none_n: int, none_sum: float) -> float | None:
        if dog_n < 1 or none_n < 1:
            return None
        return dog_sum / dog_n - none_sum / none_n

    @staticmethod
    def _acc(correct: int, scored: int) -> float | None:
        if scored < 1:
            return None
        return correct / scored

    def session_sep(self) -> float | None:
        return self._sep(self.ses_dog_n, self.ses_dog_sum, self.ses_none_n, self.ses_none_sum)

    def total_sep(self) -> float | None:
        return self._sep(
            self.base_dog_n + self.ses_dog_n,
            self.base_dog_sum + self.ses_dog_sum,
            self.base_none_n + self.ses_none_n,
            self.base_none_sum + self.ses_none_sum,
        )

    def session_acc(self) -> float | None:
        return self._acc(self.ses_correct, self.ses_scored)

    def total_acc(self) -> float | None:
        return self._acc(self.base_correct + self.ses_correct, self.base_scored + self.ses_scored)

    def export_arrays(self, n_pam: int, n_ppl1: int, n_novelty: int) -> dict[str, np.ndarray]:
        return {
            "stat_tot": np.asarray(
                [
                    self.base_pam + int(n_pam),
                    self.base_ppl1 + int(n_ppl1),
                    self.base_novelty + int(n_novelty),
                    self.base_time + self.session_time,
                    self.base_dog_n + self.ses_dog_n,
                    self.base_dog_sum + self.ses_dog_sum,
                    self.base_none_n + self.ses_none_n,
                    self.base_none_sum + self.ses_none_sum,
                    self.base_correct + self.ses_correct,
                    self.base_scored + self.ses_scored,
                ],
                dtype=np.float64,
            )
        }

    def load_arrays(self, z) -> None:
        files = set(getattr(z, "files", []))
        if "stat_tot" not in files:
            return
        tot = np.asarray(z["stat_tot"], dtype=np.float64).ravel()
        if tot.size < 10:
            return
        self.base_pam = int(tot[0])
        self.base_ppl1 = int(tot[1])
        self.base_novelty = int(tot[2])
        self.base_time = float(tot[3])
        self.base_dog_n = int(tot[4])
        self.base_dog_sum = float(tot[5])
        self.base_none_n = int(tot[6])
        self.base_none_sum = float(tot[7])
        self.base_correct = int(tot[8])
        self.base_scored = int(tot[9])
        self.session_time = 0.0
        self.ses_dog_n = self.ses_none_n = 0
        self.ses_dog_sum = self.ses_none_sum = 0.0
        self.ses_correct = self.ses_scored = 0


def fmt_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} с"
    minutes, sec = divmod(s, 60)
    if minutes < 60:
        return f"{minutes} мин {sec:02d} с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} мин"
