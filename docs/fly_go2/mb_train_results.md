# MB recognition training, no zones

One learner, two scenery peers, two moving distractors and one static distractor. No zone polygons and no zone reward. 40 s, `dt = 0.1`, `eta = 0.2`. Raw features go through a fixed random projection into PN, then the FlyWire KC→MBON weights. Probes are canonical empty / dog / distractor views and do not teach.

`sep_dog_minus_dist` is the change in (dog probe − distractor probe) from the start of the run to the end. Positive means the dog probe pulled away from the distractor probe in the direction that mode is training.

## Teacher (default): PAM when a peer is alone in view

Headless stands in for the `T` key. It pulses appetitive PAM only on frames where the learner's own camera sees a dog and not a distractor. `D` / `N` are not involved. Readout is approach MBON minus avoid MBON.

| seed | PAM pulses | drift | dog probe start → end | distractor probe start → end | sep |
|---|---:|---:|---|---|---:|
| 1 | 176 | 176 | −368 → +126 | −350 → +13 | +132 |
| 2 | 126 | 172 | +136 → +637 | −4 → +313 | +185 |
| 3 | 188 | 206 | +26 → +895 | −29 → +701 | +139 |

On the live curves (scene geometry bins the plot only), the late window is higher with a dog in view than without: seed 1, 554 vs 426; seed 2, 835 vs 370; seed 3, 944 vs 488.

The distractor probe rises too. Shared KCs carry some of the treat. The dog probe rises more on every seed.

Optional `X` / `--punish` adds PPL1 on distractor-only frames. Seed 1 with punish: 176 PAM and 78 PPL1, sep +200 (dog −368 → −143, distractor −350 → −324). That is the punish key, not the default.

## Familiarity (secondary): repetition depresses novelty MBONs

Readout is minus the mean of MBONs on the aversive mask, so a higher number is a quieter novelty response. Dogs are in view about three times as often as other views. That does not become a dog-specific memory.

| seed | novelty events | drift | dog probe start → end | distractor probe start → end | sep |
|---|---:|---:|---|---|---:|
| 1 | 353 | 122 | −39.0 → −30.7 | −38.4 → −29.5 | −0.65 |
| 2 | 364 | 105 | −27.1 → −22.3 | −48.8 → −43.2 | −0.83 |
| 3 | 342 | 115 | −45.2 → −37.1 | −42.7 → −36.1 | +1.59 |

Both probes get slightly quieter. The gap between them stays about where it started. Familiarity here is repetition of KC patterns, not a conspecific label.

## Fresh lidar window vs an accumulating cloud

The tables above used the simulator's instantaneous view. The robot's `/lidar.jpg` instead keeps every return until `/lidar/reset`. `--lidar-refresh 1.5` (the new default) commits one 1.5 s world-point window to the lidar channels and then clears it. The empty moment after a clear is not fed to the mushroom body. `--lidar-refresh 0` keeps the cloud for the whole run.

Same 25 s teacher walk, frames after 8 s, seeds 1–3. Readout gap is the mean approach−avoid with a dog in view minus the mean with no dog. PAM counts match across the pair (115, 111, 114), so the walks lined up. Mean lidar-near on the no-dog frames is the leftover trace: walls are in both, old tracks only in the accumulating cloud.

| seed | fresh 1.5 s gap | accumulate gap | fresh no-dog near | accumulate no-dog near |
|---|---:|---:|---:|---:|
| 1 | +150 | +108 | 0.79 | 0.88 |
| 2 | +464 | +336 | 0.72 | 0.86 |
| 3 | +452 | +289 | 0.76 | 0.85 |

The short window separated dog-in-view from no-dog more on every seed. The accumulating cloud left a higher near-field on frames with no dog in the camera.
