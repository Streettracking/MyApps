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

## Seek after a scripted teacher

Same arena and the same PAM teacher as the fresh-lidar walk: 25 s, seeds 1–3, learning on, scripted steer. Then learning off and autonomy on for 20 s (`dt = 0.1`). The policy is the fly readout only. «УЗНАЮ» uses the confidence latch (quiet vs busy frames, z on at 0.80). These rows used the earlier controller: yaw flipped sign (`z = ±0.35`) and range was the nearest lidar return, including the body. The current controller turns only left, ignores returns inside 0.6 m, and smooths the sector outside the mushroom body. While the word is off the dog yaws in place. While it is on, it walks toward the same sector the lidar mark uses and holds near 1 m. A hold counts only when the phase is `hold`, the word is on, and the true distance to the nearest peer is inside 0.7–1.3 m. Touches more than 1 s apart are separate events.

| seed | PAM | seek frames with the word | approach frames | of them aimed at a dog | of them aimed at a distractor | holds at 1±0.3 m |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 115 | 0 / 200 | 0 | 0 | 0 | 0 |
| 2 | 111 | 6 / 200 | 6 | 0 | 0 | 0 |
| 3 | 114 | 6 / 200 | 6 | 4 | 2 | 0 |

The word stays off for almost the whole search, so the dog keeps turning and never settles at 1 m. Seed 2's six approach frames had neither a dog nor a distractor in the aimed sector (±1 bin). Seed 3 aimed at a dog on four of those frames and at a distractor on two, then lost the word. No hold landed inside 1±0.3 m of a distractor either. This is the same modest recall already seen when the camera is not held on a dog by the scripted walker. The trial does not add a detector to force approaches.

Takeover in this build is the `M` key and the on-screen «ПЕРЕХВАТ» button. Arrows during autonomy do not change the phase. On the mock sport client, takeover calls `StopMove`, later manual axes stay inside `x` ±0.4 and `z` ±1, and 300 ms without a manual packet stops again while the mode stays manual. E-STOP still blocks `A`.

`SportClient.Move` is held by the sport service for about one second, so a dropped sender coasts until `StopMove` or that timeout. The onboard program does not call `SwitchJoystick` or `Damp`. `ObstaclesAvoidClient.UseRemoteCommandFromApi` is a different client and is not used together with `SportClient`. The SDK text does not give `SportClient` a switch that makes the Unitree stick always win. The stick is not this app's takeover. `M` and E-STOP are. If the stick and `Move` are both live they can fight. Releasing the stick does not clear autonomy and does not put this process into manual.

## Two mushroom bodies, sectors vs bilateral

Same connectome, two KC→MBON copies. MB_L sees sectors 4–7. MB_R sees sectors 0–3 mirrored onto those slots, through the one projector (`seed + 17`). «УЗНАЮ» is the sum of the two readouts. Yaw is `z = (R_L − R_R) / (|R_L| + |R_R| + ε)` with a 0.08 deadband. Search with the word off stays the one-way left turn.

A 20 s teacher with the default 1.5 s lidar window, then 15 s of seek, never latched the word in either mode (seeds 1–3): both stayed in search, so direction samples, approaches, and sign changes were all zero. The comparison below therefore teaches for 8 s with instantaneous lidar and switches while the burst is still on. Both modes share the seed and the script up to that switch. `fly_acc` is the sign of that yaw during the teacher, against the nearest peer inside the field, before either controller moves the dog. Direction during seek uses the commanded `z` and the same in-field peer. A sign change counts when commanded `z` crosses a 0.05 deadband. A hold counts when an approach/hold episode while the word is on includes a hold with that peer at 0.7–1.3 m. Python 3.8.20, numpy 1.24.4.

| seed | PAM | fly_acc | mode | direction | z sign changes / s | approach frames | hold share | distractor share | word at switch |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| 1 | 29 | 48/72 = 0.67 | sectors | 0/2 = 0 | 2 / 0.17 | 9 | 1/1 | 9/9 = 1 | on |
| 1 | 29 | 48/72 = 0.67 | bilateral | 0/1 = 0 | 2 / 0.17 | 1 | 1/1 | 0/1 = 0 | on |
| 2 | 16 | 54/80 = 0.68 | sectors | 18/22 = 0.82 | 2 / 0.17 | 22 | 0/1 | 4/22 = 0.18 | on |
| 2 | 16 | 54/80 = 0.68 | bilateral | 8/8 = 1 | 2 / 0.17 | 18 | 0/1 | 9/18 = 0.50 | on |
| 3 | 72 | 49/80 = 0.61 | sectors | 1/10 = 0.10 | 0 / 0 | 10 | 1/1 | 0/10 = 0 | off |
| 3 | 72 | 49/80 = 0.61 | bilateral | 8/8 = 1 | 4 / 0.33 | 8 | 1/1 | 0/8 = 0 | off |

Seed 1's direction counts are one or two frames: most approach frames had no peer inside the field. Seed 2 aims more often with sectors (18/22) and every sampled bilateral frame (8/8); bilateral also walks at a distractor on half of its approach frames, sectors on about a fifth. Seed 3's bilateral samples all point at the dog (8/8) and sectors almost never do (1/10); bilateral changes sign twice as often. Hold share matches across modes on every seed (1, 0, 1). The fly's own sign during the teacher is about 61–68%. Neither mode is smoother on all three seeds.

Two `forward` calls on a dense 72-d frame take 3.8 ms. Eight mark windows add 17.4 ms, 21.2 ms for a frame that also places marks. The onboard tick is 0.1 s. Empty half skips the KC pass.
