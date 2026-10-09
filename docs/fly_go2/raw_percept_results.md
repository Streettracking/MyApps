# Raw percept comparison

Headless runs, 3 agents, 60 s, seeds 0–4. Artifact: `raw_percept_compare.json`.

Reinforcement is only the agent's own zone reward (`A=-1`, `B=+1`) through DAN-gated KC→MBON plasticity. Probes are zone-odor off.

- **fixed** — labeled `peer` cue.
- **raw** — camera + lidar, no dog label; two moving distractors.
- **blind** — fixed mode with other dogs removed.
- **raw_blind** — raw mode with other dogs omitted from the sensors; distractors remain.

`learned_*_l2` is how far that probe's action-score vector moved from its own initial value. `diff_l2` is the distance between the dog shift and the distractor shift. Approach rates count MB actions that start with `Approach` while `r=0` and only that object is in view.

| condition | mean PI | plastic updates | weight drift | dog shift L2 | distractor shift L2 | diff L2 | approach dog | approach distractor | reward steps with dog | reward steps with distractor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.983 | 571 | 221 | 385 | 0 | 385 | 0.81 | — | 1.00 | 0 |
| raw | 0.981 | 479 | 179 | 196 | 177 | 68 | 0.82 | 0.80 | 0.67 | 0.37 |
| blind | 0.852 | 460 | 158 | 106 | 0 | 106 | — | — | 0 | 0 |
| raw_blind | 0.983 | 490 | 184 | 134 | 149 | 75 | — | 0.87 | 0 | 0.40 |

Blind PI is pulled down by seed 3 (`PI=+0.33`); the other four seeds sit near `+0.98`.

## What this does and does not show

Zone learning is intact in every condition: agents spend the episode in B.

Raw-feature agents do **not** develop a differential response to other dogs versus distractors.

- Outside zones, approach rate given a dog in view is 0.82 and given a distractor is 0.80.
- The dog probe and the distractor probe both move by a large, similar amount (`196` vs `177`). Their difference (`68`) is no larger than `raw_blind` (`75`), where no dog was ever seen.
- `raw_blind` still moves the dog-probe scores (`134`). The canonical dog vector is not orthogonal to zone and distractor patterns after the random PN projection, so zone plasticity leaks into the probe.

The large fixed-mode dog shift (`385` vs `106` when blind) is the labeled `peer` bit being paired with reward. Once the group sits in B, that bit is on for every rewarded step (`frac=1`). That is binding a pre-labeled cue to valence, not recognizing a dog from pixels.

Why the raw effect is absent:

1. The only teacher is "I am inside a zone". Dog pixels are reinforced only when a dog happens to be in the egocentric view at that moment, and distractors are in view on 37% of those steps too.
2. The sparse KC threshold keeps the strongest zone-driven cells. Object features ride along and share Kenyon cells with each other, so the weight update is not object-specific.
3. With `r=0`, the policy is already "go to B". Approach counts then do not mean "approach the thing I see".

## Recognition layer

Same episode length and seeds, with one change to the arena: `raw` and `recognize` now also place one **static** distractor beside the two moving ones. The table above is the earlier run without that object. The numbers below are the new comparison (`recognize_compare.json`). `fixed` and `blind` still have no distractors.

`--percept recognize` inserts one unsupervised layer per agent between the camera/lidar vector and PN. It never reads zone reward `r` and never receives a class label. Until four self-similar blobs have been stored, its output is the speed-and-size gate, so a fast small object scores like a walking dog and a stopped dog scores low (`invariance` at t=0 is −0.85). After that the output is cosine similarity to the Hebbian prototype.

| t (s) | dog | dog still | moving distractor | fast distractor | static | sep moving | sep static | invariance | purity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.96 | 0.11 | 0.30 | 0.96 | 0.09 | 0.66 | 0.88 | −0.85 | 3.00 |
| 5 | 0.96 | 0.87 | 0.32 | 0.42 | 0.28 | 0.64 | 0.68 | +0.46 | 2.73 |
| 60 | 0.97 | 0.91 | 0.27 | 0.34 | 0.23 | 0.70 | 0.73 | +0.57 | 2.87 |

Mean over 3 agents × seeds 0–4. Purity is out of 3 (dog assigned to the conspecific prototype, both distractor kinds assigned away from it). By 5 s a stopped dog outscores a fast distractor, and that gap holds through 60 s. The layer is doing its own job.

Downstream KC→MBON behavior does not pick up a clean dog-versus-distractor policy.

| condition | mean PI | plastic updates | weight drift | dog shift L2 | distractor shift L2 | diff L2 | approach dog | approach distractor | reward steps with dog | reward steps with distractor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.982 | 570 | 221 | 385 | 0 | 385 | 0.83 | — | 1.00 | 0 |
| raw | 0.983 | 493 | 184 | 200 | 182 | 71 | 0.86 | 0.94 | 0.68 | 0.49 |
| recognize | 0.983 | 494 | 189 | 376 | 361 | 100 | 0.85 | 0.70 | 0.67 | 0.48 |
| blind | 0.853 | 461 | 158 | 106 | 0 | 106 | — | — | 0 | 0 |
| raw_blind | 0.982 | 497 | 185 | 137 | 154 | 75 | — | 0.79 | 0 | 0.49 |

Zone learning is intact (PI near +0.98 except the same blind seed as before). Recognize `diff_l2` (100) sits only a little above raw (71) and raw-blind (75). Per seed the recognize values are 81, 72, 59, 191, 99, so the mean is pulled by one run and the rest overlap raw (45–88). Both probes move by ~360 because the distractor probe still carries likeness 0.27, and the random PN map still lets zone plasticity leak into object probes. Approach counts stay sparse and do not split cleanly (several seeds approach on every distractor-only step).

What the layer learned is the likeness score. What to do about a dog is still left to own-`r` plasticity, and that teacher remains "I am in a zone".
