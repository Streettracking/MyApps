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
