# FlyWire / Codex data citations

This repository redistributes a **derived mushroom-body subgraph** of the public
FlyWire female adult fly brain release (materialization **783**).

## Required citations

1. Dorkenwald et al., *Nature* (2024). Neuronal wiring diagram of an adult brain.  
   https://doi.org/10.1038/s41586-024-07558-y
2. Schlegel et al., *Nature* (2024). Whole-brain annotation and multi-connectome cell typing of Drosophila.  
   https://www.nature.com/articles/s41586-024-07686-5
3. Connectivity files: Zenodo record https://doi.org/10.5281/zenodo.10676866
4. Neuron annotations: https://github.com/flyconnectome/flywire_annotations

## Terms

Use of FlyWire/Codex resources is subject to the FlyWire Terms of Service:
https://flywire.ai/tos

The derived artifact `artifacts/connectome_mb_v1.npz` is provided for research and
engineering experiments described in `docs/fly_go2/`. It is not an official FlyWire product.

## Rebuild

```bash
python scripts/flywire/download_sources.py   # fetches ~813MB connections feather
python scripts/flywire/build_mb_subgraph.py
python scripts/flywire/verify_mb_artifact.py
```
