# MyApps

## FlyWire MB × Unitree Go2

Индивидуальный контроллер на **урезанном коннектоме грибовидного тела Drosophila (FAFB-783)** и арена из нескольких Go2.

### Данные мозга в репозитории

| Файл | Содержание |
|---|---|
| [`artifacts/connectome_mb_v1.npz`](artifacts/connectome_mb_v1.npz) | PN/KC/MBON/DAN граф + маски пластичности |
| [`artifacts/manifest.json`](artifacts/manifest.json) | метаданные, counts, sha256 |
| [`artifacts/connectome_mb_v1_neurons.csv.gz`](artifacts/connectome_mb_v1_neurons.csv.gz) | нейроны subgraph |
| [`artifacts/connectome_mb_v1_edges.csv.gz`](artifacts/connectome_mb_v1_edges.csv.gz) | рёбра syn≥5 |
| [`data/flywire/raw/`](data/flywire/raw/) | аннотации + checksum полного connectivity |

Полный `proofread_connections_783.feather` (~813MB) в git не кладётся — скачать:
`python scripts/flywire/download_sources.py`

### Симулятор (Windows OK)

```powershell
pip install -r requirements-sim.txt
python -m sim.run_sim --seconds 120
python -m sim.run_sim --headless --move-zones --seconds 90
```

См. [`sim/README.md`](sim/README.md).

### Документация

- [`docs/fly_go2/design_constraints.md`](docs/fly_go2/design_constraints.md) — individuum-first, запреты, метрики
- [`docs/fly_go2/physical_setup.md`](docs/fly_go2/physical_setup.md) — помещение, оборудование, безопасность
- [`docs/fly_go2/CITATIONS.md`](docs/fly_go2/CITATIONS.md) — цитирование FlyWire
- [`configs/examples/`](configs/examples/) — конфиги эпизода

Кратко: внешнее задаёт условия и логирует; социальность учится только внутри особи.
