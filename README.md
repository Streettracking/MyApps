# MyApps

## FlyWire MB × Unitree Go2

Индивидуальный контроллер на **урезанном коннектоме грибовидного тела Drosophila (FAFB-783)** и арена из нескольких Go2.

### Симулятор (Windows)

```powershell
pip install -r requirements-sim.txt
python -m sim.run_sim --seconds 180 --agents 3
python -m sim.run_sim --move-zones
python -m sim.run_sim --headless --seconds 90 --log-dir logs\sim_last
```

Или `sim\run_windows.bat`. Подробнее: [`sim/README.md`](sim/README.md), [`docs/fly_go2/simulator.md`](docs/fly_go2/simulator.md).

### Данные мозга в репозитории

| Файл | Содержание |
|---|---|
| [`artifacts/connectome_mb_v1.npz`](artifacts/connectome_mb_v1.npz) | PN/KC/MBON/DAN граф + маски пластичности |
| [`artifacts/manifest.json`](artifacts/manifest.json) | метаданные, counts, sha256 |
| [`artifacts/connectome_mb_v1_neurons.csv.gz`](artifacts/connectome_mb_v1_neurons.csv.gz) | нейроны subgraph |
| [`artifacts/connectome_mb_v1_edges.csv.gz`](artifacts/connectome_mb_v1_edges.csv.gz) | рёбра syn≥5 |
| [`data/flywire/raw/`](data/flywire/raw/) | аннотации + checksum полного connectivity |

Полный feather связей (~813MB) не в git: `python scripts/flywire/download_sources.py`

### Документация

- [`docs/fly_go2/design_constraints.md`](docs/fly_go2/design_constraints.md) — individuum-first
- [`docs/fly_go2/physical_setup.md`](docs/fly_go2/physical_setup.md) — оборудование
- [`docs/fly_go2/simulator.md`](docs/fly_go2/simulator.md) — симулятор
- [`docs/fly_go2/CITATIONS.md`](docs/fly_go2/CITATIONS.md) — цитирование FlyWire
- [`configs/examples/`](configs/examples/) — конфиги эпизода

Кратко: внешнее задаёт условия и логирует; социальность учится только внутри особи.
