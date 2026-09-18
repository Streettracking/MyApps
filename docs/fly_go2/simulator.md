# Симулятор арены FlyWire MB × Go2

См. также [`sim/README.md`](../../sim/README.md).

Кроссплатформенный симулятор (удобно на **Windows**): несколько агентов с индивидуальным
контроллером на `artifacts/connectome_mb_v1.npz`, виртуальные/проецируемые зоны A/B,
локальная пластичность, без внешнего social coaching.

## Быстрый старт (Windows)

```powershell
pip install -r requirements-sim.txt
python -m sim.run_sim --seconds 180 --agents 3
python -m sim.run_sim --move-zones
python -m sim.run_sim --headless --seconds 90 --log-dir logs\sim_last
```
