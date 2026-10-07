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

## Панель активности MB

В GUI справа от арены — схема грибовидного тела выбранного агента:

- слои **PN**, **KC**, **MBON** красятся по полям последнего `MBForward` (`pn`, `kc`, `mbon`);
- KC усредняются в сетку бинов, чтобы кадр оставался лёгким при ~5k клетках;
- индикатор **DAN** берёт знак локального `r` (PPL1 при штрафе, PAM при награде);
- полосы показывают action scores того же прямого прохода.

Клавиши: `1`/`2`/`3` — агент на панели, `Space` — пауза, `Z` — движение зон, `Esc` — выход.
Headless (`--headless`) окно не открывает и панель не считает.
