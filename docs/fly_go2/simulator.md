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

## Сырые сенсоры (`--percept raw`)

По умолчанию `--percept fixed`: другие собаки входят в PN уже подписанным cue `peer`.

`--percept raw` этот ярлык убирает. Каждая особь видит грубую эгоцентрическую камеру и сектора лидара (пол, тела, движение, близость) и проецирует их своей фиксированной случайной картой на PN. Квадраты на арене — дистракторы, не собаки. На панели мозга над PN рисуется полоска raw camera + lidar. Слепой контроль: `--blind-peers` (в raw собаки пропадают из кадра, дистракторы остаются).

```powershell
python -m sim.run_sim --percept raw --seconds 180
python -m sim.run_sim --headless --percept raw --seconds 60 --log-dir logs\sim_raw
python -m sim.compare_percept --seconds 60 --seeds 5
```

Сравнение пишет `logs/percept_compare.json`. Метрика `diff_l2` — насколько сдвиг action scores на пробник «собака» отличается от сдвига на пробник «дистрактор» после пластичности. Единственное подкрепление — собственный `r` зоны.
