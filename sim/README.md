# FlyWire MB × Go2 Arena Simulator

Кроссплатформенный симулятор (**Windows** / Linux / macOS): несколько агентов с
индивидуальным рантаймом `artifacts/connectome_mb_v1.npz`, виртуальные/проецируемые
зоны A/B, локальный `r`, peer-cues только через «зрение» особи.

## Windows

1. Установите [Python 3.10+](https://www.python.org/downloads/) (**Add to PATH**).
2. PowerShell:

```powershell
cd path\to\MyApps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-sim.txt
python -m sim.run_sim --seconds 180 --agents 3
```

Или двойной клик: `sim\run_windows.bat`

Окно делится на арену слева и панель грибовидного тела справа. Панель показывает
выбранного агента (по умолчанию `go2_1`): слои PN, KC и MBON из последнего
`MBForward`, полосы action scores и индикатор DAN по локальному `r`
(aversive / appetitive). KC рисуются сеткой средних по бинам, не по одной точке
на каждый из ~5k нейронов.

### Клавиши GUI

| Клавиша | Действие |
|---|---|
| `1` `2` `3` | чей мозг показан на панели |
| `SPACE` | пауза |
| `Z` | вкл/выкл движение проекций зон |
| `ESC` | выход |

`--seconds 0` — окно без авто-выхода (пока не нажмёте Esc).

### Сырое восприятие других собак

```powershell
python -m sim.run_sim --percept raw --seconds 180
python -m sim.compare_percept --seconds 60 --seeds 5
```

`fixed` (по умолчанию) подаёт готовый cue «peer». `raw` подаёт камеру и лидар без ярлыка класса; на панели видна полоска raw camera + lidar. `recognize` ставит перед PN свой слой узнавания сородичей (без `r` и без ярлыка); на панели перед PN — полоса likeness. `--blind-peers` по-прежнему выключает других собак.

```powershell
python -m sim.run_sim --percept recognize --seconds 180
python -m sim.run_sim --mode recognize_train --agents 3
```

`--mode recognize_train` — один ученик без зон и без `r`. По умолчанию учится KC→MBON: `T` даёт PAM, `X` даёт PPL1. На мониторе «УЗНАЮ СОРОДИЧА» / «НЕ УЗНАЮ» и панель «насколько обучен»; уверенность считается по тихим кадрам сенсора, без D/N/T/X. `--dan familiarity` — второй учитель, `--learner hebb` — сравнение со слоем прототипа. Живой робот и сборка exe: `docs/fly_go2/simulator.md`.

### Движущиеся проекции

```powershell
python -m sim.run_sim --move-zones --seconds 180
```

### Headless (логи)

```powershell
python -m sim.run_sim --headless --seconds 90 --move-zones --log-dir logs\sim_move
python -m sim.run_sim --headless --blind-peers --seconds 90 --log-dir logs\sim_blind
```

Смотрите `logs\...\summary.json` → `mean_PI`.

## Что внутри

| Компонент | Да/нет |
|---|---|
| FlyWire MB subgraph PN→KC→MBON | да |
| Пластичность по DAN-маскам | да |
| `r` из позы + полигонов | да |
| Зоны-проекции / движение | да |
| Сородичи только локальными cues | да |
| Полная физика Unitree / ROS2 | нет (2D упрощение) |
| Панель MB в GUI (реальные активации) | да |

## Требования

`requirements-sim.txt` — `numpy`, `pygame`.
