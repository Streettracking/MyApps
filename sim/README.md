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

### Клавиши GUI

| Клавиша | Действие |
|---|---|
| `SPACE` | пауза |
| `Z` | вкл/выкл движение проекций зон |
| `ESC` | выход |

`--seconds 0` — окно без авто-выхода (пока не нажмёте Esc).

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

## Требования

`requirements-sim.txt` — `numpy`, `pygame`.
