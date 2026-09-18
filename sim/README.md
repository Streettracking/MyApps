# FlyWire MB × Go2 Arena Simulator

Кроссплатформенный симулятор (Windows / Linux / macOS): несколько агентов с
индивидуальным рантаймом `artifacts/connectome_mb_v1.npz`, виртуальные/проецируемые
зоны A/B, локальный `r`, peer-cues только через «зрение» особи.

## Windows (рекомендуется)

1. Установите [Python 3.11+](https://www.python.org/downloads/) и отметьте **Add Python to PATH**.
2. В PowerShell:

```powershell
cd path\to\MyApps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-sim.txt
python -m sim.run_sim --seconds 120
```

Окно pygame: арена, зоны, агенты, PI в боковой панели.

### Клавиши

| Клавиша | Действие |
|---|---|
| `SPACE` | пауза |
| `M` | вкл/выкл движение проекций зон |
| `P` | вкл/выкл «проекцию» (видимость cue зон) |
| `ESC` | выход |

### Headless (без окна, для тестов/логов)

```powershell
python -m sim.run_sim --headless --seconds 90 --move-zones --log logs\sim.csv --metrics-out logs\metrics.json
```

Контроль без соц. зрения:

```powershell
python -m sim.run_sim --headless --blind-peers --seconds 90 --metrics-out logs\metrics_blind.json
```

## Что моделируется

- PN→KC→MBON из FlyWire MB subgraph
- пластичность KC→MBON по DAN-маскам (aversive/appetitive)
- `r` только из локальной позы и полигонов зон
- conspecific cues только если агент «видит» соседа (FOV + дистанция)
- опционально движущиеся проекции зон (`--move-zones` / клавиша `M`)

Не моделируется: полная динамика Unitree, лидарный SLAM, ROS2.

## Требования

См. `requirements-sim.txt` (`numpy`, `pygame`).
