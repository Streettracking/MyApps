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

## Слой узнавания (`--percept recognize`)

`--percept recognize` оставляет ту же камеру и лидар, что и `raw`, и ставит перед PN отдельный слой. Он не видит `r` и не получает ярлык «собака». Каждая особь учит свой прототип: ворота самоподобия (своя скорость ходьбы и угловая ширина своего тела) решают, какие блобы двигать прототип сородича, а какие — прототип прочего. На PN уходит скаляр likeness, умноженный на разреженный паттерн, которым в `fixed` пользовался готовый детектор. Пластичность KC→MBON по-прежнему только от собственного `r`.

В `raw` и `recognize` на арене два движущихся дистрактора и один неподвижный (серый контур). На панели мозга, над PN, видна стадия recognition: полоса likeness и столбики выученного прототипа.

```powershell
python -m sim.run_sim --percept recognize --seconds 180
python -m sim.compare_percept --seconds 60 --seeds 5
```

Кривая разделимости в отчёте: `sep_moving` (собака − движущийся дистрактор), `sep_static` (собака − неподвижный), `invariance` (остановившаяся собака − быстрый маленький объект), `purity` (0–3). Рост `invariance` из отрицательных значений означает, что слой перестал отвечать одной только скоростью.

## Тренировка узнавания без зон

Один ученик, без полигонов A/B и без награды `r`. Другие агенты только ходят в кадре. По умолчанию учится грибовидное тело: сырые признаки → PN → KC → MBON, пластичность только KC→MBON. DAN по умолчанию — оператор: `T` и кнопка «ЛАКОМСТВО» дают аппетитивный PAM, `X` даёт аверсивный PPL1. Каждая такая подача видна на шкале DAN. Знакомство — второй режим, `--dan familiarity`: повтор глушит MBON новизны, и это не признак «собака». Слой Хебба из раздела выше — `--learner hebb`.

Крупно на мониторе: «УЗНАЮ СОРОДИЧА» или «НЕ УЗНАЮ» и уверенность 0–100%. Формула — в §6.5 `design_constraints.md`: сравнение сырого выхода с тихими кадрами того же окна (низкая энергия сенсора), без клавиш D/N/T/X. Панель «насколько обучен» показывает время и лакомства этой сессии и всех прошлых. Разделение D−N и точность индикатора появляются, только когда оператор жмёт `D` или `N`; эти клавиши в узнавание не входят. `B` включает короткий звук на переход в «УЗНАЮ» (по умолчанию выключен). Если файла состояния нет, `--load` стартует с нуля и пишет это в журнал. Если файл уже есть, и `python -m`, и exe его загружают. Чистый прогон — другой `--state` или удалить `logs\mb_train_state.npz`.

Симулятор:

```powershell
python -m sim.run_sim --mode recognize_train --agents 3
python tools\recognize_trainer_entry.py --sim
python -m sim.recognize_train --headless --seconds 40
python -m sim.recognize_train --dan familiarity --headless --seconds 40
```

Стрелки водят ученика и не пересекаются с `T`. Если их отпустить, он сам идёт то к другой собаке, то к дистрактору. `Space` — стоп, `P` — пауза обучения, `R` — сброс весов и счётчиков, `B` — звук узнавания, `S` / `L` — сохранить / загрузить `logs\mb_train_state.npz`, `F12` — снимок окна, `Esc` — выход. В headless скрипт сам жмёт `T`, когда в кадре только другая собака.

Живой робот, одно окно и для вождения, и для обучения. Грибовидное тело собаку не ведёт. Порядок для Димы:

1. Превью робота на порту 8088 (`/camera.jpg`, `/lidar.jpg`).
2. По желанию мост управления: `python main.py` в `go2_wr_server_v2-v2` (UDP JSON на `127.0.0.1:5451`). Без него клавиши покажут ошибку UDP, сенсоры при живом превью всё равно учатся.
3. Тренажёр:

```powershell
python -m sim.recognize_train_live --robot-ip 192.168.35.213 --preview-port 8088 --udp-host 127.0.0.1 --udp-port 5451
python tools\recognize_trainer_entry.py
```

Пока окно в фокусе: стрелки шлют `{"method":"Move","params":{"x":±0.5,"y":0.0,"z":±1}}` примерно каждые 100 мс (влево `z=+1`, вправо `z=-1`), отпускание и `Space` — `StopMove`, `-` — `StandDown`, `+` — `StandUp`, клавиша `E` и кнопка E-STOP — `{"method":"emergency_stop"}`. Закрытие окна тоже шлёт `StopMove`. `T` шлёт PAM в KC→MBON и не шлёт `Move`. Если превью недоступно, окно пишет адрес, по которому не достучалось, и кадры в обучение не идут; headless в этом случае завершается с ненулевым кодом и тем же текстом.

Сборка одного exe. Коннектом нужен внутри пакета (`--add-data`). На Windows разделитель `;`, на Linux `:`.

```powershell
pyinstaller --onefile --noconfirm --name recognize_trainer --collect-all pygame --add-data "artifacts\connectome_mb_v1.npz;artifacts" --hidden-import sim.recognize --hidden-import sim.recognize_train --hidden-import sim.recognize_train_live --hidden-import sim.frame_sense --hidden-import sim.train_monitor --hidden-import sim.go2_udp --hidden-import sim.raw_sense --hidden-import sim.world --hidden-import sim.mb_runtime --hidden-import sim.mb_train --hidden-import sim.mb_confidence --hidden-import numpy tools\recognize_trainer_entry.py
```

`recognize_trainer.exe` — живой тренажёр, DAN по умолчанию `T`. `recognize_trainer.exe --sim` — симулятор без робота. `recognize_trainer.exe --dan familiarity` — знакомство вместо лакомства. Файл состояния по умолчанию `logs\mb_train_state.npz` рядом с текущим каталогом. Для `--learner hebb` состояние — `logs\recognizer_state.json`.
