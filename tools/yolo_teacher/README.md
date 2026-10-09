# Учитель YOLO для кадров Go2

Это подготовка датасета на ноутбуке и локальный сервис боксов. На робота YOLO не ставится. Веса и torch живут в отдельном venv, не в Python, из которого собран `DogRecognizeTrainer.exe`.

## Авто-учитель в тренажёре

Ярлык: `tools\yolo_teacher\start_teacher.cmd` (его можно положить на рабочий стол). Он поднимает `serve.py` из `%USERPROFILE%\Desktop\FlyWire-Go2\yolo\venv` с весами `yolo\runs\go2_mix_v2\weights\best.pt` и слушает только `http://127.0.0.1:8091`.

Пока обучение выключено (`P`), кадры в сервис не идут и рамок нет: на кнопке «выкл (нет обучения)». Когда обучение включено, сервис смотрит кадр камеры и рисует рамки с conf и меткой «Л», «Л+П» или «П». Кнопка пишет «смотрит». `Y` включает раздачу PAM/PPL1 по полушариям — «учит». `Y` ещё раз выключает только подкрепления, рамки остаются. `H` прячет рамки, но только при включённом обучении; если учитель включён, подкрепления продолжаются. Выключение обучения сразу прекращает запросы и снимает рамки. `T` и `X` важнее учителя. Ход собаки учитель не трогает. Нет сервиса — тренер работает как раньше, кнопка пишет «учитель не запущен».

Решение о PAM/PPL1 берёт те же `recognized_L/R`, что на плашках, и последний ответ YOLO. Пустой ответ — собаки нет. Бокс с conf < 0.5 и бокс, чей центр в другом полушарии (даже если перекрытие 0.4 задевает этот глаз), наказание не гасят. Лимит около 2/с отдельный у PAM и у PPL1. Каждое наказание пишется в журнал: «учитель: PPL1 Л — ложное узнавание» (плашка этого глаза вспыхивает красным) и «учитель: PAM Л — собака в поле» (зелёным). Если «узнаю» есть, а PPL1 не ушёл, панель считает «ложных узнаваний без наказания» и пишет причину: лимит, бокс в поле или нет ответа YOLO.

В `--onboard` JPEG берётся с `:8088`, а PAM/PPL1 уходят на `:8090` командой `teach_sides` по тому же `/cmd`, что `T` и `X`. Рамки только на экране, в `logs\yolo_frames` их нет. `F11` и `--fullscreen` разворачивают окно, раскладка масштабируется.

```powershell
python tools\yolo_teacher\serve.py --weights C:\Users\7dtra\Desktop\FlyWire-Go2\yolo\runs\go2_mix_v2\weights\best.pt
python tools\recognize_trainer_entry.py --onboard 192.168.35.213
```

Пакеты из `requirements_yolo.txt` не входят в `recognize_trainer.exe`.

## 1. Запись кадров

В тренажёре, в любом режиме (симулятор, живой робот, `--onboard`):

- кнопка **ЗАПИСЬ КАДРОВ** или клавиша **U**
- кадр пишется только если за последние 0,5 с была метка: **T** или **D** (`dog`), **X** или **N** (`no_dog`)
- кадр без T/X/D/N не сохраняется: ни jpg, ни json
- не больше `--rec-fps` таких кадров в секунду (по умолчанию 2)
- одинаковые и слишком похожие кадры пропускаются
- сырой JPEG без рамок окна

На роботе и в `--onboard` это байты `http://<робот>:8088/camera.jpg`, их пишет ноутбук. В симуляторе это картинка камеры без оверлея.

Рядом с кадром лежит JSON: время, метки оператора за последние 0,5 с, выход грибовидного тела, режим manual/auto, скорость.

Слабая метка: **T** или **D** → `dog`, **X** или **N** → `no_dog`. Если в окне несколько клавиш, берётся последняя. Удержание T или X держит метку, пока клавиша нажата, и ещё 0,5 с после отпускания.

```powershell
python tools\recognize_trainer_entry.py --sim --rec-fps 2
python tools\recognize_trainer_entry.py --rec-fps 2
python tools\recognize_trainer_entry.py --onboard 192.168.35.213 --rec-fps 2
```

Файлы: `logs\yolo_frames\session_YYYYmmdd_HHMMSS\frame_000001.jpg` и такой же `.json`. На кнопке — число сохранённых кадров и занятое место. Пока запись включена, а метки нет, кнопка пишет «жми T/X».

## 2. Пакеты только на ноутбуке

```powershell
pip install -r tools\yolo_teacher\requirements_yolo.txt
```

## 3. Скачать публичный датасет

Ключ только из окружения. В коде его нет.

1. Бесплатный аккаунт: https://app.roboflow.com
2. Private API Key: https://app.roboflow.com/settings/api
3. Датасет: https://universe.roboflow.com/robot-44qco/unitree_go2 (версия 3, формат YOLOv8)

```powershell
$env:ROBOFLOW_API_KEY = "ваш_ключ"
python tools\yolo_teacher\fetch_roboflow.py
```

Папка по умолчанию: `datasets\unitree_go2`.

## 4. Предразметка YOLO-World

Промпты: `robot dog`, `quadruped robot`. Для каждого кадра пишется YOLO txt (класс 0, центр и размер в долях кадра). Пустой файл — бокса нет.

```powershell
python tools\yolo_teacher\prelabel_yoloworld.py --frames logs\yolo_frames
```

В конце печатается, как часто бокс совпал с меткой оператора: T/D и бокс есть, X/N и бокса нет. Отдельно строка только по T/X. Отчёт: `logs\yolo_frames\yolo_world_report.json`.

На CPU первый запуск скачает веса `yolov8s-worldv2.pt` и будет медленным. Если есть CUDA, скрипт берёт её сам.

## 5. Поправить боксы

Txt лежит рядом с jpg (`frame_000001.txt`). Его правят в CVAT, Roboflow или Label Studio и кладут обратно с тем же именем. Класс 0 — собака-робот. Пустой txt — в кадре её нет. JSON трогать не нужно.

## 6. Доучить YOLOv8n и проверить на наших кадрах

Около 20% наших размеченных кадров откладываются и в обучение не попадают. mAP, precision и recall печатаются по этой отложенной части, не по валидации Roboflow.

```powershell
python tools\yolo_teacher\train_go2_teacher.py --roboflow datasets\unitree_go2 --ours logs\yolo_frames
```

Только CPU:

```powershell
python tools\yolo_teacher\train_go2_teacher.py --roboflow datasets\unitree_go2 --ours logs\yolo_frames --device cpu --epochs 30 --batch 8
```

Без наших txt скрипт останавливается: считать mAP на наших кадрах не на чем.
