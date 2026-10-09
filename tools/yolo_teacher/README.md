# Учитель YOLO для кадров Go2

Это подготовка датасета на ноутбуке. На робота ничего не ставится и ничего не пишется. Автоучитель (YOLO → PAM/PPL1 в правую половину мозга) сюда не входит.

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
