"""Download the public Unitree Go2 set from Roboflow Universe.

Dataset: robot-44qco/unitree_go2, version 3, YOLOv8 layout.
The API key is read only from the environment variable ROBOFLOW_API_KEY.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DATASET = "robot-44qco/unitree_go2"
VERSION = 3
FORMAT = "yolov8"

KEY_HELP = """\
ROBOFLOW_API_KEY не задан. Скрипт не содержит ключа.

Бесплатный ключ:
1. Аккаунт на https://app.roboflow.com (план Public / Free).
2. Ключ: https://app.roboflow.com/settings/api  (Private API Key).
3. В PowerShell, только на эту сессию:
     $env:ROBOFLOW_API_KEY = "ваш_ключ"
4. Снова:
     python tools\\yolo_teacher\\fetch_roboflow.py

Ключ остаётся в окружении. В репозиторий и на робота он не пишется.
Датасет: https://universe.roboflow.com/robot-44qco/unitree_go2
"""


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Download robot-44qco/unitree_go2 v3 (YOLOv8).")
    p.add_argument("--out", type=Path, default=Path("datasets") / "unitree_go2", help="Folder for the YOLOv8 export.")
    p.add_argument("--version", type=int, default=VERSION)
    args = p.parse_args(argv)
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print(KEY_HELP, file=sys.stderr)
        return 2
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Нет пакета roboflow. На ноутбуке: pip install -r tools\\yolo_teacher\\requirements_yolo.txt", file=sys.stderr)
        return 2
    workspace, project = DATASET.split("/", 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)
    version = rf.workspace(workspace).project(project).version(int(args.version))
    dataset = version.download(FORMAT, location=str(out), overwrite=True)
    saved = getattr(dataset, "location", str(out))
    print("скачано %s v%s → %s" % (DATASET, args.version, saved))
    print("формат YOLOv8. data.yaml лежит в этой папке.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
