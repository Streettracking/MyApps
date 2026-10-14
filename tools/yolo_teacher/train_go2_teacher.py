"""Fine-tune YOLOv8n on the Roboflow Go2 set plus our labelled frames.

Evaluation is only on a held-out slice of OUR frames (jpg + corrected txt).
Roboflow's own val split is folded into training so the printed
mAP / precision / recall are not that public split.

Laptop only. CPU works. CUDA is used when it is available and --device
is left empty.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

CLASS_GUESSES = ("go2", "unitree", "robot dog", "robot_dog", "dog", "robot")


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        print("Нет pyyaml. Он ставится вместе с ultralytics: pip install -r tools\\yolo_teacher\\requirements_yolo.txt", file=sys.stderr)
        raise SystemExit(2)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("пустой data.yaml: %s" % path)
    return data


def find_data_yaml(root: Path) -> Path:
    direct = root / "data.yaml"
    if direct.is_file():
        return direct
    found = sorted(root.rglob("data.yaml"))
    if not found:
        raise SystemExit("В %s нет data.yaml. Сначала fetch_roboflow.py." % root)
    return found[0]


def class_names(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise SystemExit("В data.yaml нет names.")


def go2_class(names: list[str]) -> int:
    folded = [name.lower() for name in names]
    for guess in CLASS_GUESSES:
        for i, name in enumerate(folded):
            if guess in name:
                return i
    return 0


def our_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    if not root.is_dir():
        return pairs
    for jpg in sorted(root.rglob("*.jpg")):
        txt = jpg.with_suffix(".txt")
        if txt.is_file():
            pairs.append((jpg, txt))
    return pairs


def split_ours(pairs: list[tuple[Path, Path]], holdout: float, seed: int) -> tuple[list, list]:
    rows = list(pairs)
    random.Random(seed).shuffle(rows)
    if not rows:
        return [], []
    n_hold = int(round(len(rows) * float(holdout)))
    n_hold = max(1, min(n_hold, len(rows) - (1 if len(rows) > 1 else 0)))
    if len(rows) == 1:
        return [], rows
    return rows[n_hold:], rows[:n_hold]


def roboflow_images(data_yaml: Path, data: dict) -> list[Path]:
    root = data_yaml.parent
    images = []
    for key in ("train", "val", "valid", "test"):
        entry = data.get(key)
        if not entry or not isinstance(entry, str):
            continue
        folder = (root / entry).resolve()
        if folder.is_dir():
            images.extend(sorted(folder.rglob("*.jpg")))
            images.extend(sorted(folder.rglob("*.jpeg")))
            images.extend(sorted(folder.rglob("*.png")))
    return images


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(str(src), str(dest))
    except OSError:
        shutil.copy2(str(src), str(dest))


def _remap(src: Path, dest: Path, class_id: int) -> None:
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        parts[0] = str(int(class_id))
        lines.append(" ".join(parts[:5]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _label_for_roboflow(image: Path) -> Path | None:
    text = str(image)
    needle = "%simages%s" % (os.sep, os.sep)
    if needle not in text:
        sibling = image.with_suffix(".txt")
        return sibling if sibling.is_file() else None
    swapped = text.replace(needle, "%slabels%s" % (os.sep, os.sep), 1)
    candidate = Path(swapped).with_suffix(".txt")
    return candidate if candidate.is_file() else None


def build_tree(work: Path, robo_images: list[Path], train_ours, hold_ours, class_id: int, names: list[str]) -> Path:
    if work.exists():
        shutil.rmtree(work)
    for split, rows in (("train", None), ("val", None)):
        (work / "images" / split).mkdir(parents=True, exist_ok=True)
        (work / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(robo_images):
        dest = work / "images" / "train" / ("rf_%06d%s" % (i, image.suffix.lower()))
        _link(image, dest)
        label = _label_for_roboflow(image)
        if label is not None:
            _link(label, work / "labels" / "train" / (dest.stem + ".txt"))
        else:
            (work / "labels" / "train" / (dest.stem + ".txt")).write_text("", encoding="utf-8")
    for split, pairs in (("train", train_ours), ("val", hold_ours)):
        for i, (jpg, txt) in enumerate(pairs):
            dest = work / "images" / split / ("ours_%06d.jpg" % i)
            _link(jpg, dest)
            _remap(txt, work / "labels" / split / (dest.stem + ".txt"), class_id)
    yaml_path = work / "data.yaml"
    yaml_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnc: %s\nnames:\n%s"
        % (
            work.resolve().as_posix(),
            len(names),
            "\n".join("  %s: \"%s\"" % (i, name.replace("\"", "'")) for i, name in enumerate(names)),
        ),
        encoding="utf-8",
    )
    return yaml_path


def print_metrics(metrics) -> None:
    box = getattr(metrics, "box", None)
    mean_ap = getattr(box, "map", None)
    precision = getattr(box, "mp", None)
    recall = getattr(box, "mr", None)
    map50 = getattr(box, "map50", None)
    print("held-out наши кадры")
    print("mAP50-95  %s" % mean_ap)
    print("mAP50     %s" % map50)
    print("precision %s" % precision)
    print("recall    %s" % recall)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8n. Metrics are on our held-out frames.")
    p.add_argument("--roboflow", type=Path, default=Path("datasets") / "unitree_go2")
    p.add_argument("--ours", type=Path, default=Path("logs") / "yolo_frames")
    p.add_argument("--work", type=Path, default=Path("runs") / "yolo_teacher" / "dataset")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="", help="Empty: CUDA if torch sees it, otherwise cpu.")
    args = p.parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Нет пакета ultralytics. На ноутбуке: pip install -r tools\\yolo_teacher\\requirements_yolo.txt", file=sys.stderr)
        return 2
    data_yaml = find_data_yaml(args.roboflow)
    data = _load_yaml(data_yaml)
    names = class_names(data)
    class_id = go2_class(names)
    pairs = our_pairs(args.ours)
    if not pairs:
        print(
            "В %s нет пар jpg+txt. Сначала prelabel_yoloworld.py и правка боксов. "
            "Без наших меток считать mAP на наших кадрах не на чем." % args.ours,
            file=sys.stderr,
        )
        return 2
    train_ours, hold_ours = split_ours(pairs, args.holdout, args.seed)
    robo = roboflow_images(data_yaml, data)
    if not robo:
        print("В %s нет картинок train/val." % data_yaml.parent, file=sys.stderr)
        return 2
    print("классы %s" % names)
    print("наши метки пишутся в класс %s (%s)" % (class_id, names[class_id]))
    print("roboflow картинок %s, наших в обучении %s, наших в проверке %s" % (len(robo), len(train_ours), len(hold_ours)))
    yaml_path = build_tree(args.work, robo, train_ours, hold_ours, class_id, names)
    device = args.device
    if not device:
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print("устройство %s" % device)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=device,
        project=str(Path("runs") / "yolo_teacher"),
        name="go2n",
        exist_ok=True,
        workers=0 if device == "cpu" else 2,
    )
    metrics = model.val(data=str(yaml_path), split="val", device=device, imgsz=int(args.imgsz), batch=int(args.batch))
    print_metrics(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
