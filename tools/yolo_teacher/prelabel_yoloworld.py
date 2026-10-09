"""Pre-label recorded camera frames with YOLO-World. Laptop only.

Reads logs/yolo_frames/**/frame_*.jpg and the sidecar JSON. Writes a YOLO
txt (class, cx, cy, w, h, all normalized) next to each JPEG. Class 0 is a
Go2-like dog: the prompts are "robot dog" and "quadruped robot".

The report compares those boxes with the operator marks from the last 0.5 s.
T and D are a weak "dog". X and N are a weak "no dog". The latest key in
that window wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROMPTS = ["robot dog", "quadruped robot"]
WEIGHTS = "yolov8s-worldv2.pt"


def latest_key(meta: dict) -> str | None:
    op = meta.get("operator") or {}
    events = op.get("events") or []
    if not events:
        return None
    events = sorted(events, key=lambda item: float(item.get("age_s", 1e9)))
    key = events[0].get("key")
    return str(key) if key else None


def weak_label(meta: dict) -> str | None:
    op = meta.get("operator") or {}
    label = op.get("weak_label")
    if label in ("dog", "no_dog"):
        return str(label)
    return None


def agrees(label: str | None, n_boxes: int) -> bool | None:
    """True when YOLO-World's presence matches a weak dog / no-dog mark."""
    if label == "dog":
        return n_boxes > 0
    if label == "no_dog":
        return n_boxes == 0
    return None


def agreement_report(rows: list[dict]) -> dict:
    """rows: weak_label, latest_key, n_boxes."""

    def pack(subset: list[dict]) -> dict:
        judged = [row for row in subset if agrees(row.get("weak_label"), int(row.get("n_boxes") or 0)) is not None]
        hits = [row for row in judged if agrees(row.get("weak_label"), int(row.get("n_boxes") or 0))]
        return {"n": len(judged), "agree": len(hits), "rate": None if not judged else len(hits) / float(len(judged))}

    tx = [row for row in rows if row.get("latest_key") in ("T", "X")]
    dn = [row for row in rows if row.get("latest_key") in ("D", "N")]
    return {
        "frames": len(rows),
        "with_mark": pack(rows),
        "T_X": pack(tx),
        "D_N": pack(dn),
        "boxes": int(sum(int(row.get("n_boxes") or 0) for row in rows)),
    }


def format_report(report: dict) -> str:
    def line(name: str, block: dict) -> str:
        if not block["n"]:
            return "%s: нет кадров с этой меткой" % name
        rate = 100.0 * float(block["rate"])
        return "%s: %s / %s  (%.0f%%)" % (name, block["agree"], block["n"], rate)

    return "\n".join(
        [
            "кадров %s, боксов %s" % (report["frames"], report["boxes"]),
            line("метка оператора (T/X/D/N)", report["with_mark"]),
            line("только T/X", report["T_X"]),
            line("только D/N", report["D_N"]),
            "согласие: T или D и есть бокс, либо X или N и бокса нет.",
        ]
    )


def yolo_lines(result) -> list[str]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xywhn is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn.cpu().numpy()
    lines = []
    for row in xywhn:
        cx, cy, w, h = [float(v) for v in row[:4]]
        lines.append("0 %.6f %.6f %.6f %.6f" % (cx, cy, w, h))
    return lines


def iter_frames(root: Path):
    for jpg in sorted(root.rglob("frame_*.jpg")):
        meta_path = jpg.with_suffix(".json")
        meta = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        yield jpg, meta


def write_txt(jpg: Path, lines: list[str], out_root: Path | None, frames_root: Path) -> Path:
    if out_root is None:
        dest = jpg.with_suffix(".txt")
    else:
        rel = jpg.relative_to(frames_root)
        dest = out_root / rel
        dest = dest.with_suffix(".txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
    text = ("\n".join(lines) + "\n") if lines else ""
    dest.write_text(text, encoding="utf-8")
    return dest


def run(frames: Path, out: Path | None, conf: float, device: str) -> dict:
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        print(
            "Нет пакета ultralytics. На ноутбуке: pip install -r tools\\yolo_teacher\\requirements_yolo.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    model = YOLOWorld(WEIGHTS)
    model.set_classes(PROMPTS)
    rows = []
    n = 0
    for jpg, meta in iter_frames(frames):
        predict = model.predict(source=str(jpg), conf=float(conf), verbose=False, device=device)
        lines = yolo_lines(predict[0]) if predict else []
        write_txt(jpg, lines, out, frames)
        rows.append({"weak_label": weak_label(meta), "latest_key": latest_key(meta), "n_boxes": len(lines), "file": str(jpg)})
        n += 1
    report = agreement_report(rows)
    report["prompts"] = list(PROMPTS)
    report["conf"] = float(conf)
    report["weights"] = WEIGHTS
    dest_report = (out or frames) / "yolo_world_report.json"
    dest_report.parent.mkdir(parents=True, exist_ok=True)
    dest_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(report))
    print("txt рядом с кадрами" if out is None else "txt в %s" % out)
    print("отчёт %s" % dest_report)
    print("кадров размечено %s" % n)
    return report


def _self_test() -> None:
    rows = [
        {"weak_label": "dog", "latest_key": "T", "n_boxes": 1},
        {"weak_label": "no_dog", "latest_key": "X", "n_boxes": 0},
        {"weak_label": "dog", "latest_key": "D", "n_boxes": 0},
        {"weak_label": None, "latest_key": None, "n_boxes": 2},
    ]
    report = agreement_report(rows)
    if report["with_mark"]["agree"] != 2 or report["with_mark"]["n"] != 3:
        raise SystemExit("agreement count %s" % report)
    if report["T_X"]["agree"] != 2 or report["D_N"]["n"] != 1:
        raise SystemExit("T/X split %s" % report)
    if agrees("dog", 0) is not False or agrees("no_dog", 0) is not True or agrees(None, 1) is not None:
        raise SystemExit("agrees")
    print("prelabel self-test ok")
    print(format_report(report))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="YOLO-World pre-labels for logs/yolo_frames.")
    p.add_argument("--frames", type=Path, default=Path("logs") / "yolo_frames")
    p.add_argument("--out", type=Path, default=None, help="Write txt here instead of next to each JPEG.")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--device", default="", help="cuda device index, or cpu. Empty picks CUDA when torch has it.")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if not args.frames.is_dir():
        print("Нет папки %s. Сначала запись в тренажёре (кнопка «ЗАПИСЬ КАДРОВ» или U)." % args.frames, file=sys.stderr)
        return 2
    device = args.device
    if not device:
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print("устройство %s, порог %.2f, промпты %s" % (device, args.conf, ", ".join(PROMPTS)))
    run(args.frames, args.out, args.conf, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
