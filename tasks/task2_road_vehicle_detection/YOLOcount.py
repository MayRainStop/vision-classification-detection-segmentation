from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "train_results" / "weights" / "best.pt"
DEFAULT_VIDEO = ROOT / "test2.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack line-crossing counter.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT / f"tracking_v8s_{timestamp}.mp4"

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    line_x = width // 2
    passed_ids: set[int] = set()
    object_positions: dict[int, float] = {}

    print("Starting tracking inference...")
    print(f"Output file: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            tracker="bytetrack.yaml",
        )

        cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 3)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                label = model.names[class_id]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} {label}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                previous_x = object_positions.get(track_id)
                if previous_x is not None and previous_x < line_x <= center_x:
                    passed_ids.add(track_id)
                object_positions[track_id] = center_x

        cv2.putText(frame, f"Total Count: {len(passed_ids)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(frame, f"Frame: {frame_count}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done. Saved to: {output_path}")


if __name__ == "__main__":
    main()
