from __future__ import annotations

import os
from pathlib import Path

import wandb
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data.yaml"


def main() -> None:
    wandb_key = os.environ.get("WANDB_API_KEY")
    wandb_mode = "online" if wandb_key else "disabled"
    if wandb_key:
        wandb.login(key=wandb_key, relogin=True)

    run = wandb.init(
        project="HW2_Road_Vehicle_Detection",
        name="yolov8s_adamw_lr001_v2",
        mode=wandb_mode,
        config={
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 16,
            "optimizer": "AdamW",
            "imgsz": 640,
            "patience": 10,
        },
    )

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.001,
        optimizer="AdamW",
        patience=10,
        augment=True,
        project=str(ROOT),
        name="train_results",
        exist_ok=True,
        save=True,
        device="cpu",
        val=True,
        plots=True,
    )

    run.finish()


if __name__ == "__main__":
    main()
