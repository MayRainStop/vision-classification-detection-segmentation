# HW2 Task 2: Road Vehicle Detection, Tracking, and Line Crossing Count

This folder contains the HW2 Task 2 road-vehicle detection project. It fine-tunes YOLOv8s on the Road Vehicle Images Dataset, applies ByteTrack to a road-video sequence, draws bounding boxes and tracking IDs, and analyzes ID switches under occlusion.

## HW2 Requirement Alignment

| HW2 Task 2 requirement | Implementation in this folder |
| --- | --- |
| Use the Road Vehicle Images Dataset | `train/` contains 2704 training images with 21764 instances; `valid/` contains 300 validation images with 2584 instances. |
| Train a YOLOv8 detector | `YOLOtrain.py` fine-tunes `yolov8s.pt` with AdamW, `imgsz=640`, batch size 16, and data augmentation. |
| Train for the required epoch range and report the training process | The best checkpoint appears at epoch 27, inside the 10-30 epoch observation range; training continued to epoch 37 and then stopped by patience-based early stopping. Curves and validation plots are in `train_results/`. |
| Run image/video inference and show bounding boxes | `YOLOcount.py` loads `train_results/weights/best.pt`, runs inference on `test2.mp4`, and writes an annotated video. |
| Display class ID / tracking ID | The output frames draw `ID:<track_id> <class_name>` for each tracked object. |
| Analyze ID switches or tracking failures | `screenshots/frame_378.jpg`, `frame_424.jpg`, and `frame_440.jpg` document an occlusion case where one vehicle changes from ID 20 to ID 33 and another keeps ID 22 but has a class jump. |

## Layout

```text
task2_road_vehicle_detection/
|- README.md
|- HW2_Task2_Road_Vehicle_Detection.ipynb
|- YOLOtrain.py
|- YOLOcount.py
|- data.yaml
|- train/
|  |- images/
|  `- labels/
|- valid/
|  |- images/
|  `- labels/
|- train_results/
|  |- results.csv
|  |- results.png
|  |- confusion_matrix.png
|  |- BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png / BoxPR_curve.png
|  `- weights/best.pt
|- report_figures/
|  |- detection_dataset_samples.png
|  |- detection_class_distribution.png
|  `- detection_dataset_stats.json
|- screenshots/
|  |- frame_378.jpg
|  |- frame_424.jpg
|  `- frame_440.jpg
|- test2.mp4
`- tracking_analysis.mp4
```

## Training

From the repository root, install the shared requirements first:

```powershell
python -m pip install -r requirements.txt
```

Online W&B logging is optional. To enable it, set `WANDB_API_KEY` in the environment before running the script. Without that variable the script disables W&B network logging.

```powershell
cd tasks\task2_road_vehicle_detection
python YOLOtrain.py
```

The current official run used:

| Setting | Value |
| --- | --- |
| Model | YOLOv8s |
| Pretrained weights | `yolov8s.pt` |
| Dataset classes | 21 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Actual epochs | 37 |
| Best epoch | 27 |
| Device | CPU |

## Tracking and Counting

```powershell
cd tasks\task2_road_vehicle_detection
python YOLOcount.py --video test2.mp4 --model train_results\weights\best.pt --output tracking_v8s_demo.mp4
```

The script uses ByteTrack via Ultralytics, draws a vertical center line, keeps a set of track IDs that cross the line from left to right, and overlays the total count and frame index.

## Results

Validation metrics for `train_results/weights/best.pt`:

| Metric | Value |
| --- | ---: |
| Precision | 0.667 |
| Recall | 0.412 |
| mAP50 | 0.520 |
| mAP50-95 | 0.304 |

Representative class mAP50 values:

| Class | mAP50 |
| --- | ---: |
| car | 0.790 |
| rickshaw | 0.733 |
| three wheelers -CNG- | 0.724 |
| bus | 0.663 |
| truck | 0.624 |

The tracking video confirms the end-to-end pipeline. The occlusion frames in `screenshots/` show the main limitation: when a nearby SUV blocks a smaller vehicle for several frames, YOLO detections become unstable and ByteTrack may reassign a new ID.
