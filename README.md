# Road Damage Detection with YOLO

Course project at Instant: train, compare, and deploy YOLO models that detect and localize road damage.

## Models compared
YOLOv5nu, YOLOv8n, YOLO11n (baseline and tuned), all evaluated on the same test set (681 images, 7 classes).
Full numbers are in `results/final_comparison.csv`.

| Model | mAP50 | mAP50-95 | Mean IoU | Size (MB) |
|---|---|---|---|---|
| YOLOv5nu | 0.560 | 0.338 | 0.345 | 5.5 |
| YOLOv5nu (tuned) | 0.502 | 0.277 | 0.300 | 5.5 |
| **YOLOv8n** | **0.566** | **0.346** | **0.354** | 6.5 |
| YOLO11n | 0.539 | 0.311 | 0.306 | 5.5 |
| YOLO11n (tuned) | 0.499 | 0.260 | 0.288 | 5.5 |

**Findings:** YOLOv8n with default settings performed best. Hyperparameter tuning (10-epoch trials) did not transfer to full 30-epoch training and scored lower on the test set. `alligator` and `longitudinal` are the hardest classes, and `edge` does not appear in the test set.

## Classes
alligator, block, crack, edge, longitudinal, pothole, transverse

## Run the app
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:7860

## Dataset and training
Road Damage dataset from Roboflow. Training, tuning and evaluation were done in Kaggle notebooks (see `notebooks/`). Set your own Roboflow key where the notebook says `YOUR_ROBOFLOW_KEY`.
