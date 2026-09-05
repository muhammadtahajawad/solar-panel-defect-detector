# Solar Panel Defect Detector (YOLOv8)

A YOLOv8-based object detection model for identifying defects on solar panels — built as a precursor study supporting my Final Year Project, an Autonomous Solar Panel Cleaning Mobile Robot.

## Overview
This project trains YOLOv8n on a public solar panel defect dataset, detecting 4 learned categories: **Bird-Drop, Clean, Defective, Dusty**. It demonstrates an end-to-end computer vision pipeline: dataset auditing, label cleanup, training, evaluation, and deployment as an interactive demo.

## Motivation
Autonomous solar panel cleaning robots need to distinguish between panels that require cleaning (dust, bird droppings) versus panels with structural damage that cleaning cannot fix. This project explores that classification problem as a standalone study.

## Dataset
- Source: [Roboflow — Solar Panel Defects](https://universe.roboflow.com/solarpanel-2me5p/solar-panel-defects-lnge0), 700 images
- **Data cleaning performed**: the original dataset had 18 inconsistently-labeled classes (e.g., "Bird Drop", "Bird-drop", "Bird_Drop" as separate classes). These were audited and merged into 7 canonical classes.
- **Class imbalance found**: of the 7 canonical classes, only 4 (Bird-Drop, Clean, Defective, Dusty) had actual labeled instances in this dataset version; Electrical-Damage, Physical-Damage, and Snow had zero annotated instances despite being defined.

## Results
| Metric | Value |
|---|---|
| mAP50 (overall) | 0.268 |
| mAP50-95 (overall) | 0.147 |
| Best class (Clean) | mAP50 = 0.403 |

Training curves and confusion matrix in `/results`.

## Known Limitations
- Model trained on aerial/thermal-style solar panel imagery; generalizes poorly to standard ground-level color photographs (out-of-distribution).
- 3 of 7 defined classes have zero training data and cannot currently be detected.
- Moderate overall accuracy (mAP50 0.268) reflects small dataset size (700 images) and YOLOv8-nano's speed/accuracy tradeoff.

## Future Work
- Retrain with YOLOv8s/m on a larger, better-balanced dataset
- Source additional images for Electrical-Damage, Physical-Damage, Snow classes
- Fine-tune on ground-level imagery to reduce domain gap

## Usage
```bash
pip install -r requirements.txt
python app.py
```

## Live Demo
[Hugging Face Space link — added after deployment]

## Author
Muhammad Taha — BS Mechatronics and Control Engineering, UET Lahore