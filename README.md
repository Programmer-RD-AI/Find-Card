# DetectX: Simple Yet Powerful Object Detection Framework

A lightweight and modular object detection framework powered by Detectron2, focusing on easy training and deployment.

## Core Features

- 🎯 Pre-configured Detectron2 models (Faster R-CNN, RetinaNet)
- 🔄 Simple data pipeline for custom datasets
- 📊 Built-in evaluation metrics (COCO metrics, RMSE, MSE, PSNR)
- 🚀 Easy model configuration and training

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
from Model.modelling.detectron2 import Detectron2

model = Detectron2(
    model="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    max_iter=500,
    base_lr=0.00025
)
model.train()
```

## Project Structure

```
ML/
├── Model/
│   ├── modelling/       # Core model implementations
│   ├── dataset/         # Dataset handling utilities
│   └── metrics/         # Evaluation metrics
└── tests/              # Unit tests
```

## Currently Supported

- Models: Faster R-CNN, RetinaNet
- Metrics: COCO AP, RMSE, MSE, PSNR
- Data formats: COCO-style annotations
- GPU acceleration with CUDA

## License

Apache License 2.0
