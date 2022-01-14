# Imports
try:
    from ray import tune
except:
    tune = None
import os

import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

# Setup Logger
setup_logger()

# Params
params = {
    "models": [
        "fast_rcnn_R_50_FPN_1x.yaml",
        "faster_rcnn_R_50_C4_1x.yaml",
        "faster_rcnn_R_50_C4_3x.yaml",
        "faster_rcnn_R_50_DC5_1x.yaml",
        "faster_rcnn_R_50_DC5_3x.yaml",
        "retinanet_R_50_FPN_1x.py",
        "retinanet_R_50_FPN_1x.yaml",
        "retinanet_R_50_FPN_3x.yaml",
        "rpn_R_50_C4_1x.yaml",
        "rpn_R_50_FPN_1x.yaml",
        "faster_rcnn_R_50_FPN_1x.yaml",
        "faster_rcnn_R_50_FPN_3x.yaml",
        "faster_rcnn_R_101_DC5_3x.yaml",
        "faster_rcnn_R_101_FPN_3x.yaml",
        "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    ],
    "max_iters": [50, 100, 125, 250, 500, 1000, 2000, 2500, 5000],
    "base_lrs": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
    "ims_per_batchs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "batch_size_per_images": [8, 16, 32, 64, 128, 256, 512],
}
models = [
    "fast_rcnn_R_50_FPN_1x.yaml",
    "faster_rcnn_R_50_C4_1x.yaml",
    "faster_rcnn_R_50_C4_3x.yaml",
    "faster_rcnn_R_50_DC5_1x.yaml",
    "faster_rcnn_R_50_DC5_3x.yaml",
    "retinanet_R_50_FPN_1x.py",
    "retinanet_R_50_FPN_1x.yaml",
    "retinanet_R_50_FPN_3x.yaml",
    "rpn_R_50_C4_1x.yaml",
    "rpn_R_50_FPN_1x.yaml",
    "faster_rcnn_R_50_FPN_1x.yaml",
    "faster_rcnn_R_50_FPN_3x.yaml",
    "faster_rcnn_R_101_DC5_3x.yaml",
    "faster_rcnn_R_101_FPN_3x.yaml",
    "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
]
max_iters = [50, 100, 125, 250, 500, 1000, 2000, 2500, 5000]
base_lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ims_per_batchs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
batch_size_per_images = [8, 16, 32, 64, 128, 256, 512]
ENTITY = "find-card"
PROJECT_NAME = "Find-Card"
