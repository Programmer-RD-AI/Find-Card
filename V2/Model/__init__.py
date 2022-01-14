# Imports
try:
    from ray import tune
except:
    tune = None
import ast
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from sklearn.model_selection import ParameterGrid
from torchmetrics import (
    PSNR,
    SSIM,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    Recall,
)
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
