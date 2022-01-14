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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
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
ENTITY = "find-card"
PROJECT_NAME = "Find-Card"
# TODO : Fix SSIM
# TODO : Check Other Models
# TODO : YoloV1
# TODO : YoloV2
# TODO : YoloV3
# TODO : YoloV4
# TODO : YoloV5
# TODO : OpenCV
