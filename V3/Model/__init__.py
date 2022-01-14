# Imports
try:
    from ray import tune
except:
    tune = None
import random
from detectron2.config.config import CfgNode
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from torchmetrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    Precision,
    Recall,
    SSIM,
    PSNR,
)
from detectron2 import model_zoo
import wandb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from detectron2.utils.logger import setup_logger

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
