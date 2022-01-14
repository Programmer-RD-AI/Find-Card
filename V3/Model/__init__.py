# Imports
try:
    from ray import tune
except:
    tune = None
import random
from detectron2.config.config import CfgNode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm, tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
import cv2
from torchmetrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    Precision,
    Recall,
    SSIM,
    PSNR,
)
import wandb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import ast
import os
import matplotlib.pyplot as plt

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
