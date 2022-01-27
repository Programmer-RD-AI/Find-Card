# Imports
import matplotlib.pyplot as plt
import gc
import threading
import urllib.request
import numpy as np
import pandas as pd
import torch

try:
    from ray import tune
except:
    tune = None
try:
    from tqdm import tqdm
except Exception as e:
    raise ImportError(
        f"""
        Cannot Import Tqdm try installing it using 
        `pip3 install tqdm` 
        or 
        `conda install tqdm`.
        \n 
        {e}"""
    )

import ast
import os

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
from Model.dataset import *
from Model.help_funcs import *
from Model.metrics import *
from Model.modelling import *
