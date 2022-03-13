# Imports
import ast
from concurrent.futures.process import _ExceptionWithTraceback
import gc
import os
import threading

try:
    from urllib.request import urlretrieve  # Python 3
except ImportError:
    from urllib import urlretrieve  # Python 2

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
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
from torchmetrics import PSNR, SSIM, AveragePrecision

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
try:
    import ray
    from ray import tune
except Exception as e:
    tune = None

from Model.dataset import *
from Model.help_funcs import *
from Model.metrics import *
from Model.modelling import *
from Model.param_tunning import *
