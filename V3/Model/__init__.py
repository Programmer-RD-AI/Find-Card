# Imports
try:
    from ray import tune
except:
    tune = None
import matplotlib.pyplot as plt
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
