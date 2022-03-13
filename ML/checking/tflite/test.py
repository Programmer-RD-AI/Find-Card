"""sumary_line

Keyword arguments:
argument -- description
Return: return_description
"""
import datetime
import os

import cv2
import numpy as np
import tensorflow as tf
from absl import logging
from PIL import Image
from tflite_model_maker import model_spec, object_detector
from tflite_model_maker.config import ExportFormat, QuantizationConfig

assert tf.__version__.startswith("2")
tf.get_logger().setLevel("ERROR")

logging.set_verbosity(logging.ERROR)
tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
gpus = tf.config.list_physical_devices("GPU")
