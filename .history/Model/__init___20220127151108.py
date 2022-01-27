# Imports
try:
    from ray import tune
except:
    tune = None
import urllib.request

import matplotlib.pyplot as plt

try:
    pass
except Exception as e:
    raise ImportError(f"""
        Cannot Import Tqdm try installing it using 
        `pip3 install tqdm` 
        or 
        `conda install tqdm`.
        \n 
        {e}""")
from Model.help_funcs import *
from Model.metrics import *
from Model.modelling.detectron import *
