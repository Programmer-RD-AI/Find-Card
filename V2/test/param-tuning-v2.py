from torchmetrics import MeanSquaredError, SSIM
import os

files_to_remove = os.listdir("./output/")
for file_to_remove in files_to_remove:
    os.remove(f"./output/{file_to_remove}")

import matplotlib.pyplot as plt
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch, torchvision
import detectron2
import json
import ast
import tensorboard, os
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import pandas as pd
import wandb
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
# Version
data = pd.read_csv("./download/Data.csv").sample(frac=1)
idx = 0
# Loading Data
def load_data(data=data, test=False):
    if test is True:
        if "data.npy" in os.listdir("./"):
            data = np.load("./data.npy", allow_pickle=True)
            data = data[:125]
            print(len(data))
            return data
    if "data.npy" in os.listdir("./"):
        data = np.load("./data.npy", allow_pickle=True)
        print(len(data))
        return data
    new_data = []
    for idx in tqdm(range(len(data))):
        record = {}
        info = data.iloc[idx]
        height, width = cv2.imread("./download/Img/" + info["Path"]).shape[:2]
        xmin, ymin, xmax, ymax = info["XMin"], info["YMin"], info["XMax"], info["YMax"]
        xmin = round(xmin * width)
        xmax = round(xmax * width)
        ymin = round(ymin * height)
        ymax = round(ymax * height)
        record["file_name"] = "./download/Img/" + info["Path"]
        record["height"] = height
        record["width"] = width
        record["cateogry_id"] = 0
        objs = [
            {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
        ]
        record["image_id"] = idx
        record["annotations"] = objs
        new_data.append(record)
    np.random.shuffle(new_data)
    np.save("data.npy", new_data)
    return new_data


# Config
labels = ["Card"]
# Adding the data
DatasetCatalog.register("data", lambda: load_data())
MetadataCatalog.get("data").set(thing_classes=labels)
metadata = MetadataCatalog.get("data")
DatasetCatalog.register("test", lambda: load_data(test=True))
MetadataCatalog.get("test").set(thing_classes=labels)
metadata_test = MetadataCatalog.get("test")
# models = [
# "fast_rcnn_R_50_FPN_1x.yaml",
# "faster_rcnn_R_50_C4_1x.yaml",
# "faster_rcnn_R_50_C4_3x.yaml",
# "faster_rcnn_R_50_DC5_1x.yaml",
# "faster_rcnn_R_50_DC5_3x.yaml",
# "retinanet_R_50_FPN_1x.py",
# "retinanet_R_50_FPN_1x.yaml",
# "retinanet_R_50_FPN_3x.yaml",
# "rpn_R_50_C4_1x.yaml",
# "rpn_R_50_FPN_1x.yaml",
# "faster_rcnn_R_50_FPN_1x.yaml",
# "faster_rcnn_R_50_FPN_3x.yaml",
# "faster_rcnn_R_101_DC5_3x.yaml",
# "faster_rcnn_R_101_FPN_3x.yaml",
# "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
# ]
from detectron2.utils.logger import setup_logger

setup_logger()
model = "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"
torch.cuda.empty_cache()
wandb.init(project="Find-Card", name="Final")
torch.cuda.empty_cache()
cfg = get_cfg()
torch.cuda.empty_cache()
cfg.merge_from_file(model_zoo.get_config_file(model))
torch.cuda.empty_cache()
cfg.DATASETS.TRAIN = ("data",)
torch.cuda.empty_cache()
cfg.DATASETS.TEST = ()
torch.cuda.empty_cache()
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
torch.cuda.empty_cache()
cfg.SOLVER.MAX_ITER = 500
torch.cuda.empty_cache()
cfg.TEST.EVAL_PERIOD = 500
cfg.SOLVER.BASE_LR = 0.00025
torch.cuda.empty_cache()
cfg.SOLVER.STEPS = []
torch.cuda.empty_cache()
cfg.SOLVER.IMS_PER_BATCH = 2
torch.cuda.empty_cache()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
torch.cuda.empty_cache()
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
torch.cuda.empty_cache()
trainer = DefaultTrainer(cfg)
torch.cuda.empty_cache()
trainer.resume_or_load(resume=False)
torch.cuda.empty_cache()
trainer.train()
torch.cuda.empty_cache()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
torch.cuda.empty_cache()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
torch.cuda.empty_cache()
predictor = DefaultPredictor(cfg)
torch.cuda.empty_cache()
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.SOLVER.SCORE_THRESH_TEST = 0.75
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("test", output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "test")
metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
wandb.log(metrics)
# evaluator = COCOEvaluator("data", output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "data")
# metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
# wandb.log(metrics)
torch.cuda.empty_cache()
logs = open("./output/metrics.json", "r").read().split("\n")
for log in tqdm(range(len(logs))):
    try:
        res = ast.literal_eval(logs[log])
        wandb.log(res)
    except:
        pass
for img in os.listdir("./test_imgs/"):
    torch.cuda.empty_cache()
    v = Visualizer(cv2.imread(f"./test_imgs/{img}")[:, :, ::-1], metadata=metadata)
    torch.cuda.empty_cache()
    v = v.draw_instance_predictions(
        predictor(cv2.imread(f"./test_imgs/{img}"))["instances"].to("cpu")
    )
    torch.cuda.empty_cache()
    v = v.get_image()[:, :, ::-1]
    torch.cuda.empty_cache()
    plt.figure(figsize=(24, 12))
    torch.cuda.empty_cache()
    plt.imshow(v)
    torch.cuda.empty_cache()
    plt.savefig(f"./preds/{img}")
    torch.cuda.empty_cache()
    plt.close()
    torch.cuda.empty_cache()
    wandb.log({f"Img/{img}": wandb.Image(cv2.imread(f"./preds/{img}"))})
info = data.iloc[589]
print(info)
img = cv2.imread("./download/Img/" + info["Path"])
height, width = cv2.imread("./download/Img/" + info["Path"]).shape[:2]
xmin, ymin, xmax, ymax = info["XMin"], info["YMin"], info["XMax"], info["YMax"]
xmin = round(xmin * width)
xmax = round(xmax * width)
ymin = round(ymin * height)
ymax = round(ymax * height)
x = xmin
y = ymin
w = xmax - xmin
h = ymax - ymin
preds = predictor(img)
print(preds)
target = torch.tensor([xmin, ymin, xmax, ymax])
mean_squared_error = MeanSquaredError(squared=False)
wandb.log({"RMSE": mean_squared_error(preds, target)})
mean_squared_error = MeanSquaredError(squared=False)
wandb.log({"MSE": mean_squared_error(preds, target)})
ssim = SSIM()
wandb.log({"SSIM": ssim(preds, target)})
wandb.finish()


# Model = COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml (Currently Using) | COCO-Detection/retinanet_R_50_FPN_1x.py (Is also good)
# 5 = 1
# 16 = 1
# 32 = 2 # best
