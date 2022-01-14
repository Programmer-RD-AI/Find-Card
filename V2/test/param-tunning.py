# import random
# import cv2
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.utils.visualizer import Visualizer
# from tqdm import tqdm
# from detectron2.structures import BoxMode
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor, DefaultTrainer
# from torchmetrics import MeanSquaredError
# from detectron2 import model_zoo
# import wandb
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# import torch
# import torchvision
# import detectron2
# import json
# import ast
# import tensorboard
# import os
# from detectron2.utils.logger import setup_logger

# setup_logger()
# files_to_remove = os.listdir('./output/')
# for file_to_remove in files_to_remove:
#     os.remove(f'./output/{file_to_remove}')
# data = pd.read_csv("./Data.csv").sample(frac=1)
# info = data.iloc[59]
# img = cv2.imread(f'./Img/{info["Path"]}')
# height, width = cv2.imread("./Img/" + info["Path"]).shape[:2]
# xmin, ymin, xmax, ymax = info["XMin"], info["YMin"], info["XMax"], info["YMax"]
# xmin = round(xmin * width)
# xmax = round(xmax * width)
# ymin = round(ymin * height)
# ymax = round(ymax * height)
# x = xmin
# y = ymin
# w = xmax - xmin
# h = ymax - ymin
# x, y, w, h = round(x), round(y), round(w), round(h)
# cv2.imwrite('./output.png', img)
# roi = img[y:y+h, x:x+w]
# cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 0), 10)
# # Loading Data

# def load_data(data=data, test=False):
#     if test is True:
#         if "data.npy" in os.listdir("./"):
#             data = np.load("./data.npy", allow_pickle=True)
#             data = data[:325]
#             print(len(data))
#             return data
#     if "data.npy" in os.listdir("./"):
#         data = np.load("./data.npy", allow_pickle=True)
#         print(len(data))
#         return data
#     new_data = []
#     for idx in tqdm(range(len(data))):
#         record = {}
#         info = data.iloc[idx]
#         height, width = cv2.imread("./Img/" + info["Path"]).shape[:2]
#         xmin, ymin, xmax, ymax = info["XMin"], info["YMin"], info["XMax"], info["YMax"]
#         xmin = round(xmin * width)
#         xmax = round(xmax * width)
#         ymin = round(ymin * height)
#         ymax = round(ymax * height)
#         record["file_name"] = "./Img/" + info["Path"]
#         record["height"] = height
#         record["width"] = width
#         objs = [
#             {
#                 "bbox": [xmin, ymin, xmax, ymax],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "category_id": 0,
#             }
#         ]
#         record["image_id"] = idx
#         record["annotations"] = objs
#         new_data.append(record)
#     np.random.shuffle(new_data)
#     np.save("data.npy", new_data)
#     return new_data

# # Config
# labels = ["Card"]
# # Adding the data
# DatasetCatalog.register("data", lambda: load_data())
# MetadataCatalog.get("data").set(thing_classes=labels)
# metadata = MetadataCatalog.get("data")
# DatasetCatalog.register("test", lambda: load_data(test=True))
# MetadataCatalog.get("test").set(thing_classes=labels)
# metadata_test = MetadataCatalog.get("test")

# models = [
#     "fast_rcnn_R_50_FPN_1x.yaml",
#     "faster_rcnn_R_50_C4_1x.yaml",
#     "faster_rcnn_R_50_C4_3x.yaml",
#     "faster_rcnn_R_50_DC5_1x.yaml",
#     "faster_rcnn_R_50_DC5_3x.yaml",
#     "retinanet_R_50_FPN_1x.py",
#     "retinanet_R_50_FPN_1x.yaml",
#     "retinanet_R_50_FPN_3x.yaml",
#     "rpn_R_50_C4_1x.yaml",
#     "rpn_R_50_FPN_1x.yaml"
#     "faster_rcnn_R_50_FPN_1x.yaml",
#     "faster_rcnn_R_50_FPN_3x.yaml",
#     "faster_rcnn_R_101_DC5_3x.yaml",
#     "faster_rcnn_R_101_FPN_3x.yaml",
#     "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
# ]
# max_iters = [
#     50, 100, 125, 250, 500, 1000, 2000, 2500, 5000
# ]
# base_lrs = [
#     0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001
# ]
# ims_per_batchs = [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# ]
# batch_size_per_images = [
#     8, 16, 32, 64, 128, 256, 512
# ]
# BASE_LR = 0.00025
# MAX_ITER = 500
# EVAL_PERIOD = 500
# IMS_PER_BATCH = 2
# BATCH_SIZE_PER_IMAGE = 128
# SCORE_THRESH_TEST = 0.625
# model = f"COCO-Detection/" + "faster_rcnn_X_101_32x8d_FPN_3x.yaml"
# NAME = "baseline"
#### files_to_remove = os.listdir("./output/")
# for file_to_remove in files_to_remove:
#     os.remove(f"./output/{file_to_remove}")
# setup_logger()
# torch.cuda.empty_cache()
wandb.init(
    entity="find-card",
    project="Find-Card",
    name=NAME,
    config={
        "BASE_LR": BASE_LR,
        "MAX_ITER": MAX_ITER,
        "EVAL_PERIOD": EVAL_PERIOD,
        "IMS_PER_BATCH": IMS_PER_BATCH,
        "BATCH_SIZE_PER_IMAGE": BATCH_SIZE_PER_IMAGE,
        "SCORE_THRESH_TEST": SCORE_THRESH_TEST,
        "MODEL": model,
        "NAME": NAME,
    },
)
# torch.cuda.empty_cache()
# cfg = get_cfg()
# torch.cuda.empty_cache()
# cfg.merge_from_file(model_zoo.get_config_file(model))
# torch.cuda.empty_cache()
# cfg.DATASETS.TRAIN = ("data",)
# torch.cuda.empty_cache()
# cfg.DATASETS.TEST = ()
# torch.cuda.empty_cache()
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
# torch.cuda.empty_cache()
# cfg.SOLVER.MAX_ITER = MAX_ITER
# torch.cuda.empty_cache()
# cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
# cfg.SOLVER.BASE_LR = BASE_LR
# torch.cuda.empty_cache()
# cfg.SOLVER.STEPS = []
# torch.cuda.empty_cache()
# cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
# torch.cuda.empty_cache()
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
# torch.cuda.empty_cache()
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
# torch.cuda.empty_cache()
# trainer = DefaultTrainer(cfg)
# torch.cuda.empty_cache()
# trainer.resume_or_load(resume=False)
# torch.cuda.empty_cache()
# trainer.train()
# torch.cuda.empty_cache()
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
# torch.cuda.empty_cache()
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# torch.cuda.empty_cache()
# predictor = DefaultPredictor(cfg)
# torch.cuda.empty_cache()
# cfg.MODEL.WEIGHTS = "./output/model_final.pth"
# cfg.SOLVER.SCORE_THRESH_TEST = SCORE_THRESH_TEST
# predictor = DefaultPredictor(cfg)
# evaluator = COCOEvaluator("test", output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "test")
# metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
# wandb.log(metrics)
# torch.cuda.empty_cache()
# logs = open("./output/metrics.json", "r").read().split("\n")
# for log in tqdm(range(len(logs))):
#     try:
#         res = ast.literal_eval(logs[log])
#         wandb.log(res)
#     except:
#         pass
# for img in os.listdir("./test_imgs/"):
#     torch.cuda.empty_cache()
#     v = Visualizer(cv2.imread(
#         f"./test_imgs/{img}")[:, :, ::-1], metadata=metadata)
#     torch.cuda.empty_cache()
#     v = v.draw_instance_predictions(
#         predictor(cv2.imread(f"./test_imgs/{img}"))["instances"].to("cpu")
#     )
#     torch.cuda.empty_cache()
#     v = v.get_image()[:, :, ::-1]
#     torch.cuda.empty_cache()
#     plt.figure(figsize=(24, 12))
#     torch.cuda.empty_cache()
#     plt.imshow(v)
#     torch.cuda.empty_cache()
#     plt.savefig(f"./preds/{img}")
#     torch.cuda.empty_cache()
#     plt.close()
#     torch.cuda.empty_cache()
#     wandb.log({f"Img/{img}": wandb.Image(cv2.imread(f"./preds/{img}"))})
# info = data.iloc[589]
# img = cv2.imread("./download/Img/" + info["Path"])
# height, width = cv2.imread("./download/Img/" + info["Path"]).shape[:2]
# xmin, ymin, xmax, ymax = info["XMin"], info["YMin"], info["XMax"], info["YMax"]
# xmin = round(xmin * width)
# xmax = round(xmax * width)
# ymin = round(ymin * height)
# ymax = round(ymax * height)
# x = xmin
# y = ymin
# w = xmax - xmin
# h = ymax - ymin
# preds = predictor(img)
# target = torch.tensor([xmin, ymin, xmax, ymax])
# lowest_rmse = 0
# r_mean_squared_error = MeanSquaredError(squared=False)
# preds_new = preds["instances"].__dict__[
#     "_fields"]["pred_boxes"].__dict__["tensor"]
# for pred_i in range(len(preds)):
#     pred = preds_new[pred_i]
#     if r_mean_squared_error(pred.to("cpu"), target) > lowest_rmse:
#         lowest_rmse = r_mean_squared_error(pred.to("cpu"), target)
lowest_mse = 0
mean_squared_error = MeanSquaredError(squared=True)
preds_new = preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__[
    "tensor"]
for pred_i in range(len(preds)):
    pred = preds_new[pred_i]
    if mean_squared_error(pred.to("cpu"), target) > lowest_mse:
        lowest_mse = mean_squared_error(pred.to("cpu"), target)
wandb.log({"MSE": lowest_mse})
wandb.log({"RMSE": lowest_rmse})
wandb.finish()
# torch.save(cfg, f'./models/cfg-{NAME}.pt')
# torch.save(cfg, f'./models/cfg-{NAME}.pth')
# torch.save(predictor, f'./models/predictor-{NAME}.pt')
# torch.save(predictor, f'./models/predictor-{NAME}.pth')
# torch.save(evaluator, f'./models/evaluator-{NAME}.pt')
# torch.save(evaluator, f'./models/evaluator-{NAME}.pth')
# torch.save(model, f'./models/model-{NAME}.pt')
# torch.save(model, f'./models/model-{NAME}.pth')
# torch.save(labels, f'./models/labels-{NAME}.pt')
# torch.save(labels, f'./models/labels-{NAME}.pth')
# torch.save(metrics, f'./models/metrics-{NAME}.pt')
# torch.save(metrics, f'./models/metrics-{NAME}.pth')
