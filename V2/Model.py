import random
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from numpy.lib.npyio import save
from tqdm import tqdm
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from torchmetrics import MeanSquaredError, MeanAbsoluteError, SSIM, PSNR
from detectron2 import model_zoo
import wandb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch
import torchvision
import detectron2
import json
import ast
import tensorboard
import os
from detectron2.utils.logger import setup_logger

setup_logger()

ENTITY = "find-card"
PROJECT_NAME = "Find-Card"


class Model:
    def __init__(
        self,
        base_lr=0.00025,
        data=pd.read_csv("./Data.csv").sample(frac=1),
        labels=["Card"],
        max_iter=50,
        eval_period=5,
        ims_per_batch=2,
        batch_size_per_image=128,
        score_thresh_test=0.625,
        model="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        name="baseline",
        create_target_and_preds=29,
    ):
        self.remove_files_in_output()
        self.data = data  # pd.read_csv("./Data.csv").sample(frac=1)
        self.labels = labels  # ["Card"]
        self.tests = {
            "models": [
                "fast_rcnn_R_50_FPN_1x.yaml",
                "faster_rcnn_R_50_C4_1x.yaml",
                "faster_rcnn_R_50_C4_3x.yaml",
                "faster_rcnn_R_50_DC5_1x.yaml",
                "faster_rcnn_R_50_DC5_3x.yaml",
                "retinanet_R_50_FPN_1x.py",
                "retinanet_R_50_FPN_1x.yaml",
                "retinanet_R_50_FPN_3x.yaml",
                "rpn_R_50_C4_1x.yaml",
                "rpn_R_50_FPN_1x.yaml" "faster_rcnn_R_50_FPN_1x.yaml",
                "faster_rcnn_R_50_FPN_3x.yaml",
                "faster_rcnn_R_101_DC5_3x.yaml",
                "faster_rcnn_R_101_FPN_3x.yaml",
                "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
            ],
            "max_iters": [50, 100, 125, 250, 500, 1000, 2000, 2500, 5000],
            "base_lrs": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
            "ims_per_batchs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "batch_size_per_images": [8, 16, 32, 64, 128, 256, 512],
        }
        DatasetCatalog.register("data", lambda: self.load_data())
        MetadataCatalog.get("data").set(thing_classes=self.labels)
        self.metadata = MetadataCatalog.get("data")
        DatasetCatalog.register("test", lambda: self.load_data(test=True))
        MetadataCatalog.get("test").set(thing_classes=self.labels)
        self.metadata_test = MetadataCatalog.get("test")
        self.BASE_LR = base_lr
        self.MAX_ITER = max_iter
        self.EVAL_PERIOD = eval_period
        self.IMS_PER_BATCH = ims_per_batch
        self.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        self.SCORE_THRESH_TEST = score_thresh_test
        self.model = model
        self.NAME = name
        self.cfg = self.create_cfg()
        self.create_target_and_preds_iter = create_target_and_preds
        self.remove_files_in_output()

    @staticmethod
    def remove_files_in_output():
        files_to_remove = os.listdir("./output/")
        for file_to_remove in files_to_remove:
            os.remove(f"./output/{file_to_remove}")

    def test(self, data_idx):
        info = self.data.iloc[data_idx]
        img = cv2.imread(f'./Img/{info["Path"]}')
        height, width = cv2.imread("./Img/" + info["Path"]).shape[:2]
        xmin, ymin, xmax, ymax = info["XMin"], info["YMin"], info["XMax"], info["YMax"]
        xmin = round(xmin * width)
        xmax = round(xmax * width)
        ymin = round(ymin * height)
        ymax = round(ymax * height)
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        x, y, w, h = round(x), round(y), round(w), round(h)
        roi = img[y : y + h, x : x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 10)
        return img, roi

    def load_data(self, test=False):
        if test is True:
            if "data.npy" in os.listdir("./"):
                self.data = np.load("./data.npy", allow_pickle=True)
                self.data = self.data[:62]
                return self.data
        if "data.npy" in os.listdir("./"):
            self.data = np.load("./data.npy", allow_pickle=True)
            return self.data
        new_data = []
        for idx in tqdm(range(len(self.data))):
            record = {}
            info = self.data.iloc[idx]
            height, width = cv2.imread("./Img/" + info["Path"]).shape[:2]
            xmin, ymin, xmax, ymax = (
                info["XMin"],
                info["YMin"],
                info["XMax"],
                info["YMax"],
            )
            xmin = round(xmin * width)
            xmax = round(xmax * width)
            ymin = round(ymin * height)
            ymax = round(ymax * height)
            record["file_name"] = "./Img/" + info["Path"]
            record["height"] = height
            record["width"] = width
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

    def save(self, **kwargs):
        torch.cuda.empty_cache()
        files_and_object = kwargs
        for files_and_object_key, files_and_object_key in tqdm(
            zip(files_and_object.keys(), files_and_object.values())
        ):
            torch.save(
                files_and_object_key, f"./models/{files_and_object_key}-{self.NAME}.pt"
            )
            torch.save(
                files_and_object_key, f"./models/{files_and_object_key}-{self.NAME}.pth"
            )
        torch.cuda.empty_cache()

    def create_cfg(self):
        torch.cuda.empty_cache()
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model))
        cfg.DATASETS.TRAIN = ("data",)
        cfg.DATASETS.TEST = ()
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)
        cfg.SOLVER.MAX_ITER = self.MAX_ITER
        cfg.TEST.EVAL_PERIOD = self.EVAL_PERIOD
        cfg.SOLVER.BASE_LR = self.BASE_LR
        cfg.SOLVER.STEPS = []
        cfg.SOLVER.IMS_PER_BATCH = self.IMS_PER_BATCH
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.labels)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.BATCH_SIZE_PER_IMAGE
        torch.cuda.empty_cache()
        return cfg

    def __train(
        self,
    ):
        torch.cuda.empty_cache()
        trainer = DefaultTrainer(self.cfg)
        torch.cuda.empty_cache()
        trainer.resume_or_load(resume=False)
        torch.cuda.empty_cache()
        trainer.train()
        torch.cuda.empty_cache()
        return trainer

    def create_predictor(self):
        torch.cuda.empty_cache()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.SCORE_THRESH_TEST
        self.cfg.MODEL.WEIGHTS = "./output/model_final.pth"
        predictor = DefaultPredictor(self.cfg)
        torch.cuda.empty_cache()
        return predictor

    def create_coco_eval(self, predictor):
        torch.cuda.empty_cache()
        evaluator = COCOEvaluator("test", output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, "test")
        metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
        torch.cuda.empty_cache()
        return metrics

    def metrics_file_to_dict(self):
        new_logs = []
        logs = open("./output/metrics.json", "r").read().split("\n")
        for log in tqdm(range(len(logs))):
            try:
                res = ast.literal_eval(logs[log])
                new_logs.append(res)
            except:
                pass
        return new_logs

    def predict_test_images(self, predictor):
        imgs = []
        torch.cuda.empty_cache()
        for img in os.listdir("./test_imgs/"):
            v = Visualizer(
                cv2.imread(f"./test_imgs/{img}")[:, :, ::-1], metadata=self.metadata
            )
            v = v.draw_instance_predictions(
                predictor(cv2.imread(f"./test_imgs/{img}"))["instances"].to("cpu")
            )
            v = v.get_image()[:, :, ::-1]
            plt.figure(figsize=(24, 12))
            plt.imshow(v)
            plt.savefig(f"./preds/{img}")
            plt.close()
            imgs.append([f"./test_imgs/{img}", v])
        torch.cuda.empty_cache()
        return imgs

    def create_target_and_preds(self, predictor):
        info = self.data[self.create_target_and_preds_iter]
        print(info)
        img = cv2.imread(info["file_name"])
        height, width = cv2.imread(info["file_name"]).shape[:2]
        xmin, ymin, xmax, ymax = (
            info["annotations"][0]["bbox"][0],
            info["annotations"][0]["bbox"][1],
            info["annotations"][0]["bbox"][2],
            info["annotations"][0]["bbox"][3],
        )
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
        if (
            len(preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"])
            <= 0
        ):
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__[
                "tensor"
            ] = torch.tensor([[0, 0, 0, 0]])
        print(preds)
        target = torch.tensor([xmin, ymin, xmax, ymax])

        return (preds, target, x, y, w, h, xmin, ymin, xmax, ymax, height, width)

    def create_rmse(self, preds, target):
        lowest_rmse = 0
        r_mean_squared_error = MeanSquaredError(squared=False)
        preds_new = (
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in range(len(preds)):
            pred = preds_new[pred_i]
            if r_mean_squared_error(pred.to("cpu"), target) > lowest_rmse:
                lowest_rmse = r_mean_squared_error(pred.to("cpu"), target)
        return lowest_rmse

    def create_mse(self, preds, target):
        lowest_mse = 0
        mean_squared_error = MeanSquaredError(squared=True)
        preds_new = (
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in range(len(preds)):
            pred = preds_new[pred_i]
            if mean_squared_error(pred.to("cpu"), target) > lowest_mse:
                lowest_mse = mean_squared_error(pred.to("cpu"), target)
        return lowest_mse

    @staticmethod
    def create_x_y_w_h(xmin, ymin, xmax, ymax):
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        return x, y, w, h

    @staticmethod
    def crop_img(x, y, w, h, img):
        crop = img[y : y + h, x : x + w]
        return crop

    def create_ssim(self, preds, target, height, width):
        lowest_ssim = 0
        ssim = SSIM()
        preds_new = (
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in range(len(preds)):
            pred = preds_new[pred_i]
            print(target)
            print(pred)
            info = self.data[self.create_target_and_preds_iter]
            img = cv2.imread(info["file_name"])
            x, y, w, h = self.create_x_y_w_h(target[0], target[1], target[2], target[3])
            crop_img_target = torch.from_numpy(self.crop_img(x, y, w, h, img))
            x, y, w, h = self.create_x_y_w_h(pred[0], pred[1], pred[2], pred[3])
            crop_img_pred = torch.from_numpy(np.array(self.crop_img(x, y, w, h, img)))
            print(crop_img_pred.shape)
            print(crop_img_target.shape)
            if ssim(crop_img_pred, crop_img_target) > lowest_ssim:
                lowest_ssim = ssim(pred.to("cpu"), target)
        return lowest_ssim

    def create_psnr(self, preds, target):
        lowest_psnr = 0
        psnr = PSNR()
        preds_new = (
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in range(len(preds)):
            pred = preds_new[pred_i]
            if psnr(pred.to("cpu"), target) > lowest_psnr:
                lowest_psnr = psnr(pred.to("cpu"), target)
        return lowest_psnr

    def create_mae(self, preds, target):
        lowest_mae = 0
        mae = MeanAbsoluteError()
        preds_new = (
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in range(len(preds)):
            pred = preds_new[pred_i]
            if mae(pred.to("cpu"), target) > lowest_mae:
                lowest_mae = mae(pred.to("cpu"), target)
        return lowest_mae

    def train(self):
        wandb.init(
            entity=ENTITY,
            project=PROJECT_NAME,
            name=self.NAME,
            config={
                "BASE_LR": self.BASE_LR,
                "MAX_ITER": self.MAX_ITER,
                "EVAL_PERIOD": self.EVAL_PERIOD,
                "IMS_PER_BATCH": self.IMS_PER_BATCH,
                "BATCH_SIZE_PER_IMAGE": self.BATCH_SIZE_PER_IMAGE,
                "SCORE_THRESH_TEST": self.SCORE_THRESH_TEST,
                "MODEL": self.model,
                "NAME": self.NAME,
            },
        )
        trainer = self.__train()
        predictor = self.create_predictor()
        metrics_coco = self.create_coco_eval(predictor)
        metrics_file = self.metrics_file_to_dict()
        test_images = self.predict_test_images(predictor)
        (
            preds,
            target,
            x,
            y,
            w,
            h,
            xmin,
            ymin,
            xmax,
            ymax,
            height,
            width,
        ) = self.create_target_and_preds(predictor)
        rmse = self.create_rmse(preds, target)
        mse = self.create_mse(preds, target)
        ssim = self.create_ssim(preds, target, height, width)
        psnr = self.create_psnr(preds, target)
        wandb.log(metrics_coco)
        for metric_file in metrics_file:
            wandb.log(metric_file)
        for test_img in test_images:
            wandb.log({test_img[0]: wandb.log(wandb.Image(test_img[1]))})
        wandb.log({"RMSE": rmse})
        wandb.log({"MSE": mse})
        wandb.log({"SSIM": ssim})
        wandb.log({"PSNR": psnr})
        self.save(
            {
                "trainer": trainer,
                "predictor": predictor,
                "metrics_coco": metrics_coco,
                "metrics_file": metrics_file,
                "test_images": test_images,
                "preds": preds,
                "target": target,
                "rmse": rmse,
                "mse": mse,
                "ssim": ssim,
                "psnr": psnr,
            }
        )
        wandb.finish()
        return {
            "trainer": trainer,
            "predictor": predictor,
            "metrics_coco": metrics_coco,
            "metrics_file": metrics_file,
            "test_images": test_images,
            "preds": preds,
            "target": target,
            "rmse": rmse,
            "mse": mse,
            "ssim": ssim,
            "psnr": psnr,
        }

    def __str__(self):
        return f"""
            BASE_LR={self.BASE_LR}
            \n
            MAX_ITER={self.MAX_ITER}
            \n
            EVAL_PERIOD={self.EVAL_PERIOD}
            \n
            IMS_PER_BATCH={self.IMS_PER_BATCH}
            \n
            BATCH_SIZE_PER_IMAGE={self.BATCH_SIZE_PER_IMAGE}
            \n
            SCORE_THRESH_TEST={self.SCORE_THRESH_TEST}
            \n
            MODEL={self.model}
            \n
            NAME={self.NAME}
            \n
            Detectron2 Model
            """

    def __repr__(self):
        return f"""
            BASE_LR={self.BASE_LR}
            \n
            MAX_ITER={self.MAX_ITER}
            \n
            EVAL_PERIOD={self.EVAL_PERIOD}
            \n
            IMS_PER_BATCH={self.IMS_PER_BATCH}
            \n
            BATCH_SIZE_PER_IMAGE={self.BATCH_SIZE_PER_IMAGE}
            \n
            SCORE_THRESH_TEST={self.SCORE_THRESH_TEST}
            \n
            MODEL={self.model}
            \n
            NAME={self.NAME}
            \n
            Detectron2 Model
            """


# TODO - Create a OOP Class which Tests all possible params
# TODO - Add Comments and What is the output of the function and description,etc..
# TODO - Add Param to load the data saved
# TODO - Add @classmethod do give update of the project
# TODO - Add A Progress Bar
