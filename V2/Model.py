# Imports
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
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

# Setup Logger
setup_logger()

# Params

ENTITY = "find-card"
PROJECT_NAME = "Find-Card"

# Model
class Model:
    """
    # Function
    - __init__ = initialize and get all of the params need
    - remove_files_in_output - remove all of the file in ./output/
    - test - croping and creating a box around the img xmin,ymin, xmax, ymax
    - load_data - loading the data in the detectron2 data format
    - save - it save the object with the {name}-{wandb-name}.pt and .pth
    - create_cfg - create the config
    - __train - trains the cfg
    - create_predictor - create the predictor to predict images
    - create_coco_eval - create COCO Evalutor and tests it
    - metrics_file_to_dict - in ./output/metrics.json it logs the metrics of the model and
    - predict_test_images - predict test images
    - create_target_and_preds - create the target and predictions
    - create_rmse - Create Root-mean-square deviation
    - create_mse - Create Mean-square deviation
    - create_x_y_w_h - Conver xmin,ymin, xmax, ymax to x,y,w,h
    - crop_img - cropping the image using x,y,w,h
    - create_ssim - create SSIM # TODO it is not done yet
    - create_psnr - Peak signal-to-noise ratio (how similar is a image)
    - create_mae - Mean absolute error
    - train - trains the model
    """

    def __init__(
        self,
        base_lr: float = 0.00025,
        data: pd.DataFrame = pd.read_csv("./Data.csv").sample(frac=1),
        labels: list = ["Card"],
        max_iter: int = 500,
        eval_period: int = 5,
        ims_per_batch: int = 2,
        batch_size_per_image: int = 128,
        score_thresh_test: float = 0.625,
        model: str = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        name: str = "baseline",
        create_target_and_preds: int = 29,
    ) -> None:
        """
        - __init__ = initialize and get all of the params need
        -------------------------------------------------------
        - base_lr = the base learning rate of the model which will allow the optimizer optimze better
        - data = the data to create the dataset in detectron2 data format
            - the data will be saved in ./data.npy file
        - labels = labels of the dataset
        - max_iter = no. of epochs or how many times the model needs to go through the data
        - eval_period = step by step amount of iters that the model will be tested
        - ims_per_batch = Number of Images that is in a Batch
        - batch_size_per_image = Batch size for every image
        - score_thresh_test = how much sure does the model be to show the predictions
        - model = the model from the detectron2 model_zoo
        - name = name of the wandb log
        - create_target_and_preds = testing image
        """
        self.remove_files_in_output()
        self.data = data
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
                "rpn_R_50_FPN_1x.yaml",
                "faster_rcnn_R_50_FPN_1x.yaml",
                "faster_rcnn_R_50_FPN_3x.yaml",
                "faster_rcnn_R_101_DC5_3x.yaml",
                "faster_rcnn_R_101_FPN_3x.yaml",
                "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
            ],
            "max_iters": [50, 100, 125, 250, 500, 1000, 2000, 2500, 5000],
            "base_lrs": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
            "ims_per_batchs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "batch_size_per_images": [8, 16, 32, 64, 128, 256, 512],
        }  # Tests for Param Tunning
        DatasetCatalog.register(
            "data", lambda: self.load_data()
        )  # Registering the training data
        MetadataCatalog.get("data").set(thing_classes=self.labels)  # Adding the labels
        self.metadata = MetadataCatalog.get("data")  # Getting the metadata
        DatasetCatalog.register(
            "test", lambda: self.load_data(test=True)
        )  # Registering the test data
        MetadataCatalog.get("test").set(thing_classes=self.labels)  # Adding the labels
        self.metadata_test = MetadataCatalog.get("test")  # Getting the metadata
        self.BASE_LR = base_lr
        self.MAX_ITER = max_iter
        self.EVAL_PERIOD = eval_period
        self.IMS_PER_BATCH = ims_per_batch
        self.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        self.SCORE_THRESH_TEST = score_thresh_test
        self.model = model
        self.NAME = name
        self.cfg = self.create_cfg()  # Creating the model config
        self.create_target_and_preds_iter = create_target_and_preds
        self.remove_files_in_output()

    @staticmethod
    def remove_files_in_output() -> None:
        files_to_remove = os.listdir("./output/")  # Get the files in the directory
        print("Remove files in output directory")
        for file_to_remove in tqdm(
            files_to_remove
        ):  # Iter over the files in the directory
            os.remove(f"./output/{file_to_remove}")  # Delete the iter file

    def test(self, data_idx: int) -> list:
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
        return [img, roi]

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
        print("Loading Data")
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
        print("Save")
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
        print("Metrics file to dict")
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
        print("Predict")
        for img in tqdm(os.listdir("./test_imgs/")):
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
            ] = torch.tensor([[1, 1, 1, 1]])
        print(preds)
        target = torch.tensor([xmin, ymin, xmax, ymax])

        return (preds, target, x, y, w, h, xmin, ymin, xmax, ymax, height, width)

    def create_rmse(self, preds, target):
        lowest_rmse = 0
        r_mean_squared_error = MeanSquaredError(squared=False)
        preds_new = (
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
        )
        print("Creating RMSE")
        for pred_i in tqdm(range(len(preds))):
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
        print("Creating MSE")
        for pred_i in tqdm(range(len(preds))):
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
        cv2.imwrite("./test.png", crop)
        return crop

    def create_ssim(self, preds, target, height, width):
        lowest_ssim = 0
        ssim = SSIM()
        preds_new = (
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
        )
        print("Creating SSIM")
        for pred_i in tqdm(range(len(preds))):
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
        print("Creating PSNR")
        for pred_i in tqdm(range(len(preds))):
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
        print("Creating MAE")
        for pred_i in tqdm(range(len(preds))):
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
        # ssim = self.create_ssim(preds, target, height, width)
        psnr = self.create_psnr(preds, target)
        wandb.log(metrics_coco)
        for metric_file in metrics_file:
            wandb.log(metric_file)
        for test_img in test_images:
            wandb.log({test_img[0]: wandb.log(wandb.Image(test_img[1]))})
        wandb.log({"RMSE": rmse})
        wandb.log({"MSE": mse})
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


class Param_Tunning:
    def __init__(self, params):
        required_labels = [
            "BASE_LR",
            "LABELS",
            "MAX_ITER",
            "EVAL_PERIOD",
            "IMS_PER_BATCH",
            "BATCH_SIZE_PER_IMAGE",
            "SCORE_THRESH_TEST",
            "MODEL",
            "CREATE_TARGET_AND_PREDS",
        ]
        params_not_in_required_labels = []

        for required_label in tqdm(list(required_labels.keys())):
            if required_label not in list(params.keys()):
                params_not_in_required_labels.append(required_label)
        if params_not_in_required_labels != []:
            raise ValueError(f"{params_not_in_required_labels} are required in params")
        self.params = ParameterGrid(params)

    def tune(self):
        models = {"Model": [], "Metrics_COCO": [], "Metrics_File": []}
        for param in tqdm(self.params):
            model = Model(
                base_lr=param["BASE_LR"],
                labels=param["LABELS"],
                max_iter=param["MAX_ITER"],
                eval_period=param["EVAL_PERIOD"],
                ims_per_batch=param["IMS_PER_BATCH"],
                batch_size_per_image=param["BATCH_SIZE_PER_IMAGE"],
                score_thresh_test=param["SCORE_THRESH_TEST"],
                model=param["MODEL"],
                name=str(param),
                create_target_and_preds=param["CREATE_TARGET_AND_PREDS"],
            )
            metrics = model.train()
            metrics_coco = metrics["metrics_coco"]
            metrics_file = metrics["metrics_file"]
            models["Model"].append(param["MODEL"])
            models["Metrics_COCO"].append(metrics_coco)
            models["Metrics_File"].append(metrics_file)
        models = pd.DataFrame(models)
        models.to_csv("./tune.csv")
        return models


# TODO - Add Comments and What is the output of the function and description,etc..
# TODO - Add Param to load the data saved
# TODO - Add @classmethod do give update of the project
# TODO - Add A Progress Bar
# TODO - Define What metric does what
# TODO - Do Unit tests
# TODO - Add Funtion what will load the models and others and predict easily
# TODO - Add Raise
