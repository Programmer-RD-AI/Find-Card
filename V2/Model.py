# Imports
try:
    from ray import tune
except:
    tune = None
from detectron2.config.config import CfgNode
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from torchmetrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    Precision,
    Recall,
    SSIM,
    PSNR,
)
from detectron2 import model_zoo
import wandb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch
import ast
import os
from detectron2.utils.logger import setup_logger

# Setup Logger
setup_logger()

# Params
params = {
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
}
models = [
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
]
max_iters = [50, 100, 125, 250, 500, 1000, 2000, 2500, 5000]
base_lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
ims_per_batchs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
batch_size_per_images = [8, 16, 32, 64, 128, 256, 512]
PROJECT_NAME = "Find-Card"

# Model


class Model:
    """
    This class helps anyone to train a detectron2 model for this project easily so anyone can train this model.
    """

    def __init__(
        self,
        base_lr: float = 0.00025,
        data: pd.DataFrame = pd.read_csv("./Data.csv").sample(frac=1),
        labels: list = ["Card"],
        max_iter: int = 500,
        eval_period: int = 500,
        ims_per_batch: int = 2,
        batch_size_per_image: int = 128,
        score_thresh_test: float = 0.625,
        model: str = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        name: str = "baseline",
        create_target_and_preds: int = 29,
        test_sample_size=50,
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
        self.data_other = data
        self.labels = labels
        self.devices = ["cpu", "cuda"]
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
        try:
            DatasetCatalog.register(
                "data", lambda: self.load_data()
            )  # Registering the training data
            MetadataCatalog.get("data").set(
                thing_classes=self.labels
            )  # Adding the labels
            self.metadata = MetadataCatalog.get("data")  # Getting the metadata
            DatasetCatalog.register(
                "test", lambda: self.load_data(test=True)
            )  # Registering the test data
            MetadataCatalog.get("test").set(
                thing_classes=self.labels
            )  # Adding the labels
            self.metadata_test = MetadataCatalog.get(
                "test")  # Getting the metadata
        except:
            self.metadata = MetadataCatalog.get("data")  # Getting the metadata
            self.metadata_test = MetadataCatalog.get(
                "test")  # Getting the metadata
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
        self.test_sample_size = test_sample_size
        self.config = (
            {
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
        self.remove_files_in_output()

    @staticmethod
    def remove_files_in_output() -> None:
        """
        - remove_files_in_output - remove all of the file in ./output/
        """
        files_to_remove = os.listdir(
            "./output/")  # Get the files in the directory
        try:
            files_to_remove.remove("test_coco_format.json")
        except:
            pass
        for file_to_remove in tqdm(
            files_to_remove
        ):  # Iter over the files in the directory
            os.remove(f"./output/{file_to_remove}")  # Delete the iter file

    def test(self, data_idx: int = 50) -> list:
        """
        - test - croping and creating a box around the img xmin,ymin, xmax, ymax
        -----------------------------------------------------
        - data_idx - the data index which is needed to be visualized
        """
        info = self.data_other.iloc[data_idx]  # getting the info of the index
        img = cv2.imread(f'./Img/{info["Path"]}')  # reading the img
        height, width = cv2.imread("./Img/" + info["Path"]).shape[
            :2
        ]  # getting the height and width of the image
        xmin, ymin, xmax, ymax = (
            info["XMin"],
            info["YMin"],
            info["XMax"],
            info["YMax"],
        )  # getting the xmin,ymin, xmax, ymax
        xmin = round(xmin * width)  # converting it to real xmin
        xmax = round(xmax * width)  # converting it to real xmax
        ymin = round(ymin * height)  # converting it to real ymin
        ymax = round(ymax * height)  # converting it to real ymax
        # The above is needed becuase open images gives their datasets xmin,ymin,xmax,ymax in a different way
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        x, y, w, h = round(x), round(y), round(w), round(h)
        roi = img[y: y + h, x: x + w]  # crop the image
        cv2.rectangle(
            img, (x, y), (x + w, y + h), (200, 0, 0), 10
        )  # draw box around the bbox
        return [img, roi]

    def load_data(self, test: bool = False) -> list:
        """
        - load_data - loading the data in the detectron2 data format
        -------------------------------------
        - test - if the return is supposed to be a test sample or not
            - Defalt = False and type = bool
        """
        if test is True:
            if "data.npy" in os.listdir("./"):
                self.data = np.load(
                    "./data.npy", allow_pickle=True
                )  # Loading already saved detectron2 format file
                self.data = self.data[: self.test_sample_size]  # TODO
                return self.data
        if "data.npy" in os.listdir("./"):
            self.data = np.load("./data.npy", allow_pickle=True)
            return self.data
        new_data = []
        for idx in tqdm(range(len(self.data))):  # iter over the data
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
        np.random.shuffle(new_data)  # Shuffling the data
        if test is True:
            return new_data[: self.test_sample_size]
        return new_data

    def save(self, **kwargs: dict) -> None:
        """
        - save - it save the object with the {name}-{wandb-name}.pt and .pth
        ----------------------------------------------------
        - **kwargs - like Model().save(a="b")
        """
        torch.cuda.empty_cache()
        files_and_object = kwargs
        for files_and_object_key, files_and_object_val in tqdm(
            zip(files_and_object.keys(), files_and_object.values())
        ):  # iterate over the file and object
            torch.save(
                files_and_object_val, f"./models/{files_and_object_key}-{self.NAME}.pt"
            )  # Save the file in .pt
            torch.save(
                files_and_object_val, f"./models/{files_and_object_key}-{self.NAME}.pth"
            )  # Save the file in .pth
        torch.cuda.empty_cache()

    def create_cfg(self) -> CfgNode:
        """
        - create_cfg - create the config of the model
        - other params - https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
        """
        torch.cuda.empty_cache()
        cfg = get_cfg()  # Creating a new cfg
        cfg.merge_from_file(model_zoo.get_config_file(
            self.model))  # Add the model
        cfg.DATASETS.TRAIN = ("data",)  # adding train DataSet
        cfg.DATASETS.TEST = ()
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            self.model
        )  # Adding the weights
        cfg.SOLVER.MAX_ITER = self.MAX_ITER  # Set Max iter
        cfg.TEST.EVAL_PERIOD = self.EVAL_PERIOD  # Set Eval Period
        cfg.SOLVER.BASE_LR = self.BASE_LR  # Set Base LR
        cfg.SOLVER.STEPS = []  # Set Steps
        cfg.SOLVER.IMS_PER_BATCH = self.IMS_PER_BATCH  # Set IMS_PER_BATCH
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
            self.labels)  # Set len(self.labels)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            self.BATCH_SIZE_PER_IMAGE
        )  # Set Batch_Size_Per_Image
        torch.cuda.empty_cache()
        return cfg

    def __train(
        self,
    ) -> DefaultTrainer:
        """
        - __train - trains the cfg
            this is used by Model.train() this is kind of the under function
        """
        torch.cuda.empty_cache()
        trainer = DefaultTrainer(self.cfg)  # Train the cfg  (Config)
        torch.cuda.empty_cache()
        # Resume the model or load a new model
        trainer.resume_or_load(resume=False)
        torch.cuda.empty_cache()
        trainer.train()  # training the model
        torch.cuda.empty_cache()
        return trainer

    def create_predictor(self) -> DefaultPredictor:
        """
        - create_predictor - create the predictor to predict images
        """
        torch.cuda.empty_cache()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            self.SCORE_THRESH_TEST
        )  # Setting SCORE_THRESH_TEST
        self.cfg.MODEL.WEIGHTS = (
            "./output/model_final.pth"  # The saved weights of the model
        )
        predictor = DefaultPredictor(self.cfg)  # Creating predictor
        torch.cuda.empty_cache()
        return predictor

    def create_coco_eval(
        self, predictor: DefaultPredictor, metadata: str = "test"
    ) -> dict:
        """
        - create_coco_eval - create COCO Evaluator and tests it
        -------------------------------
        - predictor - to create the evaluator
        """
        torch.cuda.empty_cache()
        evaluator = COCOEvaluator(
            metadata, output_dir="./output/")  # Create evaluator
        val_loader = build_detection_test_loader(
            self.cfg, metadata
        )  # Create data loader
        metrics = inference_on_dataset(
            predictor.model, val_loader, evaluator
        )  # Test the data with the evaluator
        torch.cuda.empty_cache()
        return metrics

    def metrics_file_to_dict(self) -> list:
        """
        - metrics_file_to_dict - in ./output/metrics.json it logs the metrics of the model
        """
        new_logs = []
        try:
            logs = open("./output/metrics.json", "r").read().split("\n")
            for log in tqdm(range(len(logs))):  # uterate over the logs
                try:
                    res = ast.literal_eval(
                        logs[log]
                    )  # convert str ("{'test':'test'}") to dict ({"test":"test"})
                    new_logs.append(res)
                except:
                    pass
            return new_logs
        except:
            return new_logs

    def predict_test_images(self, predictor: DefaultPredictor) -> list:
        """
        - predict_test_images - predict test images
        """
        imgs = []
        torch.cuda.empty_cache()
        for img in tqdm(os.listdir("./test_imgs/")):  # iterate over the test images
            v = Visualizer(
                cv2.imread(f"./test_imgs/{img}")[:,
                                                 :, ::-1], metadata=self.metadata
            )
            v = v.draw_instance_predictions(
                predictor(cv2.imread(
                    f"./test_imgs/{img}"))["instances"].to("cpu")
            )  # Draw pred boxes
            v = v.get_image()[:, :, ::-1]
            plt.figure(figsize=(24, 12))
            plt.imshow(v)  # plot the image
            plt.savefig(f"./preds/{img}")
            plt.close()
            imgs.append([f"./test_imgs/{img}", v])
        torch.cuda.empty_cache()
        return imgs

    def create_target_and_preds(self, predictor: DefaultPredictor) -> tuple:
        """
        - create_target_and_preds - create the target and predictions
        """
        info = self.data.iloc[self.create_target_and_preds_iter]
        img = cv2.imread("./Img/" + info["Path"])
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
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        preds = predictor(img)
        if (
            len(preds["instances"].__dict__["_fields"]
                ["pred_boxes"].__dict__["tensor"])
            <= 0
        ):
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__[
                "tensor"
            ] = torch.tensor([[1, 1, 1, 1]])
        target = torch.tensor([xmin, ymin, xmax, ymax])

        return (preds, target, x, y, w, h, xmin, ymin, xmax, ymax, height, width)

    def create_rmse(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_rmse - Create Root-mean-square deviation
        """
        lowest_rmse = 0
        r_mean_squared_error = MeanSquaredError(squared=False)
        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if r_mean_squared_error(pred.to("cpu"), target) > lowest_rmse:
                lowest_rmse = r_mean_squared_error(pred.to("cpu"), target)
        return float(lowest_rmse)

    def create_recall(self, preds: torch.tensor, target: torch.tensor) -> float:
        lowest_recall = 0
        recall = Recall()
        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if recall(pred.to("cpu"), target) > lowest_recall:
                lowest_recall = recall(pred.to("cpu"), target)
        return float(lowest_recall)

    def create_iou(self, preds: torch.tensor, targets: torch.tensor) -> float:
        ious = []
        for pred_box, true_box in zip(preds, targets):
            xA = max(true_box[0], pred_box[0])
            yA = max(true_box[1], pred_box[1])
            xB = min(true_box[2], pred_box[2])
            yB = min(true_box[3], pred_box[3])
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (true_box[2] - true_box[0] + 1) * \
                (true_box[3] - true_box[1] + 1)
            boxBArea = (pred_box[2] - pred_box[0] + 1) * \
                (pred_box[3] - pred_box[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            ious.append(iou)
        iou = np.mean(ious)
        return iou

    def create_mse(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_mse - Create Mean-square deviation
        """
        lowest_mse = 0
        mean_squared_error = MeanSquaredError(squared=True)
        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if mean_squared_error(pred.to("cpu"), target) > lowest_mse:
                lowest_mse = mean_squared_error(pred.to("cpu"), target)
        return float(lowest_mse)

    @staticmethod
    def create_x_y_w_h(xmin: int, ymin: int, xmax: int, ymax: int) -> list:
        """
        - create_x_y_w_h - Conver xmin,ymin, xmax, ymax to x,y,w,h
        """
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        return [x, y, w, h]

    @staticmethod
    def crop_img(x: int, y: int, w: int, h: int, img: np.array) -> np.array:
        """
        - crop_img - cropping the image using x,y,w,h
        """
        crop = img[y: y + h, x: x + w]
        cv2.imwrite("./test.png", crop)
        return crop

    def create_ssim(
        self, preds: torch.tensor, target: torch.tensor, height: int, width: int
    ) -> float:
        """
        - create_ssim - create SSIM # TODO it is not done yet
        """
        lowest_ssim = 0
        ssim = SSIM()
        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            info = self.data[self.create_target_and_preds_iter]
            img = cv2.imread(info["Path"])
            x, y, w, h = self.create_x_y_w_h(
                target[0], target[1], target[2], target[3])
            crop_img_target = torch.from_numpy(self.crop_img(x, y, w, h, img))
            x, y, w, h = self.create_x_y_w_h(
                pred[0], pred[1], pred[2], pred[3])
            crop_img_pred = torch.from_numpy(
                np.array(self.crop_img(x, y, w, h, img)))
            if ssim(crop_img_pred, crop_img_target) > lowest_ssim:
                lowest_ssim = ssim(pred.to("cpu"), target)
        return lowest_ssim

    def create_psnr(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_psnr - Peak signal-to-noise ratio (how similar is a image)
        """
        lowest_psnr = 0
        psnr = PSNR()
        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if psnr(pred.to("cpu"), target) > lowest_psnr:
                lowest_psnr = psnr(pred.to("cpu"), target)
        return lowest_psnr

    def create_mae(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_mae - Mean absolute error
        """
        lowest_mae = 0
        mae = MeanAbsoluteError()
        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if mae(pred.to("cpu"), target) > lowest_mae:
                lowest_mae = mae(pred.to("cpu"), target)
        return lowest_mae

    def create_precision(self, preds: torch.tensor, target: torch.tensor) -> float:
        lowest_precision = 0
        precision = Precision()
        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if precision(pred.to("cpu"), target) > lowest_precision:
                lowest_precision = precision(pred.to("cpu"), target)
        return lowest_precision

    def create_precision_and_recall(
        self, preds: torch.tensor, target: torch.tensor
    ) -> float:
        recall = self.create_recall(preds, target)
        precision = self.create_precision(preds, target)
        if recall > precision:
            precision_recall = precision - recall
        else:
            precision_recall = recall - precision
        return precision_recall

    def evaluation(self, predictor):
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
        psnr = self.create_psnr(preds, target)
        recall = self.create_recall(preds, target)
        precision_recall = self.create_precision_and_recall(preds, target)
        return {
            "Metrics": {
                "RMSE": rmse,
                "MSE": mse,
                "PSNR": psnr,
                # "IOU": iou,
                # "Precision": precision,
                "Recall": recall,
                "Precision Recall": precision_recall,
                # "SSIM": ssim,
            },
            "metrics_coco": metrics_coco,
            "metrics_file": metrics_file,
            "test_images": test_images,
        }

    def train(self) -> dict:
        """
        - train - trains the model
        """
        torch.cuda.empty_cache()
        wandb.init(
            project=PROJECT_NAME,
            name=str(self.NAME),
            sync_tensorboard=True,
        )
        trainer = self.__train()
        predictor = self.create_predictor()
        metrics = self.evaluation(predictor)
        wandb.log(metrics["metrics_coco"])
        for metric_file in metrics["metrics_file"]:
            wandb.log(metric_file)
        for test_img in metrics["test_images"]:
            wandb.log({test_img[0]: wandb.Image(test_img[1])})
        wandb.log(metrics["Metrics"])
        try:
            self.save(
                trainer=trainer,
                predictor=predictor,
                metrics_coco=metrics["metrics_coco"],
                metrics_file=metrics["metrics_file"],
                test_images=metrics["test_images"],
                metrics=metrics["Metrics"],
            )
        except:
            pass
        wandb.finish()
        return {
            "trainer": trainer,
            "predictor": predictor,
            "metrics_coco": metrics["metrics_coco"],
            "metrics_file": metrics["metrics_file"],
            "test_images": metrics["test_images"],
            "metrics": metrics["Metrics"],
        }

    def __str__(self) -> str:
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

    def __repr__(self) -> str:
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
    def __init__(
        self,
    ) -> None:
        f"""
        initialize the Class
        params - dict like {Model().test}
        """

    def tune(self, params: dict) -> dict:
        """
        Tune all of the parameters
        """
        final_metrics = []
        model = Model()
        params = ParameterGrid(params)
        params_iter = tqdm(params)
        for param in params_iter:
            try:
                torch.cuda.empty_cache()
                model.remove_files_in_output()
                torch.cuda.empty_cache()
                params_iter.set_description(str(param))
                torch.cuda.empty_cache()
                model = Model(
                    ims_per_batch=param["IMS_PER_BATCH"],
                    batch_size_per_image=param["BATCH_SIZE_PER_IMAGE"],
                    model="COCO-Detection/" + param["MODEL"],
                    base_lr=param["BASE_LR"],
                    name=str(param),
                )
                torch.cuda.empty_cache()
                metrics = model.train()
                torch.cuda.empty_cache()
                final_metrics.append(metrics)
            except Exception as e:
                torch.cuda.empty_cache()
        return final_metrics

    def ray_tune_func(self, config):
        """
        https://docs.ray.io/en/latest/tune/index.html
        """
        base_lr = config["BASE_LR"]
        ims_per_batch = (config["IMS_PER_BATCH"],)
        batch_size_per_image = config["BATCH_SIZE_PER_IMAGE"]
        model = "COCO-Detection/" + config["MODEL"]
        model = Model(
            base_lr=base_lr,
            model=model,
            ims_per_batch=ims_per_batch,
            batch_size_per_image=batch_size_per_image,
        )
        model.remove_files_in_output()
        metrics = model.train()
        ap = metrics["metrics_coco"]["bbox.AP"]
        tune.report(average_precisions=ap)

    def ray_tune(self):
        """
        https://docs.ray.io/en/latest/tune/user-guide.html
        """
        analysis = tune.run(
            self.ray_tune_func, config=params, resources_per_trial={
                "gpu": 0, "cpu": 1}
        )
        analysis.get_best_results(metrics="average_precisions", model="max")
        df = analysis.results_df
        df.to_csv("./Logs.csv")

# TODO : Fix SSIM
# TODO : Check Other Models
# TODO : YoloV1
# TODO : YoloV2
# TODO : YoloV3
# TODO : YoloV4
# TODO : YoloV5
# TODO : OpenCV
