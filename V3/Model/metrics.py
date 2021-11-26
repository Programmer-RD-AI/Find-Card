# Imports
import random
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ast
from torchmetrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    Precision,
    Recall,
    SSIM,
    PSNR,
)
import torch
try:
    from ray import tune
except:
    raise ImportError("Cant import ray from tune :(")
try:
    from tqdm import tqdm
except:
    raise ImportError("Cant import tqdm from tqdm :(")


class Metrics:
    def __init__(
        self,
        init_rmse=0,
        init_recall=0,
        init_ious=[],
        init_mse=0,
        init_ssim=0,
        init_psnr=0,
        init_mae=0,
        init_precision=0
    ) -> None:
        """
        Initialize
        """
        # RMSE
        try:
            self.lowest_rmse = init_rmse
            self.r_mean_squared_error = MeanSquaredError(squared=False)
        except Exception as e:
            raise ValueError(f"RMSE Not working {e}")
        # Recall
        try:
            self.lowest_recall = init_recall
            self.recall = Recall()
        except Exception as e:
            raise ValueError(f"RMSE Not working {e}")
        # IOU
        try:
            self.ious = init_ious
        except Exception as e:
            raise ValueError(f"RMSE Not working {e}")
        # MSE
        try:
            self.lowest_mse = init_mse
            self.mean_squared_error = MeanSquaredError(squared=True)
        except Exception as e:
            raise ValueError(f"RMSE Not working {e}")
        # SSIM
        try:
            self.lowest_ssim = init_ssim
            self.ssim = SSIM()
        except Exception as e:
            raise ValueError(f"RMSE Not working {e}")
        # PSNR
        try:
            self.lowest_psnr = init_psnr
            self.psnr = PSNR()
        except Exception as e:
            raise ValueError(f"RMSE Not working {e}")
        # MAE
        try:
            self.lowest_mae = init_mae
            self.mae = MeanAbsoluteError()
        except Exception as e:
            raise ValueError(f"RMSE Not working {e}")
        # Precision
        try:
            self.lowest_precision = init_precision
            self.precision = Precision()
        except Exception as e:
            raise ValueError(f"RMSE Not working {e}")

    def create_rmse(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_rmse - Create Root-mean-square deviation
        """
        try:
            preds_new = (
                preds["instances"].__dict__[
                    "_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                if self.r_mean_squared_error(pred.to("cpu"), target) > self.lowest_rmse:
                    self.lowest_rmse = self.r_mean_squared_error(
                        pred.to("cpu"), target)
            return float(self.lowest_rmse)
        except Exception as e:
            raise ValueError(f"Some error occurred in RMSE {e}")

    def create_recall(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_recall - Create Recall
        """
        try:
            preds_new = (
                preds["instances"].__dict__[
                    "_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                if self.recall(pred.to("cpu"), target) > self.lowest_recall:
                    self.lowest_recall = self.recall(pred.to("cpu"), target)
            return float(self.lowest_recall)
        except Exception as e:
            raise ValueError(f"Some error occured in Recall {e}")

    def create_iou(self, preds: torch.tensor, targets: torch.tensor) -> float:
        """
        - create_iou - Create IOU
        """
        try:
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
                self.ious.append(iou)
            iou = np.mean(self.ious)
            return iou
        except Exception as e:
            raise ValueError(f"Some error occured in IOU {e}")

    def create_mse(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_mse - Create Mean-square deviation
        """
        try:
            preds_new = (
                preds["instances"].__dict__[
                    "_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                if self.mean_squared_error(pred.to("cpu"), target) > self.lowest_mse:
                    self.lowest_mse = self.mean_squared_error(
                        pred.to("cpu"), target)
            return float(self.lowest_mse)
        except Exception as e:
            raise ValueError(f"Some error occured in MSE {e}")

    def create_ssim(
        self, preds: torch.tensor, target: torch.tensor, height: int, width: int
    ) -> float:
        """
        - create_ssim - create SSIM # TODO it is not done yet
        """
        try:
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
                crop_img_target = torch.from_numpy(
                    self.crop_img(x, y, w, h, img))
                x, y, w, h = self.create_x_y_w_h(
                    pred[0], pred[1], pred[2], pred[3])
                crop_img_pred = torch.from_numpy(
                    np.array(self.crop_img(x, y, w, h, img)))
                if self.ssim(crop_img_pred, crop_img_target) > self.lowest_ssim:
                    self.lowest_ssim = self.ssim(pred.to("cpu"), target)
            return self.lowest_ssim
        except Exception as e:
            raise ValueError(f"Some error occured in SSIM {e}")

    def create_psnr(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_psnr - Peak signal-to-noise ratio (how similar is a image)
        """
        try:
            preds_new = (
                preds["instances"].__dict__[
                    "_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                if self.psnr(pred.to("cpu"), target) > self.lowest_psnr:
                    self.lowest_psnr = self.psnr(pred.to("cpu"), target)
            return self.lowest_psnr
        except Exception as e:
            raise ValueError(f"Some error occured in PSNR {e}")

    def create_mae(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_mae - Mean absolute error
        """
        try:
            preds_new = (
                preds["instances"].__dict__[
                    "_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                if self.mae(pred.to("cpu"), target) > self.lowest_mae:
                    self.lowest_mae = self.mae(pred.to("cpu"), target)
            return self.lowest_mae
        except Exception as e:
            raise ValueError(f"Some error occured in MAE {e}")

    def create_precision(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_precision - Create Precision
        """
        try:
            preds_new = (
                preds["instances"].__dict__[
                    "_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                if self.precision(pred.to("cpu"), target) > self.lowest_precision:
                    self.lowest_precision = self.precision(
                        pred.to("cpu"), target)
            return self.lowest_precision
        except Exception as e:
            raise ValueError(f"Some error occured in Precision {e}")

    def create_precision_and_recall(
        self, preds: torch.tensor, target: torch.tensor
    ) -> float:
        """
        - create_precision_and_recall - Create Precision and recall
        """
        try:
            recall = self.create_recall(preds, target)
            precision = self.create_precision(preds, target)
            if recall > precision:
                precision_recall = precision - recall
            else:
                precision_recall = recall - precision
            return precision_recall
        except Exception as e:
            raise ValueError(f"Some error occured in Recall and Precision {e}")

    def metrics(
        self, preds: torch.tensor, target: torch.tensor
    ) -> dict:
        """
        - combines all metrics and easily return all of the metrics
        """
        metrics = {
            'RMSE': self.create_rmse(preds, target),
            'Recall': self.create_recall(preds, target),
            'IOU': self.create_iou(preds, target),
            'MSE': self.create_mse(preds, target),
            'SSIM': self.create_ssim(preds, target),
            'PSNR': self.create_psnr(preds, target),
            'MAE': self.create_mae(preds, target),
            'Precision': self.create_precision(preds, target),
            'Precision and Recall': self.create_precision_and_recall(preds, target)
        }
        return metrics
