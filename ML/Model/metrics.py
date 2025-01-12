from Model import *


class Metrics:
    """sumary_line

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def __init__(
        self,
        init_ious=None,
        init_ssim=0,
        init_psnr=0,
    ) -> None:
        """Initialize"""
        if init_ious is None:
            init_ious = []
        # IOU
        try:
            self.ious = init_ious
        except Exception as e:
            raise ValueError(e)
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

    def create_iou(self, preds: torch.tensor, targets: torch.tensor) -> float:
        """- create_iou - Create IOU"""
        try:
            preds = (
                preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
            )
            pred_box, true_box = preds[0].to("cpu"), targets.to("cpu")
            xA = max(true_box[0], pred_box[0])
            yA = max(true_box[1], pred_box[1])
            xB = min(true_box[2], pred_box[2])
            yB = min(true_box[3], pred_box[3])
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
            boxBArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            self.ious.append(iou)
            iou = np.mean(self.ious)
            return iou
        except Exception as e:
            raise ValueError(f"Some error occured in IOU {e}")

    def create_ssim_1(
        self,
        preds: torch.tensor,
        target: torch.tensor,
    ) -> float:
        """- create_ssim - create SSIM # TODO it is not done yet"""
        try:
            preds_new = (
                preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                if self.mean_squared_error(pred.to("cpu"), target) > self.lowest_mse:
                    self.lowest_mse = self.mean_squared_error(pred.to("cpu"), target)
            return float(self.lowest_mse)
        except Exception as e:
            raise ValueError(f"Some error occured in MSE {e}")

    def create_ssim_2(
        self,
        preds: torch.tensor,
        target: torch.tensor,
    ) -> float:
        """- create_ssim - create SSIM # TODO it is not done yet"""
        try:
            preds_new = (
                preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                info = self.data[self.create_target_and_preds_detectron2_iter]
                img = cv2.imread(info["Path"])
                x, y, w, h = self.create_x_y_w_h(
                    target[0], target[1], target[2], target[3]
                )
                crop_img_target = torch.from_numpy(self.crop_img(x, y, w, h, img))
                x, y, w, h = self.create_x_y_w_h(pred[0], pred[1], pred[2], pred[3])
                crop_img_pred = torch.from_numpy(
                    np.array(self.crop_img(x, y, w, h, img))
                )
                if self.ssim(crop_img_pred, crop_img_target) > self.lowest_ssim:
                    self.lowest_ssim = self.ssim(pred.to("cpu"), target)
            return self.lowest_ssim
        except Exception as e:
            raise ValueError(f"Some error occured in SSIM {e}")

    def create_psnr(self, preds: torch.tensor, target: torch.tensor) -> float:
        """- create_psnr - Peak signal-to-noise ratio (how similar is a image)"""
        try:
            preds_new = (
                preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
            )
            for pred_i in tqdm(range(len(preds))):
                pred = preds_new[pred_i]
                if self.psnr(pred.to("cpu"), target) > self.lowest_psnr:
                    self.lowest_psnr = self.psnr(pred.to("cpu"), target)
            return self.lowest_psnr
        except Exception as e:
            raise ValueError(f"Some error occured in PSNR {e}")

    @staticmethod
    def create_ap(preds: torch.tensor, target: torch.tensor) -> float:
        """- create_ap - Create Average Precision(AP)"""
        try:
            preds = (
                preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]
            )
            preds = preds[0].cpu()
            target = target.cpu()
            ap = AveragePrecision()
            ap = ap(target, preds)
            return ap
        except Exception as e:
            raise ValueError(f"Some error occured in Average Precision {e}")

    def metrics(self, preds: torch.tensor, target: torch.tensor) -> dict:
        """- combines all metrics and easily return all of the metrics"""
        metrics = {
            "IOU": self.create_iou(preds, target),
            "PSNR": self.create_psnr(preds, target),
        }
        return metrics
