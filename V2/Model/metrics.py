from Model import *


class Metrics:
    def __init__(self):
        self.lowest_rmse = 0
        self.r_mean_squared_error = MeanSquaredError(squared=False)
        self.lowest_recall = 0
        self.recall = Recall()
        self.ious = []
        self.lowest_mse = 0
        self.mean_squared_error = MeanSquaredError(squared=True)
        self.lowest_ssim = 0
        self.ssim = SSIM()
        self.lowest_psnr = 0
        self.psnr = PSNR()
        self.lowest_mae = 0
        self.mae = MeanAbsoluteError()
        self.lowest_precision = 0
        self.precision = Precision()

    def create_rmse(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_rmse - Create Root-mean-square deviation
        """

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

    def create_recall(self, preds: torch.tensor, target: torch.tensor) -> float:

        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if self.recall(pred.to("cpu"), target) > self.lowest_recall:
                self.lowest_recall = self.recall(pred.to("cpu"), target)
        return float(self.lowest_recall)

    def create_iou(self, preds: torch.tensor, targets: torch.tensor) -> float:

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

    def create_mse(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_mse - Create Mean-square deviation
        """

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

    def create_ssim(
        self, preds: torch.tensor, target: torch.tensor, height: int, width: int
    ) -> float:
        """
        - create_ssim - create SSIM # TODO it is not done yet
        """

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
            if self.ssim(crop_img_pred, crop_img_target) > self.lowest_ssim:
                self.lowest_ssim = self.ssim(pred.to("cpu"), target)
        return self.lowest_ssim

    def create_psnr(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_psnr - Peak signal-to-noise ratio (how similar is a image)
        """

        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if self.psnr(pred.to("cpu"), target) > self.lowest_psnr:
                self.lowest_psnr = self.psnr(pred.to("cpu"), target)
        return self.lowest_psnr

    def create_mae(self, preds: torch.tensor, target: torch.tensor) -> float:
        """
        - create_mae - Mean absolute error
        """

        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if self.mae(pred.to("cpu"), target) > self.lowest_mae:
                self.lowest_mae = self.mae(pred.to("cpu"), target)
        return self.lowest_mae

    def create_precision(self, preds: torch.tensor, target: torch.tensor) -> float:
        preds_new = (
            preds["instances"].__dict__[
                "_fields"]["pred_boxes"].__dict__["tensor"]
        )
        for pred_i in tqdm(range(len(preds))):
            pred = preds_new[pred_i]
            if self.precision(pred.to("cpu"), target) > self.lowest_precision:
                self.lowest_precision = self.precision(pred.to("cpu"), target)
        return self.lowest_precision

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
