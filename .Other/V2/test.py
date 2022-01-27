from collections import namedtuple
import cv2

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])


def bb_intersection_over_union(true_box, pred_box):
    xA = max(true_box[0], pred_box[0])
    yA = max(true_box[1], pred_box[1])
    xB = min(true_box[2], pred_box[2])
    yB = min(true_box[3], pred_box[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
    boxBArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


examples = [
    Detection("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
    Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
    Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
    Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
    Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108]),
]
for detection in examples:
    image = cv2.imread(detection.image_path)
    iou = bb_intersection_over_union(detection.gt, detection.pred)
