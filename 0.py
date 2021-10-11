# Imports
import numpy as np
import os, cv2, torch, torchvision
import detectron2
from detectron2.evaluation import COCOEvaluator, inference_context, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.model_zoo import get_checkpoint_url, get_config_file
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer, BoxMode
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# Version
print(torch.__version__, torchvision.__version__)
data = pd.read_csv("./Data.csv")
idx = 0

# Loading Data
def load_data(data=data, test=False):
    if test is True:
        if "data.npy" in os.listdir("./"):
            data = np.load("./data.npy", allow_pickle=True)[:625]
            return data
    if "data.npy" in os.listdir("./"):
        data = np.load("./data.npy", allow_pickle=True)
        return data
    new_data = []
    for idx in tqdm(range(len(data))):
        record = {}
        info = data.iloc[idx]
        xmin, ymin, xmax, ymax = info["XMin"], info["YMin"], info["XMax"], info["YMax"]
        height, width = cv2.imread("./Img/" + info["Path"]).shape[:2]
        record["file_name"] = "./Img/" + info["Path"]
        record["height"] = height
        record["width"] = width
        info["class_id"] = 0
        record["cateogry_id"] = 0
        objs = [
            {
                "bbox": [info["XMin"], info["YMin"], info["XMax"], info["YMax"]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "iscrowd": 0,
                "category_id": 0,
            }
        ]
        record["image_id"] = idx
        record["annotations"] = objs
        record["class_id"] = 0
        record["xmin"] = xmin
        record["ymin"] = ymin
        record["xmax"] = xmax
        record["ymax"] = ymax
        new_data.append(record)
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

torch.cuda.empty_cache()
wandb.init(sync_tensorboard=True, name="baseline")
model = "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"
# Create the model
cfg = get_cfg()
cfg.merge_from_file(get_config_file(model))
cfg.DATASETS.TRAIN = ("data",)
cfg.DATASETS.TEST = ()
cfg.MODEL.WEIGHTS = get_checkpoint_url(model)
cfg.SOLVER.MAX_ITER = 500
cfg.TEST.EVAL_PERIOD = 50
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.STEPS = []
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# Create trainer
trainer = DefaultTrainer(cfg)
wandb.log(trainer.test(cfg, model))
trainer.test(cfg, model)
trainer.resume_or_load(resume=False)
trainer.train()
# Eval The Model
evaluator = COCOEvaluator("test", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
wandb.log(trainer.test(cfg, model))
trainer.test(cfg, model)
evaluator = COCOEvaluator("data", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "data")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# Predict
for img in os.listdir("./test_imgs/"):
    predictor = DefaultPredictor(cfg)
    img = cv2.imread(f"./test_imgs/{img}")
    v = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=0.5,
    )
    v = v.draw_instance_predictions(predictor(img)["instances"].to("cpu"))
    plt.figure(figsize=(10, 7))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()
    wandb.log({f"Img/{img}": wandb.Image(v.get_image()[:, :, ::-1])})
wandb.finish()
