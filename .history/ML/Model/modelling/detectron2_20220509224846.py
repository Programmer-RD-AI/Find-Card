from Model import *
from Model.metrics import Metrics
from PIL import Image

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


class Detectron2:
    """
    This class helps anyone to train a detectron2 model for this project easily so anyone can train this model.
    """

    def __init__(
        self,
        base_lr: float = 0.00025,
        labels: list = None,
        max_iter: int = 250,
        eval_period: int = 250,
        ims_per_batch: int = 2,
        batch_size_per_image: int = 128,
        score_thresh_test: float = 0.625,
        model: str = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        name: str = "baseline",
        create_target_and_preds: int = 29,
        test_sample_size=625,
        data=pd.read_csv(
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/save/Data.csv"
        ),
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
        self.metrics = Metrics()
        if labels is None:
            labels = ["Find-Card"]
        self.remove_files_in_output()
        self.data = data
        # self.data_other = pd.read_csv(
        # "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/save/Data.csv"
        # )
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
            DatasetCatalog.register("data", self.load_data)  # Registering the training data
            MetadataCatalog.get("data").set(thing_classes=self.labels)  # Adding the labels
            self.metadata = MetadataCatalog.get("data")  # Getting the metadata
            DatasetCatalog.register(
                "test", lambda: self.load_data(test=True)
            )  # Registering the test data
            MetadataCatalog.get("test").set(thing_classes=self.labels)  # Adding the labels
            self.metadata_test = MetadataCatalog.get("test")  # Getting the metadata
        except Exception as e:
            self.metadata = MetadataCatalog.get("data")  # Getting the metadata
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
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/output/"
        )  # Get the files in the directory
        try:
            files_to_remove.remove("test_coco_format.json")
        except Exception as e:
            print(e)
        for file_to_remove in tqdm(files_to_remove):  # Iter over the files in the directory
            os.remove(
                f"/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/output/{file_to_remove}"
            )  # Delete the iter file

    def test(self, data_idx: int = 50) -> list:
        """
        - test - croping and creating a box around the img xmin,ymin, xmax, ymax
        -----------------------------------------------------
        - data_idx - the data index which is needed to be visualized
        """
        info = self.data_other.iloc[data_idx]  # getting the info of the index
        img = cv2.imread(
            f'/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/Img/{info["Path"]}'
        )  # reading the img
        height, width = cv2.imread(
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/Img/"
            + info["Path"]
        ).shape[
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
        roi = img[y : y + h, x : x + w]  # crop the image
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 10)  # draw box around the bbox
        return [img, roi]

    def load_data(self, test: bool = False) -> list:
        """
        - load_data - loading the data in the detectron2 data format
        -------------------------------------
        - test - if the return is supposed to be a test sample or not
            - Defalt = False and type = bool
        """
        if test is True and "data.npy" in os.listdir("./Model/save/"):
            self.data = np.load(
                "./Model/save/data.npy", allow_pickle=True
            )  # Loading already saved detectron2 format file
            self.data = self.data[: self.test_sample_size]
            return self.data
        # if "data.npy" in os.listdir("./Model/save/"):
        #     self.data = np.load("./Model/save/data.npy", allow_pickle=True)
        #     return self.data
        new_data = []
        pil_issues_idx = 0
        data_iter = tqdm(range(len(self.data)))
        for idx in data_iter:  # iter over the data
            try:
                data_iter.set_description(f"{pil_issues_idx}/{idx}")
                record = {}
                info = self.data.iloc[idx]
                Image.open(
                    "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/Img/"
                    + info["Path"]
                )
                height, width = cv2.imread(
                    "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/Img/"
                    + info["Path"]
                ).shape[:2]
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
                record["file_name"] = (
                    "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/Img/"
                    + info["Path"]
                )
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
            except OSError:
                pil_issues_idx = pil_issues_idx + 1
        print(len(new_data))
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
                files_and_object_val,
                f"./Model/models/{files_and_object_key}-{self.NAME}.pt",
            )  # Save the file in .pt
            torch.save(
                files_and_object_val,
                f"./Model/models/{files_and_object_key}-{self.NAME}.pth",
            )  # Save the file in .pth
        torch.cuda.empty_cache()

    def create_cfg(self) -> CfgNode:
        """
        - create_cfg - create the config of the model
        - other params - https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
        """
        torch.cuda.empty_cache()
        cfg = get_cfg()  # Creating a new cfg
        cfg.merge_from_file(model_zoo.get_config_file(self.model))  # Add the model
        cfg.DATASETS.TRAIN = ("data",)  # adding train DataSet
        cfg.DATASETS.TEST = ()
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)  # Adding the weights
        cfg.SOLVER.MAX_ITER = self.MAX_ITER  # Set Max iter
        cfg.TEST.EVAL_PERIOD = self.EVAL_PERIOD  # Set Eval Period
        cfg.SOLVER.BASE_LR = self.BASE_LR  # Set Base LR
        cfg.SOLVER.STEPS = []  # Set Steps
        cfg.SOLVER.IMS_PER_BATCH = self.IMS_PER_BATCH  # Set IMS_PER_BATCH
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.labels)  # Set len(self.labels)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            self.BATCH_SIZE_PER_IMAGE
        )  # Set Batch_Size_Per_Image
        cfg.OUTPUT_DIR = "./Model/output"
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
        self.cfg.MODEL.WEIGHTS = "./Model/output/model_final.pth"  # The saved weights of the model
        predictor = DefaultPredictor(self.cfg)  # Creating predictor
        torch.cuda.empty_cache()
        return predictor

    # Detectron2

    def evaluation_detectron2(self, predictor):
        metrics_coco = self.create_coco_eval_detectron2(predictor)
        metrics_file = self.metrics_file_to_dict_detectron2()
        test_images = self.predict_test_images_detectron2(predictor)
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
        ) = self.create_target_and_preds_detectron2(predictor)
        metrics = self.metrics.metrics(preds, target)
        return {
            "Metrics": metrics,
            "metrics_coco": metrics_coco,
            "metrics_file": metrics_file,
            "test_images": test_images,
        }

    def create_coco_eval_detectron2(
        self, predictor: DefaultPredictor, metadata: str = "test"
    ) -> dict:
        """
        - create_coco_eval_detectron2 - create COCO Evaluator and tests it
        -------------------------------
        - predictor - to create the evaluator
        """
        torch.cuda.empty_cache()
        evaluator = COCOEvaluator(metadata, output_dir="./Model/output/")  # Create evaluator
        val_loader = build_detection_test_loader(self.cfg, metadata)  # Create data loader
        metrics = inference_on_dataset(
            predictor.model, val_loader, evaluator
        )  # Test the data with the evaluator
        torch.cuda.empty_cache()
        return metrics

    @staticmethod
    def metrics_file_to_dict_detectron2() -> list:
        """
        - metrics_file_to_dict_detectron2 - in ./output/metrics.json it logs the metrics of the model
        """
        new_logs = []
        try:
            logs = open("./Model/output/metrics.json", "r").read().split("\n")
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

    def predict_test_images_detectron2(self, predictor: DefaultPredictor) -> list:
        """
        - predict_test_images_detectron2 - predict test images
        """
        imgs = []
        torch.cuda.empty_cache()
        for img in tqdm(
            os.listdir(
                "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/test_images/"
            )[:5]
        ):  # iterate over the test images
            v = Visualizer(
                cv2.imread(
                    f"/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/test_images/{img}"
                )[:, :, ::-1],
                metadata=self.metadata,
            )
            v = v.draw_instance_predictions(
                predictor(
                    cv2.imread(
                        f"/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/test_images/{img}"
                    )
                )["instances"].to("cpu")
            )  # Draw pred boxes
            v = v.get_image()[:, :, ::-1]
            plt.figure(figsize=(24, 12))
            plt.imshow(v)  # plot the image
            plt.savefig(
                f"/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/preds/{img}"
            )
            plt.close()
            imgs.append(
                [
                    f"/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/test_images/{img}",
                    v,
                ]
            )
        torch.cuda.empty_cache()
        return imgs

    def create_target_and_preds_detectron2(self, predictor: DefaultPredictor) -> tuple:
        """
        - create_target_and_preds_detectron2 - create the target and predictions
        """
        info = self.data.iloc[self.create_target_and_preds_iter]
        img = cv2.imread(
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/Img/"
            + info["Path"]
        )
        height, width = cv2.imread(
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/ML/Model/dataset/Img/"
            + info["Path"]
        ).shape[:2]
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
        if len(preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"]) <= 0:
            print(preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"])
            preds["instances"].__dict__["_fields"]["pred_boxes"].__dict__["tensor"] = torch.tensor(
                [[1, 1, 1, 1]]
            )
        target = torch.tensor([xmin, ymin, xmax, ymax])
        return (preds, target, x, y, w, h, xmin, ymin, xmax, ymax, height, width)

    def train(self, PROJECT_NAME, param) -> dict:
        """
        - train - trains the model
        """
        torch.cuda.empty_cache()
        wandb.init(
            project=PROJECT_NAME,
            name=str(self.NAME),
            sync_tensorboard=True,
            config=param,
        )
        trainer = self.__train()
        predictor = self.create_predictor()
        metrics = self.evaluation_detectron2(predictor)
        wandb.log(metrics["metrics_coco"])
        for metric_file in metrics["metrics_file"]:
            wandb.log(metric_file)
        for test_img in metrics["test_images"]:
            wandb.log({test_img[0]: wandb.Image(cv2.imread(test_img[0]))})
        wandb.log(metrics["Metrics"])
        try:
            self.save(
                metrics_coco=metrics["metrics_coco"],
                metrics_file=metrics["metrics_file"],
                metrics=metrics["Metrics"],
            )
        except Exception as e:
            print(e)
        wandb.finish()
        return {
            "trainer": trainer,
            "predictor": predictor,
            "metrics_coco": metrics["metrics_coco"],
            "metrics_file": metrics["metrics_file"],
            # "test_images": metrics["test_images"],
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
