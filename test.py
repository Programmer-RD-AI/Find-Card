# wandb.init(sync_tensorboard=True, name="baseline")
# torch.cuda.empty_cache()
# model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
# cfg = get_cfg()
# cfg.merge_from_file(get_config_file(model))
# cfg.DATASETS.TRAIN = ("data",)
# cfg.DATASETS.TEST = ("test",)
# cfg.MODEL.WEIGHTS = get_checkpoint_url(model)
# cfg.SOLVER.MAX_ITER = 500
# cfg.TEST.EVAL_PERIOD = 100
# cfg.SOLVER.BASE_LR = 0.00025
# cfg.SOLVER.STEPS = []
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
# trainer.test(model=cfg)
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


# for img in os.listdir("./test_imgs/"):
#     predictor = DefaultPredictor(cfg)
#     img = cv2.imread(f"./test_imgs/{img}")
#     v = Visualizer(img[:, :, ::-1], metadata=metadata)
#     v = v.draw_instance_predictions(predictor(img)["instances"].to("cpu"))
#     # plt.figure(figsize=(10, 7))
#     # plt.imshow(v.get_image()[:, :, ::-1])
#     # plt.show()
#     wandb.log({f"Img/{img}": wandb.Image(v.get_image()[:, :, ::-1])})
# wandb.finish()
