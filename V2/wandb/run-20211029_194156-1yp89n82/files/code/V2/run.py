from Model import *

model = Model()
model.train()
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
params = ParameterGrid(params)
for param in tqdm(params):
    print(param)
    # model = Model(
    # base_lr=param["BASE_LR"],
    # labels=param["LABELS"],
    # max_iter=param["MAX_ITER"],
    # eval_period=param["EVAL_PERIOD"],
    # ims_per_batch=param["IMS_PER_BATCH"],
    # batch_size_per_image=param["BATCH_SIZE_PER_IMAGE"],
    # score_thresh_test=param["SCORE_THRESH_TEST"],
    # model="COCO-Detection/" + param["MODEL"],
    # name=str(param),
    # create_target_and_preds=param["CREATE_TARGET_AND_PREDS"],
    # )
    # metrics = model.train()
# models = [
#     # "fast_rcnn_R_50_FPN_1x.yaml",
#     "faster_rcnn_R_50_C4_1x.yaml",
#     "faster_rcnn_R_50_C4_3x.yaml",
#     "faster_rcnn_R_50_DC5_1x.yaml",
#     "faster_rcnn_R_50_DC5_3x.yaml",
#     "retinanet_R_50_FPN_1x.py",
#     "retinanet_R_50_FPN_1x.yaml",
#     "retinanet_R_50_FPN_3x.yaml",
#     "rpn_R_50_C4_1x.yaml",
#     "rpn_R_50_FPN_1x.yaml",
#     "faster_rcnn_R_50_FPN_1x.yaml",
#     "faster_rcnn_R_50_FPN_3x.yaml",
#     "faster_rcnn_R_101_DC5_3x.yaml",
#     "faster_rcnn_R_101_FPN_3x.yaml",
#     "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
# ]
# max_iters = 125
# labels = ["Card"]
# create_target_and_preds = 55
# eval_period = 125
# score_thresh_test = 0.625
# base_lrs = [0.1, 0.01, 0.001, 0.0001]
# ims_per_batchs = [1, 2, 3]
# batch_size_per_images = []
# for model in models:
#     model = Model(model=f"COCO-Detection/{model}", name=model)
#     model.train()
# pt = Param_Tunning(params)
# pt.tune()
# torch.save(model.train(),'./model.pt')
# torch.save(model.train(),'./model.pth')
