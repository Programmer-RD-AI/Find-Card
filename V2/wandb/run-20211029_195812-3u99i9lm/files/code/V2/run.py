from Model import *

# model = Model()
# model.train()
params = {
    "MODEL": [
        # "fast_rcnn_R_50_FPN_1x.yaml",
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
    "BASE_LR": [0.01, 0.001, 0.0001, 0.00001, 0.000001],
    "IMS_PER_BATCH": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "BATCH_SIZE_PER_IMAGE": [8, 16, 32, 64, 128, 256, 512],
}
params = ParameterGrid(params)
params_iter = tqdm(params)
for param in params_iter:
    print(param)
    params_iter.set_description(str(param))
    model = Model(
        ims_per_batch=param["IMS_PER_BATCH"],
        batch_size_per_image=param["BATCH_SIZE_PER_IMAGE"],
        model="COCO-Detection/" + param["MODEL"],
        base_lr=param["BASE_LR"],
        name=str(param),
    )
    metrics = model.train()
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