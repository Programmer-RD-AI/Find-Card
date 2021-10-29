from Model import *

params = {
    "MODEL": [
        "fast_rcnn_R_50_FPN_1x.yaml",
        # "faster_rcnn_R_50_C4_1x.yaml",
        # "faster_rcnn_R_50_C4_3x.yaml",
        # "faster_rcnn_R_50_DC5_1x.yaml",
        # "faster_rcnn_R_50_DC5_3x.yaml",
        # "retinanet_R_50_FPN_1x.py",
        # "retinanet_R_50_FPN_1x.yaml",
        # "retinanet_R_50_FPN_3x.yaml",
        # "rpn_R_50_C4_1x.yaml",
        # "rpn_R_50_FPN_1x.yaml",
        # "faster_rcnn_R_50_FPN_1x.yaml",
        # "faster_rcnn_R_50_FPN_3x.yaml",
        # "faster_rcnn_R_101_DC5_3x.yaml",
        # "faster_rcnn_R_101_FPN_3x.yaml",
        # "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    ],
    "MAX_ITER": [500],
    "LABELS": [["Card"]],
    "CREATE_TARGET_AND_PREDS": [55],
    "EVAL_PERIOD": [500],
    "SCORE_THRESH_TEST": [0.625],
    "BASE_LR": [0.1, 0.01, 0.001, 0.0001],
    "IMS_PER_BATCH": [
        1,
        2,
        3,
        4,
        5,
    ],
    "BATCH_SIZE_PER_IMAGE": [8, 16, 32, 64, 128],
}
model = Model()
model.train()
pt = Param_Tunning()
pt.tune()
# torch.save(model.train(),'./model.pt')
# torch.save(model.train(),'./model.pth')
