from Model import *
model = Model()
model.train()
models = [
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
    ]
max_iters = 125
labels = ['Card']
create_target_and_preds = 55
eval_period = 125
score_thresh_test = 0.625
base_lrs = [0.1,0.01, 0.001, 0.0001]
ims_per_batchs = [1,2,3]
batch_size_per_images = []
for model in models:
    model = Model(model=f"COCO-Detection/{model}",name=model)
    model.train()
# pt = Param_Tunning(params)
# pt.tune()
# torch.save(model.train(),'./model.pt')
# torch.save(model.train(),'./model.pth')
