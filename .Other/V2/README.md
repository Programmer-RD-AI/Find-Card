# V2

[DataSet](https://storage.googleapis.com/openimages/web/download.html#attributes)

[Logs](https://wandb.ai/ranuga-d/Find-Card)

In this folder I am going to use detectron2

So this would be better than V1

`BASE_LR=0.0001`

`EVAL_PERIOD=5000`

`MAX_ITER=3250`

`MODEL=COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml`

`IMS_PER_BATCH=1`

`BATCH_SIZE_PER_IMAGE=32`

`SCORE_THRESH_TEST=0.625`

For parameter tunning all of the models and base_lrs and other metrics it will take around 24 Hours.

I am still waiting for a long time to train the model.

## How to train the model

`from Model import *`
`model = Model()`
`model.train()`

Do

`pip3 freeze > requirments.txt`
then
`python3 run.py`

After that you can run the final model if you want to run parameter tunning

`params = { "MODEL": [ "fast_rcnn_R_50_FPN_1x.yaml", "faster_rcnn_R_50_C4_1x.yaml", "faster_rcnn_R_50_C4_3x.yaml", "faster_rcnn_R_50_DC5_1x.yaml", "faster_rcnn_R_50_DC5_3x.yaml", "retinanet_R_50_FPN_1x.py", "retinanet_R_50_FPN_1x.yaml", "retinanet_R_50_FPN_3x.yaml", "rpn_R_50_C4_1x.yaml", "rpn_R_50_FPN_1x.yaml", "faster_rcnn_R_50_FPN_1x.yaml", "faster_rcnn_R_50_FPN_3x.yaml", "faster_rcnn_R_101_DC5_3x.yaml", "faster_rcnn_R_101_FPN_3x.yaml", "faster_rcnn_X_101_32x8d_FPN_3x.yaml", ], "BASE_LR": [0.0001, 0.00001, 0.000001], "IMS_PER_BATCH": [1, 2, 3, 4, 5], "BATCH_SIZE_PER_IMAGE": [8, 16, 32, 64, 128], }`

[SSIM and PSNR Implemenation](https://github.com/ahrooran-r/image_error_calculation)
