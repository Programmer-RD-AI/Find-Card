# Find-Card

Find-Card

[DataSet](https://storage.googleapis.com/openimages/web/download.html#attributes)

[Logs](https://wandb.ai/ranuga-d/Find-Card)

`labels = ["Debit card","Credit card","Business card",]`


`
params = {
    "MODEL": [
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
    "BASE_LR": [0.0001, 0.00001, 0.000001],
    "IMS_PER_BATCH": [1, 2, 3, 4, 5],
    "BATCH_SIZE_PER_IMAGE": [8, 16, 32, 64, 128],
}
`

[SSIM and PSNR Implemenation](https://github.com/ahrooran-r/image_error_calculation)
