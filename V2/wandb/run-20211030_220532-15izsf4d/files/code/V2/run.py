from Model import *

model = Model()
model.train()
# params = {
#     "MODEL": [
        # "faster_rcnn_X_101_32x8d_FPN_3x.yaml",
#         "faster_rcnn_R_101_C4_3x.yaml",
#         "faster_rcnn_R_50_FPN_3x.yaml",
#         "keypoint_rcnn_R_50_FPN_1x.yaml",
#         "keypoint_rcnn_R_50_FPN_3x.yaml",
#         "keypoint_rcnn_R_101_FPN_3x.py",
#         "keypoint_rcnn_X_101_32x8d_FPN_3x.yaml",
#     ],
#     "BASE_LR": [0.0001, 0.00001, 0.000001],
#     "IMS_PER_BATCH": [
#         1,
#         2,
#         3,
#     ],
#     "BATCH_SIZE_PER_IMAGE": [32, 64, 128],
# }
# params = ParameterGrid(params)
# params_iter = tqdm(params)
# for param in params_iter:
#     try:
#         torch.cuda.empty_cache()
#         model.remove_files_in_output()
#         torch.cuda.empty_cache()
#         params_iter.set_description(str(param))
#         torch.cuda.empty_cache()
#         model = Model(
#             ims_per_batch=param["IMS_PER_BATCH"],
#             batch_size_per_image=param["BATCH_SIZE_PER_IMAGE"],
#             model="COCO-Detection/" + param["MODEL"],
#             base_lr=param["BASE_LR"],
#             name=str(param),
#         )
#         torch.cuda.empty_cache()
#         metrics = model.train()
#         torch.cuda.empty_cache()
#     except Exception as e:
#         print("*" * 50)
#         print(e)
#         print("*" * 50)
#         torch.cuda.empty_cache()
