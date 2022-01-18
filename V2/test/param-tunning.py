# import random






wandb.init(entity='find-card', project="Find-Card", name=NAME, config={
    'BASE_LR': BASE_LR,
    'MAX_ITER': MAX_ITER,
    'EVAL_PERIOD': EVAL_PERIOD,
    'IMS_PER_BATCH': IMS_PER_BATCH,
    'BATCH_SIZE_PER_IMAGE': BATCH_SIZE_PER_IMAGE,
    'SCORE_THRESH_TEST': SCORE_THRESH_TEST,
    'MODEL': model,
    'NAME': NAME
})
lowest_mse = 0
mean_squared_error = MeanSquaredError(squared=True)
preds_new = preds["instances"].__dict__[
    "_fields"]["pred_boxes"].__dict__["tensor"]
for pred_i in range(len(preds)):
    pred = preds_new[pred_i]
    if mean_squared_error(pred.to("cpu"), target) > lowest_mse:
        lowest_mse = mean_squared_error(pred.to("cpu"), target)
wandb.log({"MSE": lowest_mse})
wandb.log({"RMSE": lowest_rmse})
wandb.finish()
