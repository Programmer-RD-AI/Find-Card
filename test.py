from torchmetrics import MeanSquaredError
import torch
target = torch.tensor([[2.5, 5.0, 4.0, 8.0],[2.5, 5.0, 4.0, 8.0]])
preds = torch.tensor([[3.0, 5.0, 2.5, 7.0],[3.0, 5.0, 2.5, 7.0]])
mean_squared_error = MeanSquaredError()
print(mean_squared_error(preds, target))
