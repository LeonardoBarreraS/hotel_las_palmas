from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchmetrics import MeanSquaredError
import torch
import math

def eval_hotel_model(model, dataloader_test):
    mse = MeanSquaredError()
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for features, labels in dataloader_test:
            pred = model(features)
            loss = mse(pred, labels)
            total_loss += loss.item()
            n_batches += 1

    average_mse = total_loss / n_batches
    rmse = math.sqrt(average_mse)  # Root Mean Squared Error
    print(f"RSE en validación: {rmse:.4f}")
    print(f"# of batches: {n_batches}")
    print("Len dataloader: ", len(dataloader_test))
    return rmse

def eval_hotel_model_perday(model, dataloader_test):

    model.eval()
    rmse_per_day = []

    with torch.no_grad():
        all_preds = []
        all_targets = []

        for features, labels in dataloader_test:
            pred = model(features)
            all_preds.append(pred)
            all_targets.append(labels)

        preds = torch.cat(all_preds, dim=0)   
        print(preds.shape)    # [N, 30]
        targets = torch.cat(all_targets, dim=0)
        print(targets.shape)   # [N, 30]
        
        # Error cuadrático por día
        squared_error = (preds - targets) ** 2    # [N, 30]
        rmse_per_day= torch.sqrt(squared_error.mean(dim=0))  # [30]


    for i, rmse_day in enumerate(rmse_per_day):
        print(f"Día {i+1}: MSE = {rmse_day.item():.4f}")
    return rmse_per_day