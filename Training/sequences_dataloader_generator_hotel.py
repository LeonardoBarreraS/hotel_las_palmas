#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

#%%

def generate_dataloader(df_x, df_y, features_list, sequence_length, forecast_horizon):

    X, y= [],[]

    for i in range(len(df_x)-sequence_length-forecast_horizon+1):
        X.append(df_x.iloc[i:i+sequence_length][features_list].values)
        y.append(df_y.iloc[i+sequence_length:i+sequence_length+forecast_horizon].values)

    X=np.array(X)
    y=np.array(y)

    X_tensor=torch.tensor(X, dtype=torch.float32)
    y_tensor=torch.tensor(y, dtype=torch.float32)
    dataset_hotel = TensorDataset(X_tensor, y_tensor)
    dataloader=DataLoader(dataset_hotel, batch_size=32, shuffle=True)

    return dataloader

