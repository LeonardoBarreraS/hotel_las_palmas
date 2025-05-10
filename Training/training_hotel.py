import torch
import torch.nn as nn
import torch.optim as optim
from Training import *
import math

import mlflow

def train_model_hotelpred(model, train_dataloader, epochs=800, lr=0.0001):
       criterion=nn.MSELoss()
       optimizer=optim.Adam(model.parameters(), lr=lr)

       device=torch.device("cpu")
       model.to(device)

       mlflow.log_param("num_epochs", epochs)
       mlflow.log_param("learning_rate", lr)

       model.train()
       for epoch in range(epochs):
         total_loss=0.0
         for features, labels in train_dataloader:
              features = features.to(device)
              labels = labels.to(device)

              optimizer.zero_grad()
              output=model(features)
              loss=criterion(output, labels)
              loss.backward()
              optimizer.step()
              total_loss+=loss.item()

         if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
        
       avg_mse=total_loss / len(train_dataloader)
       rmse = math.sqrt(avg_mse)  # Root Mean Squared Error
       print(f"RSE en validación: {rmse:.4f}")
       print("Len dataloader: ", len(train_dataloader))

       return model
