import torch.nn as nn
import torch

class HotelPredModel(nn.Module):
    def __init__(self, features, n_predictions=1):
        super().__init__()
        self.lstm= nn.LSTM(
            input_size=features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.dropout=nn.Dropout(0.2)
        self.fc=nn.Linear(64, n_predictions)

    
    def forward(self, x):
        h0= torch.zeros(2, x.size(0), 64).to(x.device)
        c0= torch.zeros(2, x.size(0), 64).to(x.device)
        out, _=self.lstm(x, (h0, c0))
        out=out[:, -1, :]
        out=self.dropout(out)
        out=self.fc(out)
        return out
