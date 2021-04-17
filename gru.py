import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


# Training and validation Dataset for DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, file_name, window_size, train=True):
        data = pd.read_csv(file_name).values[1:, 1:]    # remove title row and date column
        data = np.array(data, dtype=np.float32)
        data[np.isnan(data)] = 0    # Mask NaN with zero
        # Use data from previous day(s) to predict the confirmed cases of next day
        mark = int(np.round(data.shape[0]*0.8))     # front 80% of dataset

        X = []
        y = []
        for i in range(mark - window_size - 1):     # Each example is (window_size * n_features)
            X.append(data[i:window_size+i, :])
            y.append(data[i+1:window_size+i+1, 0])  # Each target is (1 * window_size)
            # y.append(data[window_size+i+1, 0])      # Each target is scaler

        split = int(np.round(len(y)*0.7))           # Split train and val 7:3
        if train:
            self.X = torch.tensor(X[:split], dtype=torch.float32)
            self.y = torch.tensor(y[:split], dtype=torch.float32)
        else:
            self.X = torch.tensor(X[split:], dtype=torch.float32)
            self.y = torch.tensor(y[split:], dtype=torch.float32)
            # self.y = torch.tensor(y[split:, -1], dtype=torch.float32)   # Target label is cases of the next date

    def __getitem__(self, index):
        return self.X[index], self.y[index] 
    
    def __len__(self):
        return len(self.y)


# GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Input to network are have shape (batch_size, seq, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = x.size(0)

        # Initial hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device) 
        
        out, _ = self.gru(x, h0)    # shape = (batch_size, seq_length, hidden_size)
        out = self.fc(out)               # Output of all cells fed into dense layer
        # out = out[:, -1, :]         # Output of the last GRU cell
        # out = self.fc(out)

        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 

num_epochs = 20     # 20
batch_size = 100    # 100
learning_rate = 0.001

input_size = 17     # num_features (fixed)
seq_len = 7    # 7        # window_size
hidden_size = 128    #110
num_layers = 2


# Load data
training_set = TimeSeriesDataset('covid19_sg_clean.csv', seq_len, train=True)
val_set = TimeSeriesDataset('covid19_sg_clean.csv', seq_len, train=False)
train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)

model = GRU(input_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (sample, target) in enumerate(train_loader):  

        sample = sample.to(device)
        target = target.to(device)

        # Forward pass
        outputs = model(sample)
        loss = torch.sqrt(criterion(outputs.squeeze(), target))   # RMSE loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    losses = []             # RMSE across all model outputs for each example
    next_day_pred = []
    next_day_target = []
    for sample, target in val_loader:
        sample = sample.to(device)
        target = target.to(device)          # shape (1, seq_len)
        prediction = model(sample)
        prediction = prediction.squeeze(2)  # shape (1, seq_len)

        loss = torch.sqrt(criterion(prediction, target))
        losses.append(loss.item())

        next_day_pred.append(prediction[0, -1].item())
        next_day_target.append(target[0, -1].item())

print(f'Validation set prediction loss across all (seq_len) outputs for each example (RMSE):')
for i in losses:
    print(i)

rmse_next_day = mean_squared_error(next_day_pred, next_day_target, squared=False)
print(f'\nOverall RMSE loss of all future predictions (not including previous days):')
print(f'{rmse_next_day}\n\n')
    