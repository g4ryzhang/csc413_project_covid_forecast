import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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
        for i in range(mark - window_size - 1):     
   
            X.append(data[i:window_size+i, 0].reshape(-1, 1))   # Each example is (window_size * n_features)
            # y.append(data[i+1:window_size+i+1, 0])  # Each target is (1 * window_size)
            y.append(data[window_size+i+1, 0])      # Each target is scaler

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

        # Input to network have shape (batch_size, seq_len, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Randomly initialized hidden states
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device) 
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device) 
        
        out, _ = self.gru(x, h0)    # shape = (batch_size, seq_length, hidden_size)
        #out = self.dense(out)               # Output of all unit fed into dense layer
        out = out[:, -1, :]         # Output of the last GRU cell
        out = self.dense(out)
        out = self.relu(out)

        # dense_out = []
        # for i in range(seq_len):
        #     d = self.dense(out[:, i, :])     # output of each gru unit fed into dense layey
        #     dense_out.append(d)   
        # out = torch.hstack(dense_out)


        return out


def train(model, train_loader, val_loader, num_epochs, learning_rate, device):
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    training_losses = []
    val_losses = []
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (sample, target) in enumerate(train_loader):  
            # if i == 0:
            #     print(sample[0])
            #     print(target[0])
            sample = sample.to(device)
            target = target.to(device)

            # Forward pass
            outputs = model(sample)

            if epoch == num_epochs - 1:
                # print(sample)
                print(outputs.squeeze())
                print(target)

            # loss = torch.sqrt(criterion(outputs.squeeze(), target))   # RMSE loss
            loss = criterion(outputs.squeeze(), target)     # MSE loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())

            val_loss = validation_loss(model, val_loader, device, epoch)
            val_losses.append(val_loss)

            if epoch % 100 == 0:
                # print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], '
                #         f'Training Loss: {loss.item():.4f}')
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], '
                        f'Training Loss: {loss.item():.4f}, Valid Loss: {val_loss:.4f}')

    plt.figure()
    plt.plot(np.arange(len(training_losses))[2500:], training_losses[2500:], color='b')
    plt.plot(np.arange(len(training_losses))[2500:], val_losses[2500:], color='r')

    plt.show()


def validation_loss(model, val_loader, device, epoch):
    # Calculate validation loss
    criterion = nn.MSELoss()
    losses = []             # RMSE across all model outputs for each example
    # next_day_pred = []
    # next_day_target = []

    with torch.no_grad():
        for sample, target in val_loader:
            sample = sample.to(device)
            target = target.to(device)          # shape (val_set_size, seq_len)
            prediction = model(sample)          # shape (val_set_size, seq_len, 1)
            # prediction = prediction.squeeze(2)  

            # loss = torch.sqrt(criterion(prediction, target))
            rounded_pred = torch.round(prediction.squeeze())
            loss = criterion(rounded_pred, target)

            if epoch % 2000 == 0:
                print(rounded_pred)
                print(target)

            # losses.append(loss.item())

            # print(prediction)
            # print(target)
            # next_day_pred.append(prediction[0, -1].item())
            # next_day_target.append(target[-1].item())

            # next_day_pred.append(prediction[0, 0].item())
            # next_day_target.append(target[0].item())

    # print(f'Validation set prediction loss across all (seq_len) outputs for each example (RMSE):')
    # for i in losses:
    #     print(i)

    # mse_next_day = mean_squared_error(next_day_pred, next_day_target, squared=True)
    # print(f'\nOverall MSE loss of all future predictions (not including previous days):')
    # print(f'{mse_next_day}\n')

    return loss.item()

    # stack = np.hstack((np.array(next_day_pred).reshape(-1, 1), np.array(next_day_target).reshape(-1, 1)))
    # print(stack)

    # plt.figure()
    # plt.plot(np.arange(len(next_day_pred)), next_day_pred, color='r')
    # plt.plot(np.arange(len(next_day_pred)), next_day_target, color='b')
    # plt.show()


    # for name, x in enumerate(model.named_parameters()):
    #     print(name, x)
    

if __name__ == '__main__':
    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters 
    num_epochs = 15000      # 8000 optimal (?)
    batch_size = 245        # 245 (full batch)
    learning_rate = 0.001   # 0.001
    input_size = 1          # 1     # num_features (fixed)
    seq_len = 11             # 7        # window_size
    hidden_size = 512       # 512
    num_layers = 2


    # Load data
    training_set = TimeSeriesDataset('covid19_sg_clean_reduced.csv', seq_len, train=True)
    val_set = TimeSeriesDataset('covid19_sg_clean_reduced.csv', seq_len, train=False)
    train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=200, shuffle=True)

    model = GRU(input_size, hidden_size, num_layers).to(device)

    train(model, train_loader, val_loader, num_epochs, learning_rate, device)