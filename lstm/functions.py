import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Dataset, random_split
import time
import sys, os
import gc
sys.path.append(os.pardir)



# # PyTorch 데이터셋 정의
# class TimeSeriesDataset(Dataset):
#     def __init__(self, X, y, seq_length):
#         self.X = X
#         self.y = y
#         self.seq_length = seq_length

#     def __len__(self):
#         return len(self.X) - self.seq_length

#     def __getitem__(self, idx):
#         x = self.X.iloc[idx:idx+self.seq_length].values
#         y = self.y.iloc[idx+self.seq_length]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(1)



class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if idx + self.seq_length <= len(self.X):
            x = self.X.iloc[idx:idx+self.seq_length].values
            y = self.y.iloc[idx+self.seq_length - 1]
        else:
            x = self.X.iloc[idx:].values
            x = np.pad(x, ((0, self.seq_length - len(x)), (0, 0)), mode='edge')
            y = self.y.iloc[-1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(1)





# 학습 함수 정의
def train_model_cv(model, dataset, batch_size, num_workers, criterion, optimizer, scheduler, num_epochs, device, seq_len):
    # big_epoch 결정기
    if num_epochs > 10:
        big_epoch = True
    else:
        big_epoch = False

    if big_epoch:
        save_dir = "C:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest\\model_param\\big_epoch_cv"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_interval = 5
        save_paths = []
    train_loss_history = []
    val_loss_history = []
    scaler = amp.GradScaler()

    # 데이터 로더 생성
    X, y = dataset
    dataset = TimeSeriesDataset(X, y, seq_len)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        val_loss = 0.0

        print(f"Starting epoch {epoch+1}/{num_epochs}")

        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

            if big_epoch and i % 1600 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Train Loss: {loss.item()}")
            elif not big_epoch and i % 800 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Train Loss: {loss.item()}")
        
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)

                if big_epoch and i % 400 == 0:
                    print(f"Epoch {epoch+1}, Batch {i}, Validation Loss: {loss.item()}")
                elif not big_epoch and i % 200 == 0:
                    print(f"Epoch {epoch+1}, Batch {i}, Validation Loss: {loss.item()}")

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if big_epoch and (epoch+1) % save_interval == 0:
            save_path = os.path.join(save_dir, f'lstm_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1}")
            save_paths.append(save_path)

        # scheduler.step()  # Update learning rate

        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")

    return train_loss_history, val_loss_history, save_paths










# 학습 함수 정의
def train_model_all(model, dataset, batch_size, num_workers, criterion, optimizer, scheduler, num_epochs, device, seq_len):
    # big_epoch 결정기
    if num_epochs > 10:
        big_epoch = True
    else:
        big_epoch = False

    if big_epoch:
        save_dir = "C:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest\\model_param\\big_epoch_all"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_interval = 5
        save_paths = []
    train_loss_history = []
    # val_loss_history = []
    scaler = amp.GradScaler()

    # 데이터 로더 생성
    X, y = dataset
    train_dataset = TimeSeriesDataset(X, y, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        # val_loss = 0.0

        print(f"Starting epoch {epoch+1}/{num_epochs}")

        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

            if big_epoch and i % 2000 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Train Loss: {loss.item()}")
            elif not big_epoch and i % 1000 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Train Loss: {loss.item()}")


        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        if big_epoch and (epoch+1) % save_interval == 0:
            save_path = os.path.join(save_dir, f'lstm_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1}")
            save_paths.append(save_path)

        # scheduler.step()  # Update learning rate

        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s")

    return train_loss_history, save_paths






# 평가 함수 정의
def eval_model(model, dataset, batch_size, num_workers, device, seq_len):
    # 텐서 초기화
    y_true_all = torch.tensor([], dtype=torch.float32, device=device)
    y_pred_all = torch.tensor([], dtype=torch.float32, device=device)

    # 데이터 로더 생성
    X, y = dataset
    test_dataset = TimeSeriesDataset(X, y, seq_len)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
 
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with amp.autocast():
                outputs = model(inputs)
            
            y_true_all = torch.cat((y_true_all, targets), dim=0)
            y_pred_all = torch.cat((y_pred_all, outputs), dim=0)


    # 전체 데이터셋에 대한 MSE와 MAE 계산
    y_true_all_cpu = y_true_all.cpu().numpy()
    y_pred_all_cpu = y_pred_all.cpu().numpy()
    mse = mean_squared_error(y_true_all_cpu, y_pred_all_cpu)
    mae = mean_absolute_error(y_true_all_cpu, y_pred_all_cpu)
    print(f"LSTM Model Test MSE: {mse}")
    print(f"LSTM Model Test MAE: {mae}")

    # 최종 반환 시 리스트로 변환
    y_true_all_list = y_true_all_cpu.tolist()
    y_pred_all_list = y_pred_all_cpu.tolist()


    return mse, mae, y_true_all_list, y_pred_all_list



def pred_model(model, dataset, batch_size, num_workers, device, seq_len):
    # 텐서 초기화
    y_pred_all = torch.tensor([], dtype=torch.float32, device=device)

    # 데이터 로더 생성
    X, y = dataset
    test_dataset = TimeSeriesDataset(X, y, seq_len)
    print(f"Length of test_dataset: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Length of test_loader: {len(test_loader.dataset)}")
 
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with amp.autocast():
                # outputs = model(inputs)
                outputs = model(inputs).squeeze()
            
            y_pred_all = torch.cat((y_pred_all, outputs), dim=0)
    
    print(f"Length of y_pred_all tensor: {y_pred_all.size(0)}")

    # 최종 반환 시 리스트로 변환
    y_pred_all_cpu = y_pred_all.cpu().numpy()
    y_pred_all_list = y_pred_all_cpu.tolist()

    return y_pred_all_list