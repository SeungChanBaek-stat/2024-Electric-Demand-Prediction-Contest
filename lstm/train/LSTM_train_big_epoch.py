import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys, os
sys.path.append(os.pardir)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lstm')))
from functions import train_model_cv, train_model_all

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_dim, output_dim, num_layers, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm_day = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # MLP layers with batch normalization and dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_dim, output_dim)
        )

    def forward(self, x):
        # Initialize hidden state and cell state for LSTM
        h0_day = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0_day = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out_day, _ = self.lstm_day(x, (h0_day, c0_day))
        
        # Take the output from the last time step
        out_day = out_day[:, -1, :]
        
        # Forward pass through MLP
        out = self.mlp(out_day)
        return out



def load_data():
    data_dir = "c:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest\\data"
    train_path = data_dir + "\\electric_train_preprocessed.csv"
    save_dir = "C:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest\\model_param"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_data = pd.read_csv(train_path)
    return train_data, save_dir



def main():
    # torch.multiprocessing.set_start_method('spawn', force=True)
    train_data, save_dir = load_data()
    print("Train data shape:", train_data.shape)
    print("Train data 2nd dimension length:", train_data.shape[1])

    # 피처와 타겟 설정
    target = 'electric.elec'
    X_train = train_data.drop(columns=target).astype(np.float32)
    y_train = train_data[target].astype(np.float32)
    print("X_train shape:", X_train.shape)
    print(X_train.columns)
    print("y_train shape:", y_train.shape)

    # 필요없는 변수 삭제
    X_train.drop(['electric.num', 'year'], axis=1, inplace=True)

    SEQ_LENGTH_DAY = 24

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    input_dim = X_train.shape[1]
    hidden_dim = 64
    mlp_dim = 32
    output_dim = 1
    num_layers = 2
    dropout_prob = 0.5
    model = LSTMModel(input_dim, hidden_dim, mlp_dim, output_dim, num_layers, dropout_prob).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=0.001)

    num_epochs = 200
    batch_size = 256
    num_workers = 2
    
    best_train_loss = float('inf')
    # best_val_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    save_paths = []
    save_interval = 5

    print("Training LSTM model...")
    # train_loss_history, val_loss_history, save_paths = train_model_cv(
    train_loss_history, save_paths = train_model_all(
        model=model,
        dataset=(X_train, y_train),
        batch_size=batch_size,
        num_workers=num_workers,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        seq_len=SEQ_LENGTH_DAY
    )

    # 최적의 모델 선택
    # best_val_loss = min(val_loss_history)
    best_train_loss = min(train_loss_history)
    # best_epoch = val_loss_history.index(best_val_loss) + 1
    best_epoch = train_loss_history.index(best_train_loss) + 1
    best_model_path = save_paths[(best_epoch // save_interval) - 1]
    best_model_state = torch.load(best_model_path)
    model.load_state_dict(best_model_state)

    # 최종 모델 저장
    final_save_path = os.path.join(save_dir, 'lstm_model_best_epoch200.pth')
    torch.save(best_model_state, final_save_path)
    # print(f"Best model saved with validation loss: {best_val_loss} at epoch {best_epoch}")
    print(f"Best model saved with validation loss: {best_train_loss} at epoch {best_epoch}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    # plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.title('Training and Validation Loss')
    plt.title('Training Loss')
    plt.show()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()












