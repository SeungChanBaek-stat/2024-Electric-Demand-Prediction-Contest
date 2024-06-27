import sys, os
sys.path.append('C:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lstm.functions import eval_model



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

features = ['month', 'hour', 'day_of_year', 'electric_train.nph_ta', 'electric_train.nph_hm', 'electric_train.nph_ws_10m', 'electric_train.nph_rn_60m']


def main():
    # torch.multiprocessing.set_start_method('spawn', force=True)
    train_data, save_dir = load_data()
    save_path = os.path.join(save_dir, 'lstm_model_best_epoch200.pth')
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

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 모델 인스턴스 생성
    input_dim = X_train.shape[1]
    hidden_dim = 64
    mlp_dim = 32
    output_dim = 1
    num_layers = 2
    dropout_prob = 0.5
    # 모델 불러오기
    model = LSTMModel(input_dim, hidden_dim, mlp_dim, output_dim, num_layers, dropout_prob).to(device)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print(model)


    # 시퀀스 길이 설정
    SEQ_LENGTH_YEAR = 24 * 365
    SEQ_LENGTH_MONTH = 24 * 30
    SEQ_LENGTH_DAY = 24    

    batch_size = 256
    # chunk_size = 200000  # 청크 크기
    num_workers = 2

    mse, mae, y_true_all, y_pred_all = eval_model(model = model,
                                                dataset = (X_train, y_train),
                                                batch_size = batch_size,
                                                num_workers = num_workers,
                                                device = device, 
                                                seq_len = SEQ_LENGTH_DAY)
    plt.figure(figsize=(14, 7))
    plt.plot(y_true_all, label='True')
    plt.plot(y_pred_all, label='Predicted')
    plt.legend()
    plt.title('True vs Predicted Power Demand')
    plt.xlabel('Samples')
    plt.ylabel('Power Demand')
    plt.show()



if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
