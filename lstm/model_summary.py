import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import os

# 임시로 PATH에 Graphviz의 bin 디렉토리 추가 (이미 경로에 추가되어 있다면 필요 없음)
os.environ["PATH"] += os.pathsep + r"C:\\Program Files\\Graphviz\\bin"

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

# 모델 인스턴스 생성
input_dim = 14  
hidden_dim = 64  
mlp_dim = 32  
output_dim = 1  
num_layers = 2  
dropout_prob = 0.5  
SEQ_LENGTH_DAY = 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMModel(input_dim, hidden_dim, mlp_dim, output_dim, num_layers, dropout_prob).to(device)
print(model)

# 요약 출력
summary(model, input_size=(256, SEQ_LENGTH_DAY, input_dim))  # (sequence_length, input_dim)

# 모델 다이어그램 생성
x = torch.randn(256, SEQ_LENGTH_DAY, input_dim).to(device)  # (batch_size, sequence_length, input_dim)
output = model(x)
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("lstm_model_diagram", format="png")
dot