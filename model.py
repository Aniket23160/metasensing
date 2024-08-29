import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [batch_size, 32, 50, 50]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [batch_size, 64, 25, 25]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [batch_size, 128, 12, 12]
        )
        
        self.fc = nn.Linear(128 * 12 * 12, 128)  # Output: [batch_size, 128]
        
        # LSTM expects [batch_size, sequence_length, input_size]
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, 128 * 12 * 12]
        x = self.fc(x)  # Output: [batch_size, 128]
        
        # Reshape to [batch_size, sequence_length, input_size] for LSTM
        x = x.unsqueeze(1).repeat(1, 25, 1)  # Repeat for 25 timesteps (matching target size)
        
        lstm_out, _ = self.lstm(x)
        output = self.output_layer(lstm_out)
        return output
