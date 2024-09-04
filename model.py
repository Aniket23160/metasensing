import torch
import torch.nn as nn

class ModifiedCNNModel(nn.Module):
    def __init__(self):
        super(ModifiedCNNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.fc = nn.Linear(128 * 100 * 100, 128)  # Adjusted to match input dimensions
        
        self.output_layer = nn.Linear(128, 60)  # Output layer to match 30 coordinates (x, y)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, 128 * 100 * 100]
        x = self.fc(x)  # Output: [batch_size, 128]
        output = self.output_layer(x)  # Output: [batch_size, 60] (30 pairs of coordinates)
        output = output.view(batch_size, 30, 2)  # Reshape to [batch_size, 30, 2]
        return output

