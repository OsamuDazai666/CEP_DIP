import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=4):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # Second convolutional layer
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 54 * 54, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        # if not grayscale make grayscale
        if x.size(1) == 3:
            x = torch.mean(x, dim=1, keepdim=True)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x