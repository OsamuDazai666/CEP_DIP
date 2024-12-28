from torch import nn
from torch.nn import functional as F

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dBlock, self).__init__()

        self.conv_layer1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.conv_layer2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)

        return x

class TinyVgg(nn.Module):
    def __init__(self, in_channels, inter_channels, classes):
        super(TinyVgg, self).__init__()

        self.conv_block = Conv2dBlock(
            in_channels=in_channels, 
            out_channels=inter_channels,
            kernel_size=3,
        )

        self.conv_block2 = Conv2dBlock(
            in_channels=inter_channels, 
            out_channels=inter_channels,
            kernel_size=3,
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32 * 28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=216)
        self.fc4 = nn.Linear(in_features=216, out_features=classes)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv_block(x)))
        x = self.max_pool(F.relu(self.conv_block2(x)))
        x = self.max_pool(F.relu(self.conv_block2(x)))
        x = x.view(x.size(0), -1) # flatten the features
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x




        

