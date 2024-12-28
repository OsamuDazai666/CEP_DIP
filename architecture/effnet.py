from torch import nn
import torch.nn.functional as F

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3, stride=1, dropout_rate=0.2):
        super(MBConv, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion Phase
        hidden_dim = in_channels * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()  # Swish activation
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise Convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        
        # Squeeze-and-Excitation (optional)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        ) if expand_ratio != 1 else nn.Identity()
        
        # Projection Phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x) * x
        x = self.project(x)
        if self.use_residual:
            x += identity
        return x


class EffNet(nn.Module):
    def __init__(self):
        super().__init__()

        # conv1
        self.conv = nn.Conv2d(
            in_channels=3, 
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # block 1
        self.block_1 = MBConv(
            in_channels=32, 
            out_channels=16,
            kernel_size=3,
        )

        # block 2 
        self.block_2 = nn.Sequential(
            MBConv(in_channels=16, out_channels=24, kernel_size=3),
            MBConv(in_channels=24, out_channels=24, kernel_size=3)
        )

        # block 3
        self.block_3 = nn.Sequential(
            MBConv(in_channels=24, out_channels=40, kernel_size=5),
            MBConv(in_channels=40, out_channels=40, kernel_size=5)
        )

        # block 4
        self.block_4 = nn.Sequential(
            MBConv(in_channels=40, out_channels=80, kernel_size=3),
            MBConv(in_channels=80, out_channels=80, kernel_size=3),
            MBConv(in_channels=80, out_channels=80, kernel_size=3)
        )

        # block 5
        self.block_5 = nn.Sequential(
            MBConv(in_channels=80, out_channels=112, kernel_size=5),
            MBConv(in_channels=112, out_channels=112, kernel_size=5),
            MBConv(in_channels=112, out_channels=112,kernel_size=5,)
        )

        # block 6
        self.block_6 = nn.Sequential(
            MBConv(in_channels=112, out_channels=192, kernel_size=5),
            MBConv(in_channels=192, out_channels=192, kernel_size=5),
            MBConv(in_channels=192, out_channels=192, kernel_size=5),
            MBConv(in_channels=192, out_channels=192, kernel_size=5)
        )

        # block 7
        self.block_7 = MBConv(
            in_channels=192,
            out_channels=320,
            kernel_size=3,
        )


        # output layer
        self.out = nn.Linear(320 * 7 * 7, 4)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), kernel_size=2))
        x = F.relu(self.block_1(x))
        x = F.relu(F.max_pool2d(self.block_2(x), kernel_size=2)) 
        x = F.relu(F.max_pool2d(self.block_3(x), kernel_size=2)) 
        x = F.relu(self.block_4(x)) 
        x = F.relu(F.max_pool2d(self.block_5(x), kernel_size=2)) 
        x = F.relu(F.max_pool2d(self.block_6(x), kernel_size=2))
        x = F.relu(self.block_7(x))
        x = x.view(x.size(0), -1) # flatten the features
        x = self.out(x)
        return x
