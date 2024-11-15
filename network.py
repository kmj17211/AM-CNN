import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias = False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // ratio, in_channels, 1, bias = False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding = kernel_size // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        out = self.conv(x)
        return self.sigmoid(out)
    
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio = 16, kernel_size = 7):
        super(CBAM, self).__init__()

        self.ch_at = ChannelAttention(in_channels, ratio)
        self.sp_at = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ch_at(x) * x
        x = self.sp_at(x) * x
        return x
    
class AM_CNN(nn.Module):
    def __init__(self, num_classes: int = 10, num_feature: int = 16):
        super(AM_CNN, self).__init__()

        # 100 x 100 x 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_feature, kernel_size = 3, bias = True),
            nn.BatchNorm2d(num_feature),
            nn.ReLU(),
            CBAM(num_feature)
        )

        # 98 x 98 x 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feature, num_feature * 2, kernel_size = 7, bias = True),
            nn.BatchNorm2d(num_feature * 2),
            nn.ReLU(),
            CBAM(num_feature * 2)
        )
        
        # 92 x 92 x 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_feature * 2, num_feature * 4, kernel_size = 5, bias = True),
            nn.BatchNorm2d(num_feature * 4),
            nn.ReLU(),
            CBAM(num_feature * 4)
        )

        # 88 x 88 x 64
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_feature * 4, num_feature * 8, kernel_size = 5, bias = True),
            nn.BatchNorm2d(num_feature * 8),
            nn.ReLU(),
            CBAM(num_feature * 8)
        )

        # 84 x 84 x 128
        # max pool
        # 42 x 42 x 128
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_feature * 8, num_feature * 16, kernel_size = 5, bias = True),
            nn.BatchNorm2d(num_feature * 16),
            nn.ReLU(),
            CBAM(num_feature * 16)
        )

        # 38 x 38 x 256
        # max pool
        # 19 x 19 x 256
        self.conv6 = nn.Sequential(
            nn.Conv2d(num_feature * 16, num_feature * 8, kernel_size = 6, bias = True),
            nn.BatchNorm2d(num_feature * 8),
            nn.ReLU(),
            CBAM(num_feature * 8)
        )

        # 14 x 14 x 128
        # max pool
        # 7 x 7 x 128
        self.conv7 = nn.Sequential(
            nn.Conv2d(num_feature * 8, num_feature * 4, kernel_size = 5, bias = True),
            nn.BatchNorm2d(num_feature * 4),
            nn.ReLU(),
            CBAM(num_feature * 4)
        )

        # 3 x 3 x 64
        self.last_conv = nn.Sequential(
            nn.Conv2d(num_feature * 4, num_classes, kernel_size = 3, bias = True),
            nn.Softmax(dim = 1),
            nn.Flatten()
        )

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool(self.conv4(x))
        x = self.max_pool(self.conv5(x))
        x = self.max_pool(self.conv6(x))
        x = self.conv7(x)
        x = self.last_conv(x)
        return x
