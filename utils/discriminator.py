import torch.nn as nn
import torch.nn.functional as F

class FCDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(FCDiscriminator, self).__init__()
        
        # Conv layers with kernel=4, stride=2
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Input x shape: [B, C, H, W], C = numero classi (output softmaxato)
        
        x = self.leaky_relu(self.conv1(x))  # [B, 64, H/2, W/2]
        x = self.leaky_relu(self.conv2(x))  # [B, 128, H/4, W/4]
        x = self.leaky_relu(self.conv3(x))  # [B, 256, H/8, W/8]
        x = self.leaky_relu(self.conv4(x))  # [B, 512, H/16, W/16]
        x = self.conv5(x)                    # [B, 1, H/32, W/32]

        # Up-sample all the way back to input spatial size
        x = F.interpolate(x, size=(x.shape[2]*32, x.shape[3]*32), mode='bilinear', align_corners=True)
        # oppure in modo più preciso se conosci input: 
        # x = F.interpolate(x, size=(input_height, input_width), mode='bilinear', align_corners=True)

        return x  # Output shape: [B, 1, H, W], spatially aligned con input

    