import torch
import torch.nn as nn
import torch.nn.functional as F
from stdcnet import STDCNet813, STDCNet1446


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super(SegHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.block(x)


class DetailHead(nn.Module):
    def __init__(self, in_channels):
        super(DetailHead, self).__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.detail(x)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))
    
class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        # x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from spatial path) + 1024(from context path) + 2048(from context path)
        # resnet18  1024 = 256(from spatial path) + 256(from context path) + 512(from context path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class STDC_Seg(nn.Module):
    def __init__(self, num_classes=19, backbone='STDC1', use_detail=True):
        super(STDC_Seg, self).__init__()
        self.use_detail = use_detail

        if backbone == 'STDC1':
            self.backbone = STDCNet813()
            feat_channels = [64, 256, 512]  # feat2, feat4, feat8
        elif backbone == 'STDC2':
            self.backbone = STDCNet1446()
            feat_channels = [64, 512, 1024]
        else:
            raise ValueError("Invalid backbone")

        # Attention Refinement Module (ARM)
        self.arm16 = AttentionRefinementModule(feat_channels[2], feat_channels[2])  # feat8 -> feat4 size
        self.arm32 = AttentionRefinementModule(feat_channels[1], feat_channels[1])  # feat4 -> feat2 size

        # Segmentation head
        self.seg_head = SegHead(feat_channels[0], 64, num_classes)

        # Detail guidance
        if self.use_detail:
            self.detail_head = DetailHead(feat_channels[0])
            self.detail_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # Feature fusion
        self.fusion = FeatureFusionModule(num_classes=num_classes,
                                          in_channels=feat_channels[0] + feat_channels[0])  # feat2 + upsampled context

        # Upsampling final prediction to input size
        self.final_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        feat2, feat4, feat8, *_ = self.backbone(x)

        # Apply ARM to context path
        context16 = self.arm16(feat8)           # Output: same channel as feat4
        context8 = F.interpolate(context16, size=feat4.size()[2:], mode='bilinear', align_corners=True)

        context8 = self.arm32(context8)         # Output: same channel as feat2
        context8_up = F.interpolate(context8, size=feat2.size()[2:], mode='bilinear', align_corners=True)

        # Feature Fusion (context + spatial)
        fused = self.fusion(feat2, context8_up)

        # Final segmentation
        out = self.seg_head(fused)
        out = self.final_upsample(out)

        if self.use_detail:
            detail = self.detail_head(feat2)
            detail = self.detail_upsample(detail)
            return out, detail

        return out



if __name__ == "__main__":
    model = STDC_Seg(num_classes=19, backbone='STDC2')
    inp = torch.randn(1, 3, 512, 1024)
    out = model(inp)
    if isinstance(out, tuple):
        print("Segmentation:", out[0].shape)
        print("Detail map:", out[1].shape)
    else:
        print("Segmentation:", out.shape)
