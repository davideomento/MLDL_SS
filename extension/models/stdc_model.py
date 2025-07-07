import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stdcnet import STDCNet813, STDCNet1446

# Semantic segmentation head: 3x3 conv -> BN -> ReLU -> 1x1 conv
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

# Detail head: extracts edge map from input features (outputs single channel)
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

# Custom loss for detail prediction: BCE + Dice
class DetailLoss(nn.Module):
    def __init__(self, eps=1.0):
        super(DetailLoss, self).__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        valid_mask = (target != 255)
        pred = pred[valid_mask]
        target = target[valid_mask]

        bce = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.pow(2).sum() + target.pow(2).sum()
        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)
        return dice + bce

# Generate binary edge map from segmentation labels using a Laplacian filter
def get_detail_target(seg):
    num_classes = 19
    ignore_index = 255

    valid_mask = (seg != ignore_index)
    seg_clean = seg.clone()
    seg_clean[~valid_mask] = 0

    one_hot = torch.nn.functional.one_hot(seg_clean, num_classes=num_classes).permute(0, 3, 1, 2).float()
    base_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=seg.device)
    laplacian = base_kernel.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)

    edges = torch.nn.functional.conv2d(one_hot, laplacian, padding=1, groups=num_classes)
    edge_map = (edges.abs().sum(dim=1, keepdim=True) > 0).float()
    edge_map *= valid_mask.unsqueeze(1).float()
    return edge_map.squeeze(1)

# Simple Conv -> BN -> ReLU block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

# Attention Refinement Module: channel attention based on global average pooling
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        x = self.avgpool(input)   # [B, C, 1, 1]
        x = self.conv(x)          # [B, out_channels, 1, 1]
        x = self.bn(x)
        x = self.sigmoid(x)
        return input * x          # Channel-wise scaling

# Feature Fusion Module: merges spatial & context paths using channel attention
class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels, num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        feature = self.convblock(x)

        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

# Spatial Pyramid Pooling Module (simplified): aggregates global context
class SPPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveAvgPool2d(4)
        ])
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in range(3)
        ])
        self.out_conv = nn.Conv2d(in_channels + 3 * out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        H, W = x.shape[2:]
        out = [x]
        for stage, conv in zip(self.stages, self.convs):
            pooled = stage(x)
            conv_out = conv(pooled)
            upsampled = F.interpolate(conv_out, size=(H, W), mode='bilinear', align_corners=True)
            out.append(upsampled)
        out = torch.cat(out, dim=1)
        return self.relu(self.bn(self.out_conv(out)))

# Full STDC segmentation model
class STDC_Seg(nn.Module):
    def __init__(self, num_classes=19, backbone='STDC2', use_detail=True):
        super(STDC_Seg, self).__init__()
        self.use_detail = use_detail
        self.num_classes = num_classes

        # Choose backbone
        if backbone == 'STDC1':
            self.backbone = STDCNet813()
            feat_channels = [32, 64, 256, 512, 1024]
            fusion_in_channels = 32 + 320  # 352
        elif backbone == 'STDC2':
            self.backbone = STDCNet1446()
            feat_channels = [32, 64, 256, 512, 1024]
            fusion_in_channels = 32 + 320  # 384
        else:
            raise ValueError("Invalid backbone")

        self.feat_channels = feat_channels
        self.arm8 = AttentionRefinementModule(feat_channels[2], feat_channels[2])
        self.arm4 = AttentionRefinementModule(feat_channels[1], feat_channels[1])
        self.arm16 = AttentionRefinementModule(feat_channels[3], feat_channels[3])
        self.context16_conv = nn.Conv2d(feat_channels[3], feat_channels[2], 1, bias=False)

        self.sppm = SPPM(feat_channels[4], feat_channels[2])
        self.global_context_conv = nn.Sequential(
            nn.Conv2d(feat_channels[2], 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )

        self.fusion = FeatureFusionModule(num_classes=num_classes, in_channels=fusion_in_channels)
        self.seg_head = SegHead(in_channels=num_classes, mid_channels=64, num_classes=num_classes)

        if self.use_detail:
            self.detail_head = DetailHead(feat_channels[0])
            self.detail_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.final_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Extract features from backbone
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        # Apply Attention Refinement Modules
        context8 = self.arm8(feat8)
        context4 = self.arm4(feat4)
        context16 = self.arm16(feat16)
        context16_up = F.interpolate(context16, size=context8.shape[2:], mode='bilinear', align_corners=True)
        context16_up = self.context16_conv(context16_up)
        context8 = context8 + context16_up
        context8_up = F.interpolate(context8, size=context4.shape[2:], mode='bilinear', align_corners=True)

        fusion_input = torch.cat([context4, context8_up], dim=1)
        fusion_input = F.interpolate(fusion_input, size=feat2.shape[2:], mode='bilinear', align_corners=True)

        # Add global context
        global_context = self.sppm(feat32)
        global_context_up = F.interpolate(global_context, size=fusion_input.shape[2:], mode='bilinear', align_corners=True)
        global_context_up = self.global_context_conv(global_context_up)
        fusion_input = fusion_input + global_context_up

        # Fuse detail and context features
        fused = self.fusion(feat2, fusion_input)
        out = self.seg_head(fused)
        out = self.final_upsample(out)

        if self.use_detail:
            detail = self.detail_head(feat2)
            detail = self.detail_upsample(detail)
            return out, detail

        return out

# Test the model
if __name__ == "__main__":
    model = STDC_Seg(num_classes=19, backbone='STDC2', use_detail=True)
    inp = torch.randn(1, 3, 512, 1024)
    out = model(inp)
    if isinstance(out, tuple):
        print("Segmentation:", out[0].shape)
        print("Detail map:", out[1].shape)
    else:
        print("Segmentation:", out.shape)
