import torch
import torch.nn as nn
import torch.nn.functional as F
from stdcnet import STDCNet813, STDCNet1446

def get_detail_target(seg):
    """
    Genera una mappa dei bordi dai target di segmentazione, ignorando i pixel con valore 255.
    
    Args:
        seg (Tensor): tensore [B, H, W] con classi da 0 a 18, e 255 per i pixel da ignorare.
    
    Returns:
        Tensor: mappa binaria dei bordi [B, H, W], con pixel ignorati esclusi dal calcolo.
    """
    #B, H, W = seg.shape
    num_classes = 19
    ignore_index = 255

    # Crea maschera dei pixel validi
    valid_mask = (seg != ignore_index)  # [B, H, W]

    # Sostituisci i 255 temporaneamente con 0 per evitare errori nell'one-hot
    seg_clean = seg.clone()
    seg_clean[~valid_mask] = 0

    # One-hot encoding: [B, H, W, C] -> [B, C, H, W]
    one_hot = torch.nn.functional.one_hot(seg_clean, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Filtro laplaciano replicato per ogni classe
    base_kernel = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=torch.float32, device=seg.device)
    laplacian = base_kernel.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)  # [C, 1, 3, 3]

    # Convoluzione depthwise (gruppi = num_classes)
    edges = torch.nn.functional.conv2d(one_hot, laplacian, padding=1, groups=num_classes)

    # Mappa binaria dei bordi, esclusi i pixel ignorati
    edge_map = (edges.abs().sum(dim=1, keepdim=True) > 0).float()  # [B,1,H,W]
    edge_map *= valid_mask.unsqueeze(1).float()  # ignora i 255

    return edge_map.squeeze(1)  # [B, H, W]




#Conv 2D + BatchNorm + ReLU 
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
    

class DetailLoss(nn.Module):
    def __init__(self, eps=1.0):
        super(DetailLoss, self).__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        bce = self.bce(pred, target) # BCE with logits (no need for sigmoid)
        pred_sigmoid = torch.sigmoid(pred)

        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.pow(2).sum() + target.pow(2).sum()

        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)
        return dice + bce

# Dummy placeholders for external modules
class STDCNet813(nn.Module):
    def forward(self, x):
        B, C, H, W = x.size()
        return [torch.randn(B, 32, H//2, W//2),
                torch.randn(B, 64, H//4, W//4),
                torch.randn(B, 256, H//8, W//8),
                torch.randn(B, 512, H//16, W//16),
                torch.randn(B, 1024, H//32, W//32)]

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SPPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.AdaptiveAvgPool2d(output_size=2),
            nn.AdaptiveAvgPool2d(output_size=4)
        ])
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(3)
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

class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat2, fusion):
        x = torch.cat([feat2, fusion], dim=1)
        return self.conv(x)

class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.conv(x)

class DetailHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)

class STDC_Seg(nn.Module):
    def __init__(self, num_classes=19, backbone='STDC1', use_detail=True):
        super(STDC_Seg, self).__init__()
        self.use_detail = use_detail
        self.num_classes = num_classes
        if backbone == 'STDC1':
            self.backbone = STDCNet813()
            feat_channels = [32, 64, 256, 512, 1024]
            # Per STDC1: feat2 ha 32 canali, fusion_input (prima di fusion) avrà 320
            fusion_in_channels = 32 + 320  # = 352

        elif backbone == 'STDC2':
            self.backbone = STDCNet1446()
            feat_channels = [64, 512, 1024, 2048, 2048]
            # Per STDC2: feat2 ha 64 canali, fusion_input (prima di fusion) avrà sempre 320
            fusion_in_channels = 64 + 320  # = 384
        else:
            raise ValueError("Invalid backbone")
        
        self.feat_channels = feat_channels
        self.arm8 = AttentionRefinementModule(feat_channels[2], feat_channels[2])
        self.arm4 = AttentionRefinementModule(feat_channels[1], feat_channels[1])
        self.arm16 = AttentionRefinementModule(feat_channels[3], feat_channels[3])
        self.context16_conv = nn.Conv2d(feat_channels[3], feat_channels[2], kernel_size=1, bias=False)
        self.refine_context16 = nn.Sequential(
            nn.Conv2d(feat_channels[2], feat_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.sppm = SPPM(in_channels=feat_channels[4], out_channels=feat_channels[2])
        self.global_context_conv = nn.Sequential(
            nn.Conv2d(feat_channels[2], 320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )

        # Istanzio qui il FeatureFusionModule con i canali già calcolati:
        self.fusion = FeatureFusionModule(
            num_classes=self.num_classes,
            in_channels=fusion_in_channels
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(320, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.seg_head = SegHead(in_channels=128 + feat_channels[1], mid_channels=64, num_classes=num_classes)

        if self.use_detail:
            self.detail_head = DetailHead(feat_channels[0])
            self.detail_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        context8 = self.arm8(feat8)
        context4 = self.arm4(feat4)
        context16 = self.arm16(feat16)

        context16_up = F.interpolate(context16, size=context8.shape[2:], mode='bilinear', align_corners=True)
        context16_up = self.context16_conv(context16_up)
        context16_up = self.refine_context16(context16_up)
        context8 = context8 + context16_up

        context8_up = F.interpolate(context8, size=context4.shape[2:], mode='bilinear', align_corners=True)
        fusion_input = torch.cat([context4, context8_up], dim=1)
        fusion_input = F.interpolate(fusion_input, size=feat2.shape[2:], mode='bilinear', align_corners=True)

        global_context = self.sppm(feat32)
        global_context_up = F.interpolate(global_context, size=fusion_input.shape[2:], mode='bilinear', align_corners=True)
        global_context_up = self.global_context_conv(global_context_up)

        fusion_input = fusion_input + global_context_up

        fused = self.fusion(feat2, fusion_input)

        decoder_out = self.decoder(fused)
        feat4_up = F.interpolate(feat4, size=decoder_out.shape[2:], mode='bilinear', align_corners=True)
        decoder_out = torch.cat([decoder_out, feat4_up], dim=1)

        out = self.seg_head(decoder_out)
        out = self.final_upsample(out)

        if self.use_detail:
            detail = self.detail_head(feat2)
            detail = self.detail_upsample(detail)
            return out, detail

        return out