import torch
import torch.nn as nn
import torch.nn.functional as F
from stdcnet import STDCNet813, STDCNet1446


#Prende feature con in_channels canali, fa una conv 3x3, poi norm e poi ReLU. Poi conv 1x1 per ridurre a num_classes canali
#Quindi output è di dimensione [B, num_classes, H, W]
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

#Estrae una detail map (un singolo canale)
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
        return self.detail(x)  # (B, 1, H, W)
    
class DetailLoss(nn.Module):
    def __init__(self, eps=1.0):
        super(DetailLoss, self).__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, valid_mask=None):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        valid_mask = (target != 255)

        pred = pred[valid_mask]
        target = target[valid_mask]

        bce = self.bce(pred, target)  # BCE with logits
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.pow(2).sum() + target.pow(2).sum()

        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)
        return dice + bce

    
    
# Calcola la mappa di dettaglio come edge map usando un kernel Laplaciano
# Il kernel è [[0, 1, 0], [1, -4, 1], [0, 1, 0]] che evidenzia i bordi.

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

#Raffina le feature con l'attenzione: calcola la media spaziale di ogni canale (vettore di dim [B,C]) e poi lo passa a conv 1x1 + BatchNorm + Sigmoid per ottenere pesi per canale tra 0 e 1.
# Moltiplica l'input per questi pesi, cioè aumenta o diminuisce l'importanza di ogni canale in base alla sua media spaziale. 
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        x = self.avgpool(input)       # [B, C, 1, 1]
        x = self.conv(x)              # [B, out_channels, 1, 1]
        x = self.bn(x)
        x = self.sigmoid(x)
        return input * x              # Broadcasting will match channels automatically


#Fonde due feature map (dettaglio + contesto), concatena i canali delle due feature poi conv 3x3 + BN + ReLU senza stride per mantenere dimensioni.
# Calcola attenzione canale con conv 1x1, relu, conv 1x1, sigmoid come pesi.
# Moltiplica le feature per pesi di attenzione e aggiunge residual connection (somma le feature originali). Restituisce feature fuse e “rafforzate”.
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

# SPPM semplificato per il contesto globale da feat32
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

class STDC_Seg(nn.Module):
    def __init__(self, num_classes=19, backbone='STDC2', use_detail=True):
        super(STDC_Seg, self).__init__()
        self.use_detail = use_detail
        self.num_classes = num_classes

        # Backbone and feature channels
        if backbone == 'STDC1':
            self.backbone = STDCNet813()
            feat_channels = [32, 64, 256, 512, 1024]
        elif backbone == 'STDC2':
            self.backbone = STDCNet1446()
            feat_channels = [32, 64, 256, 512, 1024]
        else:
            raise ValueError("Invalid backbone")
        self.feat_channels = feat_channels

        # Attention Refinement Modules on Stage5 (1/32) and Stage4 (1/16)
        self.arm32 = AttentionRefinementModule(feat_channels[4], feat_channels[4])
        self.arm16 = AttentionRefinementModule(feat_channels[3], feat_channels[3])
        # Adjust channels after upsample
        self.context32_conv = nn.Conv2d(feat_channels[4], feat_channels[3], kernel_size=1, bias=False)
        self.context16_conv = nn.Conv2d(feat_channels[3], feat_channels[2], kernel_size=1, bias=False)

        # Spatial Pyramid Pooling Module on Stage5
        self.sppm = SPPM(in_channels=feat_channels[4], out_channels=feat_channels[2])

        # Global context via GAP (added to Stage3 level)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_channels[4], feat_channels[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_channels[2]),
            nn.ReLU(inplace=True)
        )

        # Feature Fusion: fuse Stage3 context (1/8) with spatial path (feat2)
        fusion_in_channels = feat_channels[2] + feat_channels[0]  # context + spatial
        self.fusion = FeatureFusionModule(
            num_classes=self.num_classes,
            in_channels=fusion_in_channels
        )

        # Segmentation head
        self.seg_head = SegHead(in_channels=num_classes, mid_channels=64, num_classes=num_classes)

        # Detail branch (spatial path)
        if self.use_detail:
            self.detail_head = DetailHead(feat_channels[0])  # low-level feature
            self.detail_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.final_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Backbone features: feat2 (1/2), feat4 (1/4), feat8 (1/8), feat16 (1/16), feat32 (1/32)
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        # Context path U-shaped fusion
        # Stage5 -> Stage4
        context32 = self.arm32(feat32)  # [B,1024, H/32, W/32]
        up16 = F.interpolate(context32, size=feat16.shape[2:], mode='bilinear', align_corners=True)
        up16 = self.context32_conv(up16)  # [B,512, H/16, W/16]

        context16 = self.arm16(feat16)    # [B,512, H/16, W/16]
        fused16 = up16 + context16        # combine

        # Stage4 -> Stage3
        up8 = F.interpolate(fused16, size=feat8.shape[2:], mode='bilinear', align_corners=True)
        fused8 = up8 + feat8              # no ARM on feat8, direct sum
        fused8 = self.context16_conv(fused8)  # reduce from 512 to 256

        # Global context addition
        global_ctx = self.sppm(feat32)  # [B,256, H/8, W/8]
        global_ctx = self.global_pool(feat32)  # [B,256, 1, 1]
        global_ctx = F.interpolate(global_ctx, size=fused8.shape[2:], mode='bilinear', align_corners=True)
        fused8 = fused8 + global_ctx

        # Spatial path (detail)
        if self.use_detail:
            detail = self.detail_head(feat2)
            detail = self.detail_upsample(detail)  # [B, C, H/8, W/8]
        else:
            detail = feat2  # fallback if no detail head

        # Feature fusion at Stage3 (1/8)
        fused = self.fusion(detail, fused8)

        # Segmentation prediction
        out = self.seg_head(fused)
        out = self.final_upsample(out)

        if self.use_detail:
            return out, detail
        return out


if __name__ == "__main__":
    model = STDC_Seg(num_classes=19, backbone='STDC2', use_detail=True)
    inp = torch.randn(1, 3, 512, 1024)
    out = model(inp)
    if isinstance(out, tuple):
        print("Segmentation:", out[0].shape)
        print("Detail map:", out[1].shape)
    else:
        print("Segmentation:", out.shape)
