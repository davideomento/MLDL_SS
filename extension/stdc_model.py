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

    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        bce = self.bce(pred, target) # BCE with logits (no need for sigmoid)
        pred_sigmoid = torch.sigmoid(pred)

        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.pow(2).sum() + target.pow(2).sum()

        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)
        return dice + bce
    
    
# Calcola la mappa di dettaglio come edge map usando un kernel Laplaciano
# Il kernel è [[0, 1, 0], [1, -4, 1], [0, 1, 0]] che evidenzia i bordi.

def get_detail_target(seg):
    # seg: [B, H, W] con classi 0..18
    B, H, W = seg.shape
    num_classes = 19
    
    # One-hot encoding
    one_hot = torch.nn.functional.one_hot(seg, num_classes=num_classes)  # [B, H, W, C]
    one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
    
    laplacian = torch.tensor([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=torch.float32, device=seg.device).view(1,1,3,3)
    
    edges = torch.nn.functional.conv2d(one_hot, laplacian, padding=1, groups=num_classes)
    
    edge_map = (edges.abs().sum(dim=1, keepdim=True) > 0).float()  # [B,1,H,W]
    
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

class STDC_Seg(nn.Module):
    def __init__(self, num_classes=19, backbone='STDC1', use_detail=True):
        super(STDC_Seg, self).__init__()
        self.use_detail = use_detail
        self.num_classes = num_classes

        if backbone == 'STDC1':
            self.backbone = STDCNet813()
            feat_channels = [64, 256, 256, 512, 1024]
        elif backbone == 'STDC2':
            self.backbone = STDCNet1446()
            feat_channels = [64, 512, 1024, 2048, 2048]
        else:
            raise ValueError("Invalid backbone")

        self.feat_channels = feat_channels
        self.arm8 = AttentionRefinementModule(feat_channels[2], feat_channels[2])
        self.arm4 = AttentionRefinementModule(feat_channels[1], feat_channels[1])

        self.fusion = None  # inizializzata dinamicamente nel primo forward() TATANDRE

        self.seg_head = SegHead(in_channels=num_classes, mid_channels=64, num_classes=num_classes)

        if self.use_detail:
            self.detail_head = DetailHead(feat_channels[0]) 
            self.detail_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.final_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        print(f"feat2 shape: {feat2.shape}")
        print(f"feat4 shape: {feat4.shape}")
        print(f"feat8 shape: {feat8.shape}")
        print(f"feat16 shape: {feat16.shape}")
        print(f"feat32 shape: {feat32.shape}")

        context8 = self.arm8(feat8)
        context4 = F.interpolate(context8, size=feat4.size()[2:], mode='bilinear', align_corners=True)
        context4 = self.arm4(context4)
        context2 = F.interpolate(context4, size=feat2.size()[2:], mode='bilinear', align_corners=True)

        # Inizializza dinamicamente FeatureFusionModule se non ancora fatto
        if self.fusion is None:
            in_channels = feat2.size(1) + context2.size(1)
            self.fusion = FeatureFusionModule(
                num_classes=self.num_classes,
                in_channels=in_channels
            ).to(x.device)

        fused = self.fusion(feat2, context2)
        out = self.seg_head(fused)
        out = self.final_upsample(out)

        if self.use_detail:
            detail = self.detail_head(feat2)
            detail = self.detail_upsample(detail)
            return out, detail

        return out


if __name__ == "__main__":
    model = STDC_Seg(num_classes=19, backbone='STDC1', use_detail=True)
    inp = torch.randn(1, 3, 512, 1024)
    out = model(inp)
    if isinstance(out, tuple):
        print("Segmentation:", out[0].shape)
        print("Detail map:", out[1].shape)
    else:
        print("Segmentation:", out.shape)
