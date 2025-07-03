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

# Detail head on feat8 (stage3)
class DetailHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, 1, 1)
        )

    def forward(self, x):
        return self.block(x)
    
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

class STDC(nn.Module):
    def __init__(self, num_class=19, backbone='STDC2', use_aux=False, use_detail=False):
        super().__init__()
        assert not (use_aux and use_detail), "Only one of aux or detail at training"
        self.use_aux = use_aux
        self.use_detail = use_detail

        # Backbone
        if backbone == 'STDC1':
            self.backbone = STDCNet813()
        elif backbone == 'STDC2':
            self.backbone = STDCNet1446()
        else:
            raise ValueError("Unsupported backbone")

        # Auxiliary heads (in training)
        if use_aux:
            self.aux3 = SegHead(256, num_class, num_class)
            self.aux4 = SegHead(512, num_class, num_class)
            self.aux5 = SegHead(1024, num_class, num_class)

        # Global pooling and ARM modules
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.arm5 = AttentionRefinementModule(1024, 1024)
        self.arm4 = AttentionRefinementModule(512, 512)

        # Channel reductions via ConvBlock 1x1
        self.conv5 = ConvBlock(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv4 = ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)

        # Feature Fusion Module
        self.ffm = FeatureFusionModule(num_classes=128, in_channels=512)

        # Segmentation head
        self.seg_head = SegHead(128, 128, num_class)

        # Detail head (apply on feat8, stage3)
        if use_detail:
            self.detail_head = DetailHead(256)

    def forward(self, x, is_training=False):
        size = x.size()[2:]
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        # Auxiliary outputs
        if self.use_aux and is_training:
            aux3 = self.aux3(feat8)
            aux4 = self.aux4(feat16)
            aux5 = self.aux5(feat32)

        # Context path Stage5
        x5_pool = self.pool(feat32)
        x5_arm = self.arm5(feat32)
        x5 = x5_pool + x5_arm
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)

        # Context path Stage4
        x4_arm = self.arm4(feat16)
        x4 = self.conv4(x4_arm + feat16)
        x4 = x4 + x5
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)

        # Feature fusion and segmentation
        x_fuse = self.ffm(x4, feat8)
        seg_out = self.seg_head(x_fuse)
        seg_out = F.interpolate(seg_out, size=size, mode='bilinear', align_corners=True)

        # Detail branch
        if self.use_detail and is_training:
            detail_out = self.detail_head(feat8)
            detail_out = F.interpolate(detail_out, size=size, mode='bilinear', align_corners=True)
            return seg_out, detail_out

        if self.use_aux and is_training:
            return seg_out, (aux3, aux4, aux5)

        return seg_out

if __name__ == '__main__':
    model = STDC(num_class=19, backbone='STDC2', use_detail=True)
    inp = torch.randn(1, 3, 512, 1024)
    out = model(inp, is_training=True)
    print("Segmentation:", out[0].shape)
    if isinstance(out, tuple):
        print("Detail:", out[1].shape)