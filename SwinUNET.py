import torch
import torch.nn as nn
import timm
from torchvision.transforms import functional as TF

# DoubleConv 모듈 정의
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# Attention block for the decoder
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Swin Transformer 모델의 encoder 부분만 추출
class SwinEncoder(nn.Module):
    def __init__(self, swin_model):
        super(SwinEncoder, self).__init__()
        self.patch_embed = swin_model.patch_embed
        self.layers = swin_model.layers
        self.norm = swin_model.norm

    def forward(self, x):
        skips = []
        x = self.patch_embed(x)
        skips.append(x)  # skip1: Initial embedding
        for i, layer in enumerate(self.layers):
            x = layer(x)
            skips.append(x)  # Add intermediate features
        x = self.norm(x)
        return x, skips

# UNetDecoder 모듈 정의
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        self.attention = AttentionBlock(out_channels, skip_channels, out_channels // 2)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = TF.resize(x, size=skip.shape[2:])
        skip = self.attention(x, skip)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x

# UNet 모델 정의
class SwinUNet(nn.Module):
    def __init__(self, swin_encoder, num_classes):
        super(SwinUNet, self).__init__()
        self.encoder = swin_encoder

        # Decoder layers
        self.decoder4 = UNetDecoder(1536, 768, 768)
        self.decoder3 = UNetDecoder(768, 384, 384)
        self.decoder2 = UNetDecoder(384, 192, 192)
        self.decoder1 = UNetDecoder(192, 192, 96)  # Adjusted to 192 skip channels for consistency

        self.final_conv = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc_out, skips = self.encoder(x)

        # Permute the outputs to match the expected input shape for ConvTranspose2d
        enc_out = enc_out.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]
        skips = [skip.permute(0, 3, 1, 2) for skip in skips]

        # 스킵 연결 정의 (예시로 적절한 위치에서 추출)
        skip1 = skips[0]  # 초기 패치 임베딩 [1, 192, 56, 56]
        skip2 = skips[1]  # 첫 번째 레이어 이후 [1, 192, 56, 56]
        skip3 = skips[2]  # 두 번째 레이어 이후 [1, 384, 28, 28]
        skip4 = skips[3]  # 세 번째 레이어 이후 [1, 768, 14, 14]

        # Decoder
        dec4 = self.decoder4(enc_out, skip4)   # Output: [1, 768, 14, 14]
        dec3 = self.decoder3(dec4, skip3)      # Output: [1, 384, 28, 28]
        dec2 = self.decoder2(dec3, skip2)      # Output: [1, 192, 56, 56]
        dec1 = self.decoder1(dec2, skip1)      # Output: [1, 96, 56, 56]

        # Final Convolution
        out = self.final_conv(dec1)       # Output: [1, num_classes, 56, 56]
        out = TF.resize(out, size=(224, 224))  # Output: [1, num_classes, 224, 224]
        return out
