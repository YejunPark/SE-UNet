import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

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

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=33, features=[64, 128, 256, 512]):
        super(UNetPlusPlus, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder path
        self.up_convs = nn.ModuleList()
        self.double_convs = nn.ModuleList()

        for feature in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            double_convs_block = nn.ModuleList([
                DoubleConv(feature * (2 + i), feature) for i in range(len(features))
            ])
            self.double_convs.append(double_convs_block)

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Deep supervision
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(feature, out_channels, kernel_size=1) for feature in reversed(features)
        ])

    def forward(self, x):
        input_size = x.shape[2:]  # Original input size for resizing
        skip_connections = [[] for _ in range(len(self.features))]
        x = self.encoder_forward(x, skip_connections)
        x = self.bottleneck(x)
        outputs = self.decoder_forward(x, skip_connections)
        
        # Final output with deep supervision
        deep_outputs = [TF.resize(output, size=input_size) for output in outputs]
        final_output = torch.stack(deep_outputs, dim=0).mean(dim=0)
        
        return final_output

    def encoder_forward(self, x, skip_connections):
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            skip_connections[idx].append(x)
            x = self.pool(x)
        return x

    def decoder_forward(self, x, skip_connections):
        outputs = []
        for idx in range(len(self.features)):
            up_conv = self.up_convs[idx]
            x = up_conv(x)
            skips = skip_connections[len(self.features) - 1 - idx]
            for jdx, skip in enumerate(skips):
                if x.shape != skip.shape:
                    x = TF.resize(x, size=skip.shape[2:])
                x = torch.cat([x, skip], dim=1)
                x = self.double_convs[idx][jdx](x)
            outputs.append(self.deep_supervision[idx](x))
        return outputs