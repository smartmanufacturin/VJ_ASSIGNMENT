import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math


# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # B x N x (C//8)
        proj_key   = self.key_conv(x).view(B, -1, N)                    # B x (C//8) x N
        energy     = torch.bmm(proj_query, proj_key)/math.sqrt(N)                   # B x N x N
        attention  = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, N)                    # B x C x N
        out        = torch.bmm(proj_value, attention.permute(0, 2, 1))     # B x C x N
        out        = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

# Residual Block with nn.ReLU Activation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.activation(out)
        return out

# Residual U-Net with Skip Connections, Self-Attention, and nn.ReLU Activation
class ResidualAttnUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(ResidualAttnUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET
        for feature in features:
            self.downs.append(ResidualBlock(in_channels, feature))
            in_channels = feature 
            
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(ResidualBlock(feature*2, feature)) 
            
        self.bottleneck = ResidualBlock(features[-1], features[-1]*2)
        self.attention = SelfAttention(features[3] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.attention(x)
        
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
    
    
def build_model(
    in_channels: int, 
    out_channels: int
):
    model = ResidualAttnUNet(in_channels=in_channels, out_channels=out_channels) 
    return model 

# Test Code
if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = build_model(in_channels=3, out_channels=1)
    output = model(x)
    print("ResidualAttnUNet Output shape:", output.shape)
