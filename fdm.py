class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        avg_features = self.shared_MLP(self.avg_pool(features))
        max_features = self.shared_MLP(self.max_pool(features))
        return self.sigmoid(avg_features + max_features)


class SptialAttention(nn.Module):
    def __init__(self):
        super(SptialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out)
        return self.sigmoid(out)


class FDM(nn.Module):
    def __init__(self, in_channels):
        super(FDM, self).__init__()
        self.ChannelAttention = ChannelAttention(in_channels)
        self.SpatialAttention = SptialAttention()

    def forward(self, features):
        channel_features = self.ChannelAttention(features) * features
        out = self.SpatialAttention(channel_features) * channel_features
        out1 = self.SpatialAttention(features) * features
        out = out + out1

        return out
