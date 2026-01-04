import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]

class ChangeDetectionModel(nn.Module):
    """
    Siamese U-Net for Change Detection.
    Features from both images are extracted using a shared encoder,
    then their differences are passed through a U-Net decoder with skip connections.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(ChangeDetectionModel, self).__init__()
        self.encoder = UNetEncoder(n_channels)
        
        # Decoder blocks
        # We use the absolute difference of features for skip connections
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2):
        # Extract features (Siamese)
        feat1 = self.encoder(t1) # [x1, x2, x3, x4, x5]
        feat2 = self.encoder(t2)
        
        # Calculate feature differences for skip connections
        diffs = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]
        
        # Decode
        # diffs[-1] is the bottleneck diff
        x = self.up1(diffs[4])
        x = torch.cat([x, diffs[3]], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, diffs[2]], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, diffs[1]], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, diffs[0]], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return self.sigmoid(logits)

    def get_features(self, x):
        """Helper for legacy inference if needed"""
        return self.encoder(x)[-1] # Return bottleneck features

if __name__ == '__main__':
    # Test tensor shapes
    model = ChangeDetectionModel()
    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)
    output = model(t1, t2)
    print(f"Output shape: {output.shape}")
    
    # Check if backward works (sanity check)
    loss = output.sum()
    loss.backward()
    print("Backward pass successful")
