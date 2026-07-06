import torch
import torch.nn as nn
import torchvision.models as models

class UNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(UNetClassifier, self).__init__()
        
        # We start with 3 channels since we converted images to RGB
        self.enc1 = self._conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        # Final conv before pooling
        self.out_conv = nn.Conv2d(32, 32, kernel_size=1)
        
        # Classification Head (Global Average Pooling + Linear)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.upconv3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        
        # Classification
        out = self.global_pool(out) # shape: (B, 32, 1, 1)
        out = out.view(out.size(0), -1) # shape: (B, 32)
        out = self.classifier(out) # shape: (B, num_classes)
        
        return out


def get_resnet18(num_classes=2, pretrained=True):
    # Load pretrained resnet18
    # Using older torchvision API for compatibility: pretrained=True
    model = models.resnet18(pretrained=pretrained)
    
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def get_densenet121(num_classes=2, pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    
    # Modify the final classifier
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
