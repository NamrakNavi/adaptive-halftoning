import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    # U-Net для генерации полутоновых изображений
    
    def __init__(self, in_channels=3, out_channels=1, use_sigmoid=False):
        super(UNet, self).__init__()
        self.use_sigmoid = use_sigmoid  # по умолчанию False для полутонов
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder с билинейной интерполяцией вместо транспонированной свертки
        self.upconv4 = self.upsample_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = self.upsample_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = self.upsample_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = self.upsample_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        
        # Выходной слой (без сигмоиды для полутонов)
        self.outconv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        # Опциональная сигмоида (если нужны бинарные выходы)
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        # Обрезаем до размера enc4 при необходимости
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear')
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear')
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear')
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear')
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Выход
        out = self.outconv(dec1)
        
        # Нормализуем выход в диапазон [0, 1]
        out_min = out.min()
        out_max = out.max()
        if out_max - out_min > 0:
            out = (out - out_min) / (out_max - out_min)
        
        # Применяем сигмоиду только если нужно бинарное изображение
        if self.sigmoid is not None:
            out = self.sigmoid(out)
        
        return out