"""
Baseline emotion recognition model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Option 1: 2D CNN
class CNN2D(nn.Module):
    def __init__(self, num_classes=8, in_channels=3, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, emb_size=128, patch_size=(16,16)):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = None
        self.emb_size = emb_size

    def forward(self, x):
        x = self.proj(x)                   
        B,E,H,W = x.shape
        self.num_patches = H*W
        x = x.flatten(2).transpose(1,2)    
        return x

class LightTransformer(nn.Module):
    def __init__(self, num_classes=8, in_channels=1, emb_size=128, depth=4, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch = PatchEmbedding(in_channels=in_channels, emb_size=emb_size, patch_size=(16,16))
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + 1000, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.to_logits = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, num_classes))

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch(x)  
        B,P,E = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) 
        pos = self.pos_emb[:, :x.size(1), :].to(x.dtype).to(x.device)
        x = x + pos
        x = x.transpose(0,1) 
        x = self.encoder(x)
        x = x.transpose(0,1) 
        cls_out = x[:,0]
        return self.to_logits(cls_out)


# Option 3: Hybrid CNN -> BiLSTM
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=8, in_channels=1, cnn_feat=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, cnn_feat, kernel_size=3, padding=1), nn.BatchNorm2d(cnn_feat), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((None, 64))  
        self.lstm = nn.LSTM(input_size=cnn_feat, hidden_size=lstm_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden*2, num_classes)
        )
        

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x) 
        x = x.mean(dim=2)  
        x = x.permute(0,2,1) 
        out,_ = self.lstm(x)  
        out = out.mean(dim=1)
        return self.classifier(out)


class MobileNetBaseline(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        in_feat = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    

class EfficientNetBaseline(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        in_feat = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# -----------------------
# Factory
# -----------------------
def get_model(name="cnn2d", num_classes=8, in_channels=3, **kwargs):
    name = name.lower()
    if name == "cnn2d":
        return CNN2D(num_classes=num_classes, in_channels=in_channels, **kwargs)
    elif name in ["transformer", "light_transformer"]:
        return LightTransformer(num_classes=num_classes, in_channels=in_channels, **kwargs)
    elif name in ["cnn_lstm", "hybrid"]:
        return CNN_LSTM(num_classes=num_classes, in_channels=in_channels, **kwargs)
    elif name in ["mobilenet", "efficientnet"]:
        # optionally fallback to torchvision backbones:
        if name == "mobilenet":
            model = models.mobilenet_v2(pretrained=True)
            in_feat = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Linear(in_feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))
            return model
        else:
            model = models.efficientnet_b0(pretrained=True)
            in_feat = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Linear(in_feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))
            return model
    else:
        raise ValueError(f"Unknown model: {name}")
