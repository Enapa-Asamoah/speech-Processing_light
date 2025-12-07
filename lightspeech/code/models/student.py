import torch
import torch.nn as nn
import torchvision.models as models 

class StudentCNN(nn.Module):
    """
    Lightweight student model to be distilled by the baseline teacher.
    Uses MobileNetV2 backbone with reduced width.
    """
    def __init__(self, num_classes=8):
        super().__init__()

        # Smaller backbone
        self.backbone = models.mobilenet_v2(width_mult=0.5)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# Backwards-compatible alias expected by some scripts
StudentModel = StudentCNN

__all__ = ["StudentCNN", "StudentModel"]
