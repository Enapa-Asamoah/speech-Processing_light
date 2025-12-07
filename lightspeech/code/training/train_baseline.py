import yaml
import os
from pathlib import Path
import torch
from code.models.baseline import get_model
from code.training.trainer import Trainer
from code.data.loader import get_dataloaders
from torchvision import transforms

# Load config
cfg_path = "configs/baseline_.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# dataloaders (assumes PNG features saved under data/features/<split>)
feature_dir_train = os.path.join(cfg['dataset']['features_dir'], 'train')
feature_dir_val = os.path.join(cfg['dataset']['features_dir'], 'val')

# image transforms for pretrained/backbone models
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_loader, val_loader, test_loader = get_dataloaders(cfg['dataset']['features_dir'], batch_size=cfg['training']['batch_size'])

# build model
model = get_model(cfg['model']['name'], num_classes=cfg['model']['architecture']['num_classes'], in_channels=3)

# wrap trainer
trainer = Trainer(model=model, train_dataset=train_loader.dataset, val_dataset=val_loader.dataset, cfg=cfg)
logs = trainer.train()

# save logs
import json
os.makedirs("results/logs", exist_ok=True)
with open("results/logs/train_log.json", "w") as f:
    json.dump(logs, f)
print("Training finished. Best model saved to:", cfg['training']['checkpoint']['save_dir'])
