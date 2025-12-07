import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss * self.alpha
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

# Early stopping helper
class EarlyStopping:
    def __init__(self, patience=10, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False
        improve = (value > self.best) if self.mode == 'max' else (value < self.best)
        if improve:
            self.best = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Trainer class
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, cfg, device=None, collate_fn=None):
        self.cfg = cfg
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn

        self.batch_size = cfg['training']['batch_size']
        self.num_epochs = cfg['training']['num_epochs']
        self.lr = cfg['training']['learning_rate']
        self.weight_decay = cfg['training'].get('weight_decay', 0.0)

        # dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=cfg['hardware']['num_workers'], pin_memory=cfg['hardware'].get('pin_memory', True),
                                       collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=cfg['hardware']['num_workers'], pin_memory=cfg['hardware'].get('pin_memory', True),
                                     collate_fn=collate_fn)

        # loss
        loss_name = cfg['training'].get('loss', 'cross_entropy')
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == 'focal_loss':
            self.criterion = FocalLoss(alpha=cfg['training'].get('focal_alpha', 1.0),
                                       gamma=cfg['training'].get('focal_gamma', 2.0))
        else:
            raise ValueError("Unsupported loss")

        # optimizer
        opt_name = cfg['training'].get('optimizer', 'adam')
        if opt_name.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # scheduler
        sched_name = cfg['training'].get('scheduler', 'cosine')
        if sched_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        elif sched_name == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg['training'].get('step_size',10), gamma=0.1)
        elif sched_name == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5)
        else:
            self.scheduler = None

        # extras
        self.best_val = -np.inf
        self.checkpoint_dir = cfg['training']['checkpoint']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_best = cfg['training']['checkpoint'].get('save_best', True)
        self.save_freq = cfg['training']['checkpoint'].get('save_frequency', 5)
        self.early_stopper = EarlyStopping(patience=cfg['training']['early_stopping'].get('patience', 10),
                                           mode=cfg['training']['early_stopping'].get('mode', 'max'))

        self.use_amp = cfg['hardware'].get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # optional experiment tracking (W&B / MLflow)
        self.tracker_cfg = cfg.get('tracking', {})
        self.wandb = None
        if self.tracker_cfg.get('use_wandb', False):
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=self.tracker_cfg.get('project_name','lightspeech'),
                           name=self.tracker_cfg.get('experiment_name','run'))
                wandb.watch(self.model, log="gradients", log_freq=100)
            except Exception as e:
                print("W&B init failed:", e)

    def train_epoch(self):
        self.model.train()
        losses = []
        preds_all, labels_all = [], []
        loop = tqdm(self.train_loader, desc="Train", leave=False)
        for batch in loop:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            losses.append(loss.item())
            preds_all.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            labels_all.extend(y.detach().cpu().numpy())
        avg_loss = np.mean(losses)
        acc = accuracy_score(labels_all, preds_all)
        return avg_loss, acc

    def validate(self):
        self.model.eval()
        losses = []
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val", leave=False):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                losses.append(loss.item())
                preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels_all.extend(y.cpu().numpy())
        avg_loss = np.mean(losses)
        acc = accuracy_score(labels_all, preds_all)
        f1 = f1_score(labels_all, preds_all, average='weighted')
        return avg_loss, acc, f1

    def save_checkpoint(self, epoch, name="model"):
        path = os.path.join(self.checkpoint_dir, f"{name}_epoch{epoch}.pth")
        torch.save(self.model.state_dict(), path)

    def save_best_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save(self.model.state_dict(), path)

    def train(self):
        logs = []
        for epoch in range(1, self.num_epochs+1):
            start = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            epoch_time = time.time() - start

            # scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            elif self.scheduler:
                self.scheduler.step()

            # logging
            print(f"Epoch {epoch}/{self.num_epochs} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_f1 {val_f1:.4f} | time {epoch_time:.1f}s")
            if self.wandb:
                self.wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "epoch": epoch})

            # checkpointing
            if self.save_best and val_acc > self.best_val:
                self.best_val = val_acc
                self.save_best_checkpoint()

            if epoch % self.save_freq == 0:
                self.save_checkpoint(epoch)

            # early stopping
            if self.early_stopper.step(val_acc):
                print("Early stopping triggered.")
                break

            logs.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
        return logs


# Backwards-compatible simple trainer used by scripts
class BaselineTrainer:
    def __init__(self, model, lr=1e-4, epochs=10, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_loader, val_loader=None):
        history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            losses = []
            preds, labels = [], []
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(y.cpu().numpy())
            train_loss = float(np.mean(losses)) if losses else 0.0
            train_acc = float(accuracy_score(labels, preds)) if labels else 0.0
            val_loss, val_acc = 0.0, 0.0
            val_f1 = 0.0
            if val_loader is not None:
                self.model.eval()
                vlosses, vpreds, vlabels = [], [], []
                with torch.no_grad():
                    for x, y in tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False):
                        x, y = x.to(self.device), y.to(self.device)
                        logits = self.model(x)
                        loss = self.criterion(logits, y)
                        vlosses.append(loss.item())
                        vpreds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        vlabels.extend(y.cpu().numpy())
                val_loss = float(np.mean(vlosses)) if vlosses else 0.0
                val_acc = float(accuracy_score(vlabels, vpreds)) if vlabels else 0.0
                val_f1 = float(f1_score(vlabels, vpreds, average='weighted')) if vlabels else 0.0
                print(f"Epoch {epoch}/{self.epochs} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_f1 {val_f1:.4f}")
            else:
                print(f"Epoch {epoch}/{self.epochs} | train_loss {train_loss:.4f} train_acc {train_acc:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            })
        return history

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "baseline_model.pth")
        torch.save(self.model.state_dict(), out_path)
        print(f"[INFO] Saved baseline model to: {out_path}")
