import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.compression import DistillationLoss


# Core trainer that performs a single-epoch distillation step and evaluation
class _CoreDistillationTrainer:
    def __init__(self, student, teacher, optimizer, device, temperature=4.0, alpha=0.5):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()

        self.optimizer = optimizer
        self.device = device
        self.criterion = DistillationLoss(temperature, alpha)

    def train_one_epoch(self, dataloader):
        self.student.train()
        epoch_loss = 0

        for X, y in tqdm(dataloader, desc="Distillation Training"):
            X, y = X.to(self.device), y.to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(X)

            student_logits = self.student(X)

            loss = self.criterion(student_logits, teacher_logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.student(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total


# Script-friendly wrapper matching the CLI scripts' expectations
class DistillationTrainer:
    def __init__(self, teacher, student, temperature=4.0, alpha=0.5, lr=1e-3, epochs=10, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        self.core = _CoreDistillationTrainer(student=student, teacher=teacher, optimizer=self.optimizer, device=self.device, temperature=temperature, alpha=alpha)

    def train(self, train_loader, val_loader=None):
        history = []
        for epoch in range(1, self.epochs + 1):
            loss = self.core.train_one_epoch(train_loader)
            val_acc = None
            if val_loader is not None:
                val_acc = self.core.evaluate(val_loader)
                print(f"Epoch {epoch}/{self.epochs} | loss {loss:.4f} | val_acc {val_acc:.4f}")
            else:
                print(f"Epoch {epoch}/{self.epochs} | loss {loss:.4f}")
            history.append({"epoch": epoch, "loss": loss, "val_acc": val_acc})
        return history

    def save(self, out_path):
        try:
            torch.save(self.core.student.state_dict(), out_path)
            print(f"[INFO] Saved distilled student model to: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save distilled model: {e}")
