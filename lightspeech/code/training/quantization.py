import torch
import torch.nn as nn

class QATTrainer:
    """
    Quantization-Aware Training for student model.
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def prepare_qat(self):
        # Insert fake quantization modules
        self.model.train()
        self.model.fuse_model() if hasattr(self.model, "fuse_model") else None
        self.qat_model = torch.quantization.prepare_qat(self.model, inplace=False)
        return self.qat_model

    def train_one_epoch(self, dataloader):
        self.qat_model.train()
        epoch_loss = 0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            logits = self.qat_model(X)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def convert(self):
        return torch.quantization.convert(self.qat_model.eval(), inplace=False)


# Simple script-friendly wrapper expected by CLI scripts
class QuantizationTrainer:
    def __init__(self, model, backend="qnnpack", device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.backend = backend
        self.qmodel = None

    def quantize(self):
        """Apply post-training dynamic quantization (safe and fast)."""
        try:
            # quantize linear layers dynamically
            self.qmodel = torch.quantization.quantize_dynamic(
                self.model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
            )
            print("[INFO] Applied dynamic quantization to model (CPU).")
        except Exception as e:
            print(f"[WARN] Dynamic quantization failed: {e}")
            self.qmodel = self.model
        return self.qmodel

    def save(self, out_path, sample_input_shape=(1,3,224,224)):
        """Save quantized model. If `out_path` ends with .onnx, attempt to export to ONNX using a dummy input; otherwise save state_dict."""
        if self.qmodel is None:
            print("[WARN] Model not quantized yet; saving original model.")
            model_to_save = self.model
        else:
            model_to_save = self.qmodel

        try:
            if str(out_path).lower().endswith('.onnx'):
                dummy = torch.randn(*sample_input_shape)
                try:
                    torch.onnx.export(model_to_save.eval(), dummy, out_path)
                    print(f"[INFO] Exported model to ONNX: {out_path}")
                    return out_path
                except Exception as e:
                    print(f"[WARN] ONNX export failed: {e}; falling back to saving a PyTorch checkpoint (.pt)")
                    # Avoid saving a state_dict to a .onnx filename — use .pt instead
                    fallback = str(out_path).rstrip('.onnx') + '_fallback.pt'
                    torch.save(model_to_save.state_dict(), fallback)
                    print(f"[INFO] Saved model state_dict to fallback path: {fallback}")
                    return fallback

            # If out_path doesn't request ONNX, save a proper .pt checkpoint
            torch.save(model_to_save.state_dict(), out_path)
            print(f"[INFO] Saved model state_dict to: {out_path}")
            return out_path
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            return None
