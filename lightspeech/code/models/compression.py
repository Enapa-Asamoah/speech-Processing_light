import torch
import torch.nn.functional as F

class DistillationLoss(torch.nn.Module):
    """
    Combined soft + hard loss for knowledge distillation.
    """
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        T = self.temperature

        # Soft label loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean"
        ) * (T * T)

        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, targets)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def apply_dynamic_quantization(model):
    """
    Post-training dynamic quantization for linear layers.
    """
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
def apply_static_quantization(model, calibration_loader):
    """
    Post-training static quantization for linear layers.
    """
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibration with representative data
    model.eval()
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x)
    torch.quantization.convert(model, inplace=True)
    return model


def load_teacher_model(num_classes=8, arch="cnn2d", ckpt_path=None, device=None):
    """Factory that returns a teacher model instance.

    - `arch` selects the architecture name supported by `get_model` (default 'cnn2d').
    - If `ckpt_path` is provided, attempt to load weights (state_dict) into the model.
    """
    from .baseline import get_model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(name=arch, num_classes=num_classes)
    if ckpt_path:
        try:
            state = torch.load(ckpt_path, map_location=device)
            # state may be a dict with 'model_state' or direct state_dict
            if isinstance(state, dict) and 'model_state' in state:
                state = state['model_state']
            model.load_state_dict(state)
        except Exception as e:
            print(f"[WARN] Could not load checkpoint '{ckpt_path}': {e}")
    return model