import torch
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

def evaluate_model(model, dataloader, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "confusion_matrix": cm, "report": report}

def model_latency(model, input_shape=(1,3,224,224), device=None, n_runs=100):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()
    x = torch.randn(input_shape).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    avg_ms = (np.mean(times) * 1000)
    return avg_ms

def model_size_mb(model):
    import tempfile
    import os
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(model.state_dict(), tmp)
    size_mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return size_mb
