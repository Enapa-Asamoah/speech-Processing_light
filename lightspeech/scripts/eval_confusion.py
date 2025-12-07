import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import torch
from lightspeech.code.models.compression import load_teacher_model
from lightspeech.code.data.loader import get_dataloaders
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def evaluate():
    print('Loading dataloaders...')
    train_dl, val_dl, test_dl = get_dataloaders('data/processed', batch_size=32, arch='cnn2d')
    print('Loading model...')
    model = load_teacher_model(num_classes=8, arch='cnn2d', ckpt_path='results/models/baseline_model.pth', device='cpu')
    model.eval()
    def run_on(loader, name):
        ys = []
        yps = []
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                ys.extend(yb.cpu().numpy().tolist())
                yps.extend(preds.tolist())
        ys = np.array(ys)
        yps = np.array(yps)
        print(f"\n{name} samples: {len(ys)}")
        print('unique preds:', np.unique(yps, return_counts=True))
        cm = confusion_matrix(ys, yps)
        print('Confusion matrix:\n', cm)
        print('\nClassification report:\n')
        print(classification_report(ys, yps, digits=4, zero_division=0))

    run_on(train_dl, 'TRAIN')
    run_on(val_dl, 'VAL')
    run_on(test_dl, 'TEST')

if __name__ == '__main__':
    evaluate()
