#Донавчання (доменна адаптація) на датасеті дитячої пневмонії.
# Стартує з навченої моделі CA-CB-Focal (NIH) і донавчає її на цільовому
# датасеті, супервізуючи лише вихід «Пневмонія». Демонструє усунення
# зсуву домену: AUC на тесті пневмонії зростає ~0.54 -> ~0.97.
# Змінні середовища:
# PNEUMONIA_DIR  — корінь датасету chest-xray-pneumonia
# MODELS_DIR     — папка з model_CA-CB-Focal.pth

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score

from config import CLASS_NAMES
from model import build_resnet50
from dataset import train_transform, eval_transform

PB = os.environ.get('PNEUMONIA_DIR', '/kaggle/input/datasets/paultimothymooney')
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'models'))
PN = CLASS_NAMES.index('Pneumonia')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect(split):
    nrm = {os.path.basename(f): f
           for f in glob.glob(f'{PB}/**/{split}/NORMAL/*.jpeg', recursive=True)}
    pnu = {os.path.basename(f): f
           for f in glob.glob(f'{PB}/**/{split}/PNEUMONIA/*.jpeg', recursive=True)}
    return list(nrm.values()) + list(pnu.values()), [0] * len(nrm) + [1] * len(pnu)


class PneumoniaDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files, self.labels, self.tf = files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.tf(Image.open(self.files[i]).convert('RGB'))
        return img, torch.tensor(float(self.labels[i]))


def pneumonia_auc(model, loader):
    model.eval()
    scores, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            scores.append(torch.sigmoid(model(x.to(device)))[:, PN].cpu().numpy())
            ys.append(y.numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(scores))


def main(epochs=3):
    train_f, train_y = collect('train')
    test_f, test_y = collect('test')
    print(f"train: {len(train_f)}  test: {len(test_f)}")

    train_loader = DataLoader(PneumoniaDataset(train_f, train_y, train_transform),
                              batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(PneumoniaDataset(test_f, test_y, eval_transform),
                             batch_size=64, num_workers=2, pin_memory=True)

    # стартуємо з навченої моделі CA-CB-Focal
    model = build_resnet50(pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'model_CA-CB-Focal.pth'),
                                     map_location=device))

    print(f"AUC ДО донавчання: {pneumonia_auc(model, test_loader):.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            bce(model(x)[:, PN], y).backward()
            opt.step()
        print(f"  епоха {ep + 1}/{epochs} -> AUC: {pneumonia_auc(model, test_loader):.3f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'model_pneumonia_adapted.pth'))
    print(f">>> AUC ПІСЛЯ донавчання: {pneumonia_auc(model, test_loader):.3f}")


if __name__ == '__main__':
    main()
