#оцінка збережених моделей на тестовій вибірці (без повторного навчання):
#для кожного файлу model_*.pth друкує macro AUC і macro F1 за порогом 0,5
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch.utils.data import DataLoader

from config import CLASS_NAMES
from model import build_model
from dataset import NIHDataset, build_image_index, eval_transform

#ті самі шляхи, що й у train.py
BASE = os.environ.get('NIH_DIR', '/kaggle/input/datasets/organizations/nih-chest-xrays/data')
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'models'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#ймовірності й правильні мітки для всіх знімків (така сама функція, як у train.py)
def predict(model, loader):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


#готує тестову вибірку, завантажує кожну модель і друкує її метрики
def main():
    #мітки 14 класів з таблиці NIH, як у train.prepare_dataframes
    df = pd.read_csv(os.path.join(BASE, 'Data_Entry_2017.csv'))
    for c in CLASS_NAMES:
        df[c] = df['Finding Labels'].apply(lambda s: 1 if c in s.split('|') else 0)
    index = build_image_index(BASE)
    df = df[df['Image Index'].isin(index)].copy()
    #лише офіційний тестовий список
    ts = set(pd.read_csv(os.path.join(BASE, 'test_list.txt'), header=None)[0])
    df_test = df[df['Image Index'].isin(ts)].copy()
    loader = DataLoader(NIHDataset(df_test, index, eval_transform),
                        batch_size=128, num_workers=2, pin_memory=True)

    #назва конфігурації -> файл ваг, як в app.py
    mfiles = {os.path.basename(f)[6:-4]: f
              for f in glob.glob(os.path.join(MODELS_DIR, 'model_*.pth'))}
    assert mfiles, f'Моделі не знайдено в {MODELS_DIR}'

    print(f"{'Конфіг':14s}{'test AUC':>10s}{'test F1':>9s}")
    for name, path in mfiles.items():
        m = build_model(pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        P, Y = predict(m, loader)
        auc = np.mean([roc_auc_score(Y[:, c], P[:, c]) for c in range(14) if Y[:, c].sum() > 0])
        #тут поріг для всіх класів однаковий — 0,5, тому F1 нижчий, ніж з порогами Юдена
        f1 = f1_score(Y, (P > 0.5).astype(int), average='macro', zero_division=0)
        print(f"{name:14s}{auc:10.3f}{f1:9.3f}")


if __name__ == '__main__':
    main()
