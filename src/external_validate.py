#зовнішня перевірка: моделі, навчені на NIH, без донавчання оцінюються на чужому наборі
#знімків (типово — дитяча пневмонія). друкує AUC з довірчим інтервалом, чутливість
#і специфічність для одного класу. налаштування — змінними середовища
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from config import CLASS_NAMES
from model import build_resnet50
from dataset import eval_transform

#папка з набором і частина (test або train)
EXT_DIR = os.environ.get('EXT_DIR', '/kaggle/input/chest-xray-pneumonia/chest_xray')
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'models'))
SPLIT = os.environ.get('SPLIT', 'test')
#назви підпапок з хворими і здоровими
POSITIVE = os.environ.get('POSITIVE', 'PNEUMONIA')
NEGATIVE = os.environ.get('NEGATIVE', 'NORMAL')
#який із 14 виходів моделі оцінювати
TARGET = os.environ.get('TARGET', 'Pneumonia')
#розширення файлів знімків
FILE_EXT = os.environ.get('EXT', 'jpeg')
N_BOOT = 1000
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#шляхи до знімків і мітки: спершу здорові (0), потім хворі (1)
def collect():
    #усі файли з підпапки sub частини SPLIT; словник за назвою прибирає дублікати
    def grab(sub):
        return {os.path.basename(f): f for f in
                glob.glob(f'{EXT_DIR}/**/{SPLIT}/{sub}/*.{FILE_EXT}', recursive=True)}
    neg, pos = grab(NEGATIVE), grab(POSITIVE)
    #без хворих або без здорових AUC не порахувати, тому зупиняємось з підказкою
    if not neg or not pos:
        raise SystemExit(
            f'Не знайдено знімків у {EXT_DIR} (split={SPLIT}, {NEGATIVE}/{POSITIVE}, *.{FILE_EXT}).\n'
            f'Знайдено: {NEGATIVE}={len(neg)}, {POSITIVE}={len(pos)}. '
            f'Перевірте EXT_DIR, SPLIT та EXT.')
    files = list(neg.values()) + list(pos.values())
    labels = np.array([0] * len(neg) + [1] * len(pos))
    return files, labels


#набір даних, що повертає лише підготовлені знімки (мітки зберігаються окремо)
class FolderDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return eval_transform(Image.open(self.files[i]).convert('RGB'))


#ймовірності цільового класу для всіх знімків
def scores_for(model, loader, cls_idx):
    model.eval()
    out = []
    with torch.no_grad():
        for x in loader:
            out.append(torch.sigmoid(model(x.to(device)))[:, cls_idx].cpu().numpy())
    return np.concatenate(out)


#швидкий AUC через ранги, як у bootstrap_ci.py
def fast_auc(y, s):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind='stable')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


#95 % довірчий інтервал AUC: n разів AUC на випадковій вибірці з поверненням
def bootstrap_ci(y, s, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a = fast_auc(y[idx], s[idx])
        if not np.isnan(a):
            vals.append(a)
    return np.percentile(vals, [2.5, 97.5])


#чутливість і специфічність у точці максимуму індексу Юдена
def operating_point(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.argmax(tpr - fpr))
    #специфічність — частка правильно названих здоровими, тобто 1 - fpr
    return tpr[i], 1 - fpr[i]


#оцінює кожен файл model_*.pth і друкує таблицю
def main():
    cls_idx = CLASS_NAMES.index(TARGET)
    files, y = collect()
    loader = DataLoader(FolderDataset(files), batch_size=64, num_workers=2, pin_memory=True)

    print(f'Набір:  {EXT_DIR}')
    print(f'Вибірка: {SPLIT} — {len(files)} знімків '
          f'({int((y == 0).sum())} {NEGATIVE} / {int(y.sum())} {POSITIVE})')
    print(f'Клас:   {TARGET} (вихід №{cls_idx})\n')

    models = sorted(glob.glob(os.path.join(MODELS_DIR, 'model_*.pth')))
    if not models:
        raise SystemExit(f'Моделі не знайдено в {MODELS_DIR}')

    print(f"{'Модель':26s}{'AUC':>7s}{'95 % ДІ':>18s}{'чутл.':>8s}{'специф.':>9s}")
    print('-' * 68)
    for path in models:
        name = os.path.basename(path)[6:-4]
        m = build_resnet50(pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        s = scores_for(m, loader, cls_idx)
        auc = roc_auc_score(y, s)
        lo, hi = bootstrap_ci(y, s)
        sens, spec = operating_point(y, s)
        print(f'{name:26s}{auc:7.3f}   [{lo:.3f}; {hi:.3f}]{sens:8.3f}{spec:9.3f}')

    print('\nAUC = 0,5 — рівень випадкового вгадування.')
    print('Довірчий інтервал, що містить 0,5, означає: перенесення не доведено.')


if __name__ == '__main__':
    main()
