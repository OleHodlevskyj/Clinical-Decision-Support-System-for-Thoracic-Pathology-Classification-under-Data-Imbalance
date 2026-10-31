import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from config import CLASS_NAMES, UA_LABELS
from model import build_model
from dataset import eval_transform

IMG_DIR = os.environ.get('IMG_DIR', '')
NIH_CSV = os.environ.get('NIH_CSV', '')
TEST_LIST = os.environ.get('TEST_LIST', '')
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'models'))
TOPK = int(os.environ.get('TOPK', '3'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_truth():
    """{ім'я файлу: [істинні патології]} із Data_Entry_2017.csv."""
    if not NIH_CSV or not os.path.exists(NIH_CSV):
        return {}
    import pandas as pd
    df = pd.read_csv(NIH_CSV, usecols=['Image Index', 'Finding Labels'])
    return {r['Image Index']: [x for x in r['Finding Labels'].split('|') if x != 'No Finding']
            for _, r in df.iterrows()}


def load_test_split():
    """Множина імен файлів офіційної тестової частини."""
    if not TEST_LIST or not os.path.exists(TEST_LIST):
        return None
    with open(TEST_LIST) as f:
        return {ln.strip() for ln in f if ln.strip()}


def predict_all(model, files):
    """Матриця ймовірностей [кількість знімків x 14]."""
    model.eval()
    batch = torch.stack([eval_transform(Image.open(f).convert('RGB')) for f in files])
    with torch.no_grad():
        return torch.sigmoid(model(batch.to(device))).cpu().numpy()


def main():
    if not IMG_DIR:
        raise SystemExit('Вкажіть IMG_DIR — папку зі знімками.')
    files = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')))
    if not files:
        raise SystemExit(f'У {IMG_DIR} не знайдено файлів .png')

    truth = load_truth()
    test_split = load_test_split()

    print(f'Знімків: {len(files)}   Папка: {IMG_DIR}')
    if truth:
        print(f'Мітки:   завантажено ({len(truth)} записів)')
    else:
        print('Мітки:   НЕМАЄ — вкажіть NIH_CSV, щоб порівняти з істиною')

    if test_split is not None:
        names = [os.path.basename(f) for f in files]
        n_in = sum(n in test_split for n in names)
        print(f'Розбиття: {n_in} із {len(names)} — у тестовій частині '
              f'(модель їх не бачила)')
        if n_in < len(names):
            print('           УВАГА: решта могла бути у навчанні — '
                  'їхні прогнози не є незалежною перевіркою')
    else:
        print('Розбиття: не перевірено — вкажіть TEST_LIST')

    # пацієнти (перші 8 цифр імені файлу) — 5 знімків одного пацієнта
    # не є п'ятьма незалежними спостереженнями
    pids = {os.path.basename(f)[:8] for f in files}
    print(f'Пацієнтів: {len(pids)}')
    if len(files) < 50:
        print('\nЦе якісна перевірка, а не статистична оцінка:')
        print('    для AUC із довірчим інтервалом потрібні сотні знімків')
        print('    від різних пацієнтів. Див. src/external_validate.py\n')

    models = sorted(glob.glob(os.path.join(MODELS_DIR, 'model_*.pth')))
    if not models:
        raise SystemExit(f'Моделі не знайдено в {MODELS_DIR}')

    preds = {}
    for path in models:
        name = os.path.basename(path)[6:-4]
        m = build_model(pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        preds[name] = predict_all(m, files)
        print(f'  прогнано: {name}')

    for i, f in enumerate(files):
        base = os.path.basename(f)
        print(f'\n--- {base} ---')
        if base in truth:
            t = truth[base]
            print('Істина: ' + (', '.join(UA_LABELS.get(x, x) for x in t) if t else 'без патології'))
        for name, P in preds.items():
            top = np.argsort(P[i])[::-1][:TOPK]
            s = ',  '.join(f'{UA_LABELS[CLASS_NAMES[c]]} {P[i, c]:.2f}' for c in top)
            print(f'  {name:20s} {s}')

    # чи потрапила істинна патологія у топ-K — груба міра влучності
    labelled = [i for i, f in enumerate(files) if truth.get(os.path.basename(f))]
    if labelled:
        print(f'\n=== Істинна патологія у топ-{TOPK} (із {len(labelled)} розмічених) ===')
        for name, P in preds.items():
            hits = sum(any(CLASS_NAMES[c] in truth[os.path.basename(files[i])]
                           for c in np.argsort(P[i])[::-1][:TOPK]) for i in labelled)
            print(f'  {name:20s} {hits}/{len(labelled)}')


if __name__ == '__main__':
    main()
