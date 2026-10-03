import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from config import CLASS_NAMES, UA_LABELS

PRED = os.path.join(os.path.dirname(__file__), '..', 'models', 'predictions')
ORDER = ['BCE', 'Focal', 'CB-Focal', 'CA-CB-Focal']
N_BINS = 15


def ece(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    total, err = len(y), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        err += m.sum() / total * abs(y[m].mean() - p[m].mean())
    return err


def main():
    Y = np.load(os.path.join(PRED, 'Yt.npy'))
    P = {n: np.load(os.path.join(PRED, f'P_{n}.npy'))
         for n in ORDER if os.path.exists(os.path.join(PRED, f'P_{n}.npy'))}
    valid = [c for c in range(Y.shape[1]) if Y[:, c].sum() > 0]

    print(f"{'Конфігурація':16s}{'AUC-ROC':>9s}{'AUC-PR':>9s}{'ECE':>8s}")
    print('-' * 42)
    for n, p in P.items():
        roc = np.mean([roc_auc_score(Y[:, c], p[:, c]) for c in valid])
        pr = np.mean([average_precision_score(Y[:, c], p[:, c]) for c in valid])
        e = np.mean([ece(Y[:, c], p[:, c]) for c in valid])
        print(f'{n:16s}{roc:9.3f}{pr:9.3f}{e:8.3f}')

    print('\nAUC-PR по класах (за спаданням частоти)')
    print(f"{'Патологія':20s}{'n+':>6s}{'база':>8s}" + ''.join(f'{n:>14s}' for n in P))
    for c in sorted(valid, key=lambda c: -Y[:, c].sum()):
        npos = int(Y[:, c].sum())
        base = npos / len(Y)          # AUC-PR випадкового класифікатора = частота класу
        vals = ''.join(f'{average_precision_score(Y[:, c], p[:, c]):14.3f}' for p in P.values())
        print(f'{UA_LABELS[CLASS_NAMES[c]]:20s}{npos:6d}{base:8.3f}{vals}')

    print('\nECE: менше — краще. AUC-PR порівнюйте з колонкою «база»')
    print('(частота класу — рівень випадкового вгадування для цієї метрики).')


if __name__ == '__main__':
    main()
