#додаткові метрики з готових прогнозів на тесті, без запуску моделей:
#macro AUC-ROC, macro AUC-PR і ECE (похибка калібрування) для кожної конфігурації,
#а також AUC-PR окремо для кожної патології
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from config import CLASS_NAMES, UA_LABELS

PRED = os.path.join(os.path.dirname(__file__), '..', 'models', 'predictions')
#порядок стовпців у таблицях
ORDER = ['BCE', 'Focal', 'CB-Focal', 'CA-CB-Focal']
#на скільки рівних проміжків ділиться шкала ймовірностей 0..1 для ECE
N_BINS = 15


#ECE (expected calibration error) для одного класу: наскільки ймовірності моделі
#відповідають дійсній частоті хвороби. приклад: серед знімків, яким модель дала близько 0,3,
#хворих має бути близько 30 %. y — мітки, p — ймовірності
def ece(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    total, err = len(y), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        #знімки, чия ймовірність потрапила в проміжок (lo; hi]
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        #різниця між часткою хворих і середньою ймовірністю, зважена часткою знімків у проміжку
        err += m.sum() / total * abs(y[m].mean() - p[m].mean())
    return err


#завантажує прогнози й друкує дві таблиці
def main():
    Y = np.load(os.path.join(PRED, 'Yt.npy'))
    #лише ті конфігурації, для яких є файл прогнозів
    P = {n: np.load(os.path.join(PRED, f'P_{n}.npy'))
         for n in ORDER if os.path.exists(os.path.join(PRED, f'P_{n}.npy'))}
    #класи, у яких у тесті є хоч один хворий (інакше метрики не визначені)
    valid = [c for c in range(Y.shape[1]) if Y[:, c].sum() > 0]

    print(f"{'Конфігурація':16s}{'AUC-ROC':>9s}{'AUC-PR':>9s}{'ECE':>8s}")
    print('-' * 42)
    for n, p in P.items():
        roc = np.mean([roc_auc_score(Y[:, c], p[:, c]) for c in valid])
        #average_precision_score — площа під кривою точність-повнота (AUC-PR)
        pr = np.mean([average_precision_score(Y[:, c], p[:, c]) for c in valid])
        e = np.mean([ece(Y[:, c], p[:, c]) for c in valid])
        print(f'{n:16s}{roc:9.3f}{pr:9.3f}{e:8.3f}')

    print('\nAUC-PR по класах (за спаданням частоти)')
    print(f"{'Патологія':20s}{'n+':>6s}{'база':>8s}" + ''.join(f'{n:>14s}' for n in P))
    for c in sorted(valid, key=lambda c: -Y[:, c].sum()):
        npos = int(Y[:, c].sum())
        base = npos / len(Y)          #AUC-PR випадкового класифікатора = частота класу
        vals = ''.join(f'{average_precision_score(Y[:, c], p[:, c]):14.3f}' for p in P.values())
        print(f'{UA_LABELS[CLASS_NAMES[c]]:20s}{npos:6d}{base:8.3f}{vals}')

    print('\nECE: менше — краще. AUC-PR порівнюйте з колонкою «база»')
    print('(частота класу — рівень випадкового вгадування для цієї метрики).')


if __name__ == '__main__':
    main()
