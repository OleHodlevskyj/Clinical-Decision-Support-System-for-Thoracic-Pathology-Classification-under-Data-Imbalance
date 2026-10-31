#Обчислення по-класових порогів класифікації за двома критеріями:
#: поріг — це налаштуіндекс Юдена — межа на користь чутливості, знайти якомога більше хворих;
#максимум F1-міри — межа на користь точності, щоб тривогам можна було вірити.
#рахуються на ВАЛІДАЦІЙНІЙ вибірці (Yv.npy, Pv_*.npy),
import os
import json

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, '..', 'models', 'predictions')
OUT = os.path.join(HERE, '..', 'models', 'thresholds.json')
MAIN = 'CA-CB-Focal'


def youden(y, p):
    fpr, tpr, t = roc_curve(y, p)
    return float(t[np.argmax(tpr - fpr)])


def f1_max(y, p):
    pr, rc, t = precision_recall_curve(y, p)
    f = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return float(t[np.argmax(f)])


def bands(Y, P, thr):
    r = (P / np.array(thr)).max(1)
    healthy = Y.sum(1) == 0
    weak = max(1.0, float(np.median(r[healthy])))
    moderate = max(weak, float(np.percentile(r[~healthy], 75)))
    strong = max(moderate, float(np.percentile(r[~healthy], 90)))
    return [round(weak, 2), round(moderate, 2), round(strong, 2)]


def main():
    yv = os.path.join(PRED, 'Yv.npy')
    if not os.path.exists(yv):
        raise SystemExit('Немає Yv.npy')
    Y = np.load(yv)
    res = {'youden': {}, 'f1': {}, 'bands': {}}
    for f in sorted(os.listdir(PRED)):
        if not f.startswith('Pv_'):
            continue
        name = f[3:-4]
        P = np.load(os.path.join(PRED, f))
        for crit, fn in [('youden', youden), ('f1', f1_max)]:
            res[crit][name] = [fn(Y[:, c], P[:, c]) if Y[:, c].sum() > 0 else 0.5
                               for c in range(Y.shape[1])]
        print(f'{name:14s} Юден {min(res["youden"][name]):.3f}–{max(res["youden"][name]):.3f}'
              f'   F1 {min(res["f1"][name]):.3f}–{max(res["f1"][name]):.3f}')

    P = np.load(os.path.join(PRED, f'Pv_{MAIN}.npy'))
    for crit in ('youden', 'f1'):
        res['bands'][crit] = bands(Y, P, res[crit][MAIN])
    print(f'\nмежі сили сигналу ({MAIN}): Юден {res["bands"]["youden"]}   F1 {res["bands"]["f1"]}')

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1)
    print(f'Збережено: {os.path.abspath(OUT)}')


if __name__ == '__main__':
    main()
