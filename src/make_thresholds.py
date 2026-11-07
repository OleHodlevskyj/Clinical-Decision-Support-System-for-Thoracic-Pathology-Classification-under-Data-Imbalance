#пороги класів для застосунку за двома критеріями:
#індекс Юдена — межа на користь чутливості, знайти якомога більше хворих;
#максимум F1-міри — межа на користь точності, щоб позитивним результатам можна було вірити.
#рахуються на ВАЛІДАЦІЙНІЙ вибірці (Yv.npy, Pv_*.npy): поріг — це налаштування,
#тому тестова вибірка має лишатися недоторканою для оцінки якості.
#результат — файл models/thresholds.json, який читає app.py
import os
import json

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
#звідки брати прогнози і куди записати пороги
PRED = os.path.join(HERE, '..', 'models', 'predictions')
OUT = os.path.join(HERE, '..', 'models', 'thresholds.json')
#модель, для якої рахуються межі сили сигналу (її використовує вкладка «Діагностика»)
MAIN = 'CA-CB-Focal'


#поріг одного класу за індексом Юдена: найбільша різниця між часткою знайдених хворих (tpr)
#і часткою здорових, яких хибно названо хворими (fpr); y — мітки, p — ймовірності
def youden(y, p):
    fpr, tpr, t = roc_curve(y, p)
    return float(t[np.argmax(tpr - fpr)])


#поріг одного класу, за якого F1-міра найбільша
def f1_max(y, p):
    #precision_recall_curve дає точність pr і повноту rc для кожного порога t
    pr, rc, t = precision_recall_curve(y, p)
    #F1 = 2 * точність * повнота / (точність + повнота); останній елемент pr і rc
    #не має свого порога, тому відкидається ([:-1]); clip захищає від ділення на нуль
    f = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return float(t[np.argmax(f)])


#три межі сили сигналу (слабкий, помірний, виражений) для заголовка на вкладці «Діагностика»
def bands(Y, P, thr):
    #для кожного знімка найбільша кратність p / поріг серед 14 класів
    r = (P / np.array(thr)).max(1)
    #здорові — знімки без жодної патології
    healthy = Y.sum(1) == 0
    #слабкий сигнал — вище, ніж у половини здорових знімків (але не менше за 1, тобто за поріг)
    weak = max(1.0, float(np.median(r[healthy])))
    #помірний — вище, ніж у 75 % знімків з патологією, виражений — ніж у 90 %;
    #max гарантує, що межі йдуть за зростанням
    moderate = max(weak, float(np.percentile(r[~healthy], 75)))
    strong = max(moderate, float(np.percentile(r[~healthy], 90)))
    return [round(weak, 2), round(moderate, 2), round(strong, 2)]


#рахує пороги для всіх моделей, що мають файл Pv_*.npy, і межі сили сигналу для MAIN
def main():
    yv = os.path.join(PRED, 'Yv.npy')
    if not os.path.exists(yv):
        raise SystemExit('Немає Yv.npy')
    Y = np.load(yv)
    res = {'youden': {}, 'f1': {}, 'bands': {}}
    for f in sorted(os.listdir(PRED)):
        if not f.startswith('Pv_'):
            continue
        #з Pv_BCE.npy виходить назва BCE
        name = f[3:-4]
        P = np.load(os.path.join(PRED, f))
        #14 порогів кожним із двох способів; клас без жодного хворого отримує 0,5
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
