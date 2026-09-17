import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

from config import CLASS_NAMES, UA_LABELS

_MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
_DEFAULT = (os.path.join(_MODELS, 'predictions')
            if os.path.isdir(os.path.join(_MODELS, 'predictions'))
            else os.path.join(_MODELS, 'predictions.npz'))
N_BOOT = 1000
#опорна конфігурація і та, яку з нею порівнюємо по класах
BASELINE = os.environ.get('BASELINE', 'BCE')
TARGET = os.environ.get('TARGET', 'CA-CB-Focal')
_ORDER = ['BCE', 'Focal', 'CB-Focal', 'CA-CB-Focal',
          'CACB_lam0.00', 'CACB_lam0.25', 'CACB_lam0.50', 'CACB_lam1.00']


def _sorted(names):
    known = [n for n in _ORDER if n in names]
    return known + sorted(n for n in names if n not in _ORDER)


def fast_auc(y, s):
    #AUC через ранги: (сума рангів позитивів − n1(n1+1)/2) / (n1·n0)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind='stable')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def macro_auc(Y, P, idx):
    aucs = [fast_auc(Y[idx, c], P[idx, c]) for c in range(Y.shape[1])]
    aucs = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else np.nan


def ci(values, level=95):
    lo = (100 - level) / 2
    return np.nanpercentile(values, lo), np.nanpercentile(values, 100 - lo)


def make_boots(n, n_boot=N_BOOT, seed=42):
    #однакові передвибірки для всіх моделей — основа парного порівняння
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def class_diff_ci(Y, p_target, p_base, c, boots):
    point = fast_auc(Y[:, c], p_target[:, c]) - fast_auc(Y[:, c], p_base[:, c])
    diffs = np.array([fast_auc(Y[b, c], p_target[b, c]) - fast_auc(Y[b, c], p_base[b, c])
                      for b in boots])
    lo, hi = ci(diffs)
    return point, lo, hi


def load_predictions(path):
    if os.path.isdir(path):
        Y = np.load(os.path.join(path, 'Yt.npy'))
        names = [f[2:-4] for f in os.listdir(path) if f.startswith('P_') and f.endswith('.npy')]
        return Y, {c: np.load(os.path.join(path, f'P_{c}.npy')) for c in _sorted(names)}
    d = np.load(path)
    names = [k[2:] for k in d.files if k.startswith('P_')]
    return d['Yt'], {c: d[f'P_{c}'] for c in _sorted(names)}


def main():
    Y, P = load_predictions(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT)
    Y = Y.astype(np.int8)
    n = len(Y)
    print(f"Тестових прикладів: {n};  моделей: {len(P)};  ітерацій bootstrap: {N_BOOT}",
          flush=True)

    boots = make_boots(n)
    all_idx = np.arange(n)

    #macro AUC кожної конфігурації з ДІ
    print("\nmacro AUC (95% довірчий інтервал)", flush=True)
    print(f"{'Конфігурація':14s}{'AUC':>8s}{'95% ДІ':>22s}", flush=True)
    boot_macro = {}
    for c, p in P.items():
        vals = np.array([macro_auc(Y, p, b) for b in boots])
        boot_macro[c] = vals
        lo, hi = ci(vals)
        print(f"{c:14s}{macro_auc(Y, p, all_idx):8.3f}      [{lo:.3f}; {hi:.3f}]", flush=True)

    #парні різниці відносно опорної конфігурації
    if BASELINE in boot_macro:
        print(f"\nРізниця macro AUC відносно {BASELINE} (парний bootstrap)", flush=True)
        for c in boot_macro:
            if c == BASELINE:
                continue
            diffs = boot_macro[c] - boot_macro[BASELINE]
            point = macro_auc(Y, P[c], all_idx) - macro_auc(Y, P[BASELINE], all_idx)
            lo, hi = ci(diffs)
            verdict = 'значуща' if (lo > 0 or hi < 0) else 'НЕ значуща (ДІ містить 0)'
            print(f"{c:14s} Δ={point:+.3f}   [{lo:+.3f}; {hi:+.3f}]   {verdict}", flush=True)

    #по-класова різниця
    if TARGET in P and BASELINE in P:
        print(f"\nAUC по класах: {TARGET} − {BASELINE}", flush=True)
        print(f"{'Клас':20s}{'n+':>6s}{'Δ':>8s}{'95% ДІ':>22s}", flush=True)
        for c in range(Y.shape[1]):
            npos = int(Y[:, c].sum())
            if npos == 0:
                continue
            point, lo, hi = class_diff_ci(Y, P[TARGET], P[BASELINE], c, boots)
            mark = '  ***' if lo > 0 else ('  (гірше)' if hi < 0 else '')
            name = UA_LABELS.get(CLASS_NAMES[c], CLASS_NAMES[c])
            print(f"{name:20s}{npos:6d}{point:+8.3f}      [{lo:+.3f}; {hi:+.3f}]{mark}",
                  flush=True)

    print("\n***  — перевага статистично значуща (нижня межа ДІ > 0)", flush=True)


if __name__ == '__main__':
    main()
