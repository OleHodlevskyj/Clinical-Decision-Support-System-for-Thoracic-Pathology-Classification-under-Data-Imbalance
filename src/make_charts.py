#графіки для звіту та презентації; усі числа рахуються зі збережених прогнозів
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from config import CLASS_NAMES, UA_LABELS, K_CRITICALITY
from bootstrap_ci import make_boots, class_diff_ci

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')
PRED = os.path.join(HERE, '..', 'models', 'predictions')
PRED_LAMBDA = os.path.join(HERE, '..', 'models', 'predictions_lambda')
#умови λ-експерименту у файлах прогнозів не зберігаються
LAMBDA_RUN = 'підвибірка 20 000, 5 епох'

GRAY, ORANGE, TEAL, RED = '#9aa7b5', '#e2711d', '#0f766e', '#c0392b'
ORDER = ['BCE', 'Focal', 'CB-Focal', 'CA-CB-Focal']
HERNIA = CLASS_NAMES.index('Hernia')
PNEUMO = CLASS_NAMES.index('Pneumothorax')


def load(path=PRED, names=ORDER):
    Y = np.load(os.path.join(path, 'Yt.npy'))
    P = {n: np.load(os.path.join(path, 'P_' + n + '.npy')) for n in names}
    return Y, P, [c for c in range(Y.shape[1]) if Y[:, c].sum() > 0]


def _macro_roc(Y, p, valid):
    return np.mean([roc_auc_score(Y[:, c], p[:, c]) for c in valid])


def _padded(values, pad):
    #межі осі за даними, щоб стовпці й підписи не виходили за графік
    lo, hi = min(values), max(values)
    return lo - (hi - lo) * pad, hi + (hi - lo) * pad


def _style(ax):
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)


def chart_ablation():
    Y, P, valid = load()
    roc = [_macro_roc(Y, P[n], valid) for n in ORDER]
    pr = [np.mean([average_precision_score(Y[:, c], P[n][:, c]) for c in valid]) for n in ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, title in [(axes[0], roc, 'macro AUC-ROC'), (axes[1], pr, 'macro AUC-PR')]:
        lim = _padded(vals, 0.5)
        bars = ax.bar(ORDER, vals, color=[TEAL] + [GRAY] * 2 + [ORANGE], width=0.6, zorder=3)
        ax.set_ylim(*lim)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.yaxis.grid(True, color='#e6e6e6')
        ax.tick_params(axis='x', labelrotation=20, labelsize=9)
        _style(ax)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (lim[1] - lim[0]) * 0.02,
                    '%.3f' % v, ha='center', fontsize=9.5)
    fig.suptitle('Порівняння конфігурацій функції втрат (тест)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'chart_ablation.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def chart_hernia_pr():
    Y, P, _ = load()
    base = Y[:, HERNIA].mean()  #рівень випадкового вгадування
    vals = [average_precision_score(Y[:, HERNIA], P[n][:, HERNIA]) for n in ORDER]
    off = max(vals) * 0.025

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(ORDER, vals, color=[TEAL] + [GRAY] * 2 + [ORANGE], width=0.55, zorder=3)
    ax.axhline(base, color=RED, ls='--', lw=1.4, zorder=2)
    ax.text(len(ORDER) - 0.55, base + off, 'випадкове вгадування: %.3f' % base,
            ha='right', fontsize=8.5, color=RED)
    ax.set_ylabel('AUC-PR')
    ax.set_title('%s (n = %d) — найрідкісніша патологія'
                 % (UA_LABELS['Hernia'], int(Y[:, HERNIA].sum())),
                 fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, color='#e6e6e6')
    ax.tick_params(axis='x', labelrotation=15)
    _style(ax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + off, '%.3f' % v, ha='center', fontsize=10)
    #кратність відносно базової моделі — всередині стовпця
    for b, v in zip(bars[1:], vals[1:]):
        ax.text(b.get_x() + b.get_width() / 2, v * 0.5,
                'x%.1f' % (v / vals[0]), ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
    ax.text(0.02, 0.93, 'CA-CB-Focal: у %.1f раза вище за базову'
            % (vals[ORDER.index('CA-CB-Focal')] / vals[0]),
            transform=ax.transAxes, fontsize=10.5, fontweight='bold', color=ORANGE)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'chart_hernia_pr.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def chart_forest():
    #парний bootstrap, як у bootstrap_ci.py — рахується близько хвилини
    Y, P, valid = load(names=['BCE', 'CA-CB-Focal'])
    Y = Y.astype(np.int8)
    boots = make_boots(len(Y))
    rows = []
    for c in valid:
        d, lo, hi = class_diff_ci(Y, P['CA-CB-Focal'], P['BCE'], c, boots)
        rows.append((UA_LABELS[CLASS_NAMES[c]], int(Y[:, c].sum()), d, lo, hi))
    rows.sort(key=lambda r: -r[2])

    labels = ['%s (n=%d)' % (name, k) for name, k, *_ in rows]
    d, lo, hi = (np.array([r[i] for r in rows]) for i in (2, 3, 4))
    y = np.arange(len(rows))
    colors = [ORANGE if l >= 0 else (GRAY if h > 0 else '#8fa0b0') for l, h in zip(lo, hi)]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.axvline(0, color='#3d4b58', lw=1.2, zorder=2)
    for i in range(len(rows)):
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=colors[i], lw=2.4,
                solid_capstyle='round', zorder=3)
    ax.scatter(d, y, s=46, color=colors, zorder=4, edgecolor='white', linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('Δ AUC-ROC (CA-CB-Focal − BCE) з 95 % довірчим інтервалом')
    ax.set_title('По-класова різниця якості', fontsize=13, fontweight='bold')
    ax.xaxis.grid(True, color='#eef1f4')
    ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'chart_forest.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def chart_lambda():
    #окремий контрольований експеримент: між прогонами змінюється лише λ
    Y = np.load(os.path.join(PRED_LAMBDA, 'Yt.npy'))
    valid = [c for c in range(Y.shape[1]) if Y[:, c].sum() > 0]
    prefix = 'P_CACB_lam'
    runs = sorted(((float(f[len(prefix):-4]), np.load(os.path.join(PRED_LAMBDA, f)))
                   for f in os.listdir(PRED_LAMBDA) if f.startswith(prefix)),
                  key=lambda r: r[0])
    lam = [l for l, _ in runs]
    hernia = [roc_auc_score(Y[:, HERNIA], p[:, HERNIA]) for _, p in runs]
    pneumo = [roc_auc_score(Y[:, PNEUMO], p[:, PNEUMO]) for _, p in runs]
    macro = [_macro_roc(Y, p, valid) for _, p in runs]
    bce = _macro_roc(Y, np.load(os.path.join(PRED_LAMBDA, 'P_BCE.npy')), valid)
    lim = _padded(hernia + pneumo + macro + [bce], 0.15)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(lam, hernia, 'o-', color=ORANGE, lw=2, ms=8, zorder=3,
            label='%s (n=%d)' % (UA_LABELS['Hernia'], int(Y[:, HERNIA].sum())))
    ax.plot(lam, pneumo, 's-', color=RED, lw=2, ms=7, zorder=3,
            label='%s (k_c = %s)' % (UA_LABELS['Pneumothorax'],
                                     ('%.1f' % K_CRITICALITY['Pneumothorax']).replace('.', ',')))
    ax.plot(lam, macro, '^-', color=TEAL, lw=2, ms=7, zorder=3,
            label='macro AUC (усі %d класів)' % len(valid))
    ax.axhline(bce, color=GRAY, ls='--', lw=1.5, zorder=2)
    ax.text(lam[-1], bce + (lim[1] - lim[0]) * 0.015,
            'BCE: macro %s' % ('%.3f' % bce).replace('.', ','),
            ha='right', fontsize=8.5, color='#5b6874')
    ax.set_xlabel('λ — вплив клінічної критичності на фокусування')
    ax.set_ylabel('AUC-ROC (тест)')
    ax.set_title('Компроміс, керований параметром λ\n(%s)' % LAMBDA_RUN,
                 fontsize=12, fontweight='bold')
    ax.set_xticks(lam)
    ax.set_ylim(*lim)
    ax.grid(True, color='#eef1f4')
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc='center left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'chart_lambda.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    chart_ablation()
    chart_hernia_pr()
    chart_forest()
    chart_lambda()
    print('Графіки збережено в', os.path.abspath(OUT))
