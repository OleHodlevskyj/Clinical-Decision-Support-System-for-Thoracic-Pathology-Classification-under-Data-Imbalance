import os
import sys
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from config import CLASS_NAMES, K_CRITICALITY, BETA, GAMMA0, LAMBDA
from loss import CACBFocalLoss, effective_number_weights
from model import build_resnet50
from dataset import NIHDataset, build_image_index, train_transform, eval_transform

BASE = os.environ.get('NIH_DIR', '/kaggle/input/datasets/organizations/nih-chest-xrays/data')
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'models'))
SUBSET = int(os.environ.get('SUBSET', '0'))
EPOCHS = int(os.environ.get('EPOCHS', '8'))
PATIENCE = int(os.environ.get('PATIENCE', '3'))
SCHEDULER = os.environ.get('SCHEDULER', 'cosine')
SEED = int(os.environ.get('SEED', '42'))
SPLIT_SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_dataframes():
    df = pd.read_csv(os.path.join(BASE, 'Data_Entry_2017.csv'))
    for c in CLASS_NAMES:
        df[c] = df['Finding Labels'].apply(lambda s: 1 if c in s.split('|') else 0)
    index = build_image_index(BASE)
    df = df[df['Image Index'].isin(index)].copy()

    tv = set(pd.read_csv(os.path.join(BASE, 'train_val_list.txt'), header=None)[0])
    ts = set(pd.read_csv(os.path.join(BASE, 'test_list.txt'), header=None)[0])
    df_tv = df[df['Image Index'].isin(tv)].copy()
    df_test = df[df['Image Index'].isin(ts)].copy()
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SPLIT_SEED)
                .split(df_tv, groups=df_tv['Patient ID']))
    return df_tv.iloc[a].copy(), df_tv.iloc[b].copy(), df_test, index


def predict(model, loader):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def youden_thresholds(P, Y):
    th = np.full(P.shape[1], 0.5)
    for c in range(P.shape[1]):
        if Y[:, c].sum() > 0:
            fpr, tpr, t = roc_curve(Y[:, c], P[:, c])
            th[c] = t[np.argmax(tpr - fpr)]
    return th


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df_train, df_val, df_test, index = prepare_dataframes()

    counts = df_train[CLASS_NAMES].sum().values
    w_arr = effective_number_weights(counts, beta=BETA)
    k_arr = [K_CRITICALITY[c] for c in CLASS_NAMES]

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = NIHDataset(df_train, index, train_transform)
    if SUBSET and SUBSET < len(df_train):
        train_ds = Subset(train_ds, random.sample(range(len(df_train)), SUBSET))
    print(f'навчальних: {len(train_ds)}  валідація: {len(df_val)}  тест: {len(df_test)}'
          f'   епох: {EPOCHS}  seed: {SEED}', flush=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(NIHDataset(df_val, index, eval_transform),
                            batch_size=128, num_workers=2, pin_memory=True)
    test_loader = DataLoader(NIHDataset(df_test, index, eval_transform),
                             batch_size=128, num_workers=2, pin_memory=True)

    # (назва, ваги, k, γ0, λ) — абляція
    configs = [
        ('BCE',         [1.0] * 14, [1.0] * 14, 0.0, 0.0),
        ('Focal',       [1.0] * 14, [1.0] * 14, GAMMA0, 0.0),
        ('CB-Focal',    w_arr,      [1.0] * 14, GAMMA0, 0.0),
        ('CA-CB-Focal', w_arr,      k_arr,      GAMMA0, LAMBDA),
    ]

    print(f"{'Конфіг':14s}{'test AUC':>10s}{'test F1':>9s}", flush=True)
    preds = {}
    for name, w, k, g, l in configs:
        model = build_resnet50().to(device)
        crit = CACBFocalLoss(w, k, g, l).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
                 if SCHEDULER == 'cosine' else None)

        ckpt = os.path.join(MODELS_DIR, f'model_{name}.pth')
        best_auc, best_ep = -1.0, 0
        for ep in range(EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
            if sched is not None:
                sched.step()
            # відбір найкращої епохи за валідаційним AUC (захист від перенавчання)
            Pv, Yv = predict(model, val_loader)
            v_auc = np.mean([roc_auc_score(Yv[:, c], Pv[:, c])
                             for c in range(14) if Yv[:, c].sum() > 0])
            mark = ''
            if v_auc > best_auc:
                best_auc, best_ep = v_auc, ep + 1
                torch.save(model.state_dict(), ckpt)
                mark = '  <- best'
            lr_now = opt.param_groups[0]['lr']
            print(f"  [{name}] епоха {ep + 1}/{EPOCHS}  val AUC={v_auc:.4f}"
                  f"  lr={lr_now:.2e}{mark}", flush=True)
            if ep + 1 - best_ep >= PATIENCE:
                print(f"  [{name}] рання зупинка: {PATIENCE} епох без приросту",
                      flush=True)
                break

        # фінальна оцінка — на найкращому чекпойнті
        model.load_state_dict(torch.load(ckpt, map_location=device))
        Pv, Yv = predict(model, val_loader)
        Pt, Yt = predict(model, test_loader)
        th = youden_thresholds(Pv, Yv)
        preds[name] = Pt
        auc = np.mean([roc_auc_score(Yt[:, c], Pt[:, c]) for c in range(14) if Yt[:, c].sum() > 0])
        f1 = f1_score(Yt, (Pt > th).astype(int), average='macro', zero_division=0)
        print(f"{name:14s}{auc:10.3f}{f1:9.3f}   (найкраща епоха: {best_ep})", flush=True)

    pdir = os.path.join(MODELS_DIR, 'predictions')
    os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, 'Yt.npy'), Yt)
    for n, P in preds.items():
        np.save(os.path.join(pdir, f'P_{n}.npy'), P)
    print(f"\nПрогнози збережено: {pdir}", flush=True)

    print("\nAUC по класах (тест): BCE vs CA-CB-Focal", flush=True)
    for c, name in enumerate(CLASS_NAMES):
        if Yt[:, c].sum() == 0:
            continue
        print(f"{name:18s}{int(Yt[:, c].sum()):6d}"
              f"{roc_auc_score(Yt[:, c], preds['BCE'][:, c]):8.3f}"
              f"{roc_auc_score(Yt[:, c], preds['CA-CB-Focal'][:, c]):8.3f}", flush=True)


if __name__ == '__main__':
    main()
