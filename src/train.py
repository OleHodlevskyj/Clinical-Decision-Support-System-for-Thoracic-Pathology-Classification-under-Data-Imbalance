#навчання чотирьох моделей для абляції (BCE, Focal, CB-Focal, CA-CB-Focal) на однакових даних,
#відбір найкращої епохи за валідацією і збереження ваг та прогнозів.
#запускався на Kaggle з GPU; налаштування передаються змінними середовища
import os
import sys
import random

#щоб import config, loss, model, dataset знаходили сусідні файли з папки src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from config import CLASS_NAMES, K_CRITICALITY, BETA, GAMMA0, LAMBDA
from loss import CACBFocalLoss, effective_number_weights
from model import build_model
from dataset import NIHDataset, build_image_index, train_transform, eval_transform

#налаштування запуску; кожне можна змінити змінною середовища без правки коду
#папка з набором даних NIH (типово — шлях на Kaggle)
BASE = os.environ.get('NIH_DIR', '/kaggle/input/datasets/organizations/nih-chest-xrays/data')
#куди зберігати ваги й прогнози
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'models'))
#якщо не 0 — навчатися лише на стількох випадкових знімках (для швидких пробних запусків)
SUBSET = int(os.environ.get('SUBSET', '0'))
#найбільша кількість епох, тобто повних проходів по навчальній вибірці
EPOCHS = int(os.environ.get('EPOCHS', '8'))
#рання зупинка: скільки епох поспіль можна не покращуватись
PATIENCE = int(os.environ.get('PATIENCE', '3'))
#cosine — швидкість навчання плавно зменшується до кінця навчання, інше значення — стала
SCHEDULER = os.environ.get('SCHEDULER', 'cosine')
#зерно випадковості: з тим самим зерном навчання повторюється однаково
SEED = int(os.environ.get('SEED', '42'))
#скільки паралельних процесів читають знімки з диска
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', '4'))
#зерно поділу на навчальну й валідаційну частини окреме і стале,
#щоб за різних SEED поділ лишався тим самим
SPLIT_SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#читає таблицю міток NIH і ділить знімки на навчальні, валідаційні й тестові;
#повертає три таблиці і словник шляхів до файлів
def prepare_dataframes():
    df = pd.read_csv(os.path.join(BASE, 'Data_Entry_2017.csv'))
    #у стовпці Finding Labels діагнози записано через «|», наприклад «Effusion|Mass»;
    #для кожної патології створюємо окремий стовпець з 1 або 0
    for c in CLASS_NAMES:
        df[c] = df['Finding Labels'].apply(lambda s: 1 if c in s.split('|') else 0)
    index = build_image_index(BASE)
    #лишаємо лише ті рядки, для яких файл знімка справді є на диску
    df = df[df['Image Index'].isin(index)].copy()

    #офіційний поділ авторів набору: train_val_list — для навчання, test_list — для тесту
    tv = set(pd.read_csv(os.path.join(BASE, 'train_val_list.txt'), header=None)[0])
    ts = set(pd.read_csv(os.path.join(BASE, 'test_list.txt'), header=None)[0])
    df_tv = df[df['Image Index'].isin(tv)].copy()
    df_test = df[df['Image Index'].isin(ts)].copy()
    #15 % навчальної частини йде на валідацію; поділ за пацієнтами (groups),
    #щоб знімки однієї людини не потрапили водночас у навчання й валідацію
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SPLIT_SEED)
                .split(df_tv, groups=df_tv['Patient ID']))
    return df_tv.iloc[a].copy(), df_tv.iloc[b].copy(), df_test, index


#проганяє всі знімки з loader через модель; повертає таблицю ймовірностей
#(рядок — знімок, стовпець — патологія) і таблицю правильних міток того самого розміру
def predict(model, loader):
    model.eval()
    ps, ys = [], []
    #градієнти не потрібні, бо ваги тут не змінюються
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            ys.append(y.numpy())
    #склеюємо відповіді всіх пакетів в одну таблицю
    return np.concatenate(ps), np.concatenate(ys)


#пороги за індексом Юдена: для кожного класу поріг, за якого різниця
#«частка знайдених хворих мінус частка хибнопозитивних» найбільша
def youden_thresholds(P, Y):
    th = np.full(P.shape[1], 0.5)
    for c in range(P.shape[1]):
        #якщо у вибірці немає жодного хворого з цим класом, лишається 0,5
        if Y[:, c].sum() > 0:
            #roc_curve перебирає всі можливі пороги t і для кожного дає tpr і fpr
            fpr, tpr, t = roc_curve(Y[:, c], P[:, c])
            th[c] = t[np.argmax(tpr - fpr)]
    return th


#увесь процес: дані -> ваги класів -> навчання чотирьох конфігурацій -> збереження
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df_train, df_val, df_test, index = prepare_dataframes()

    #скільки навчальних знімків з кожною патологією -> ваги рідкісності w_c
    counts = df_train[CLASS_NAMES].sum().values
    w_arr = effective_number_weights(counts, beta=BETA)
    #критичність k_c у порядку виходів моделі
    k_arr = [K_CRITICALITY[c] for c in CLASS_NAMES]

    #фіксуємо всі джерела випадковості: Python, PyTorch і NumPy
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = NIHDataset(df_train, index, train_transform)
    if SUBSET and SUBSET < len(df_train):
        train_ds = Subset(train_ds, random.sample(range(len(df_train)), SUBSET))
    print(f'навчальних: {len(train_ds)}  валідація: {len(df_val)}  тест: {len(df_test)}'
          f'   епох: {EPOCHS}  seed: {SEED}  arch: {os.environ.get("ARCH", "resnet50")}'
          f'  workers: {NUM_WORKERS}', flush=True)
    #навчальні знімки подаються пакетами по 64 і щоепохи перемішуються (shuffle);
    #pin_memory пришвидшує передачу пакетів на GPU
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    #для валідації й тесту градієнти не рахуються, пам'яті треба менше, тому пакети по 128
    val_loader = DataLoader(NIHDataset(df_val, index, eval_transform),
                            batch_size=128, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(NIHDataset(df_test, index, eval_transform),
                             batch_size=128, num_workers=NUM_WORKERS, pin_memory=True)

    #(назва, ваги, k, γ0, λ) — абляція: кожна наступна конфігурація додає одну складову;
    #BCE — усі множники вимкнені, Focal — додано фокусування, CB-Focal — ваги класів,
    #CA-CB-Focal — ще й критичність
    configs = [
        ('BCE',         [1.0] * 14, [1.0] * 14, 0.0, 0.0),
        ('Focal',       [1.0] * 14, [1.0] * 14, GAMMA0, 0.0),
        ('CB-Focal',    w_arr,      [1.0] * 14, GAMMA0, 0.0),
        ('CA-CB-Focal', w_arr,      k_arr,      GAMMA0, LAMBDA),
    ]

    print(f"{'Конфіг':14s}{'test AUC':>10s}{'test F1':>9s}", flush=True)
    preds, val_preds = {}, {}
    for name, w, k, g, l in configs:
        #кожна конфігурація починає з тих самих ваг ImageNet
        model = build_model().to(device)
        crit = CACBFocalLoss(w, k, g, l).to(device)
        #оптимізатор AdamW змінює ваги за градієнтами; lr — розмір кроку,
        #weight_decay — легке стягування ваг до нуля проти перенавчання
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
                 if SCHEDULER == 'cosine' else None)

        #файл, куди записується найкраща епоха цієї конфігурації
        ckpt = os.path.join(MODELS_DIR, f'model_{name}.pth')
        best_auc, best_ep = -1.0, 0
        for ep in range(EPOCHS):
            #режим навчання: нормалізація пакетів рахує середні по поточному пакету й оновлює збережені
            model.train()
            #один крок навчання на кожен пакет з 64 знімків
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                #обнулити градієнти попереднього кроку
                opt.zero_grad()
                #прогноз -> помилка -> градієнт кожної ваги (зворотне поширення)
                crit(model(x), y).backward()
                #зсунути ваги на малий крок проти градієнта
                opt.step()
            if sched is not None:
                sched.step()
            #відбір найкращої епохи за валідаційним AUC (захист від перенавчання)
            Pv, Yv = predict(model, val_loader)
            #macro AUC: середнє AUC-ROC по класах, у яких є хоч один хворий
            v_auc = np.mean([roc_auc_score(Yv[:, c], Pv[:, c])
                             for c in range(14) if Yv[:, c].sum() > 0])
            mark = ''
            #якщо ця епоха найкраща — перезаписуємо файл ваг
            if v_auc > best_auc:
                best_auc, best_ep = v_auc, ep + 1
                torch.save(model.state_dict(), ckpt)
                mark = '  <- best'
            lr_now = opt.param_groups[0]['lr']
            print(f"  [{name}] епоха {ep + 1}/{EPOCHS}  val AUC={v_auc:.4f}"
                  f"  lr={lr_now:.2e}{mark}", flush=True)
            #рання зупинка: PATIENCE епох поспіль без покращення
            if ep + 1 - best_ep >= PATIENCE:
                print(f"  [{name}] рання зупинка: {PATIENCE} епох без приросту",
                      flush=True)
                break

        #фінальна оцінка — на найкращому чекпойнті
        model.load_state_dict(torch.load(ckpt, map_location=device))
        Pv, Yv = predict(model, val_loader)
        Pt, Yt = predict(model, test_loader)
        #пороги підбираються на валідації і лише потім застосовуються до тесту
        th = youden_thresholds(Pv, Yv)
        preds[name] = Pt
        val_preds[name] = Pv
        auc = np.mean([roc_auc_score(Yt[:, c], Pt[:, c]) for c in range(14) if Yt[:, c].sum() > 0])
        #macro F1: ймовірності перетворюються на «так/ні» за порогами, F1 усереднюється по класах
        f1 = f1_score(Yt, (Pt > th).astype(int), average='macro', zero_division=0)
        print(f"{name:14s}{auc:10.3f}{f1:9.3f}   (найкраща епоха: {best_ep})", flush=True)

    #валідаційні прогнози (Pv_*, Yv) потрібні, щоб пороги підбиралися не на тесті
    pdir = os.path.join(MODELS_DIR, 'predictions')
    os.makedirs(pdir, exist_ok=True)
    #правильні мітки тесту і валідації та прогнози кожної конфігурації — у файли .npy
    np.save(os.path.join(pdir, 'Yt.npy'), Yt)
    np.save(os.path.join(pdir, 'Yv.npy'), Yv)
    for n, P in preds.items():
        np.save(os.path.join(pdir, f'P_{n}.npy'), P)
    for n, P in val_preds.items():
        np.save(os.path.join(pdir, f'Pv_{n}.npy'), P)
    print(f"\nПрогнози збережено: {pdir}", flush=True)

    #коротке порівняння по класах: кількість хворих у тесті, AUC базової й запропонованої моделі
    print("\nAUC по класах (тест): BCE vs CA-CB-Focal", flush=True)
    for c, name in enumerate(CLASS_NAMES):
        if Yt[:, c].sum() == 0:
            continue
        print(f"{name:18s}{int(Yt[:, c].sum()):6d}"
              f"{roc_auc_score(Yt[:, c], preds['BCE'][:, c]):8.3f}"
              f"{roc_auc_score(Yt[:, c], preds['CA-CB-Focal'][:, c]):8.3f}", flush=True)


#виконується лише під час прямого запуску; інші скрипти імпортують звідси функції без навчання
if __name__ == '__main__':
    main()
