#валідаційні прогнози з уже збережених ваг, без перенавчання
#потрібні для make_thresholds.py: пороги мають підбиратися на валідації, а не на тесті
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import NIHDataset, eval_transform
from model import build_model
#той самий поділ даних і та сама функція прогнозу, що й під час навчання
from train import prepare_dataframes, predict, MODELS_DIR, device

#конфігурації, для яких шукаються файли ваг model_<назва>.pth
CONFIGS = ['BCE', 'Focal', 'CB-Focal', 'CA-CB-Focal']


#для кожної наявної моделі проганяє валідаційні знімки і зберігає Pv_<назва>.npy
def main():
    #потрібна лише валідаційна таблиця; навчальна й тестова пропускаються через _
    _, df_val, _, index = prepare_dataframes()
    loader = DataLoader(NIHDataset(df_val, index, eval_transform),
                        batch_size=128, num_workers=2, pin_memory=True)
    pdir = os.path.join(MODELS_DIR, 'predictions')
    os.makedirs(pdir, exist_ok=True)
    print(f'валідаційних знімків: {len(df_val)}', flush=True)

    for name in CONFIGS:
        path = os.path.join(MODELS_DIR, f'model_{name}.pth')
        #відсутній файл ваг не зупиняє скрипт: ця конфігурація просто пропускається
        if not os.path.exists(path):
            print(f'  {name}: пропущено, немає {os.path.basename(path)}', flush=True)
            continue
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        P, Y = predict(model, loader)
        np.save(os.path.join(pdir, f'Pv_{name}.npy'), P)
        #правильні мітки однакові для всіх моделей, файл просто перезаписується
        np.save(os.path.join(pdir, 'Yv.npy'), Y)
        print(f'  {name}: збережено Pv_{name}.npy', flush=True)

    print(f'\nГотово: {os.path.abspath(pdir)}', flush=True)


if __name__ == '__main__':
    main()
