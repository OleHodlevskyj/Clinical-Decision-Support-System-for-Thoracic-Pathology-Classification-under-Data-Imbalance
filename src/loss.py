#клінічно-адаптивна класо-збалансована фокальна функція втрат (CA-CB-Focal):
#звичайна бінарна крос-ентропія, помножена на три множники — вагу рідкісності класу w_c,
#критичність k_c і фокальний множник з класозалежним фокусуванням gamma_c
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#ваги класів за ефективною кількістю зразків: що рідший клас, то більша вага;
#class_counts — скільки знімків з кожною патологією в навчальній вибірці
def effective_number_weights(class_counts, beta=0.9999):
    counts = np.asarray(class_counts, dtype=np.float64)
    #ефективна кількість (1 - beta^n) / (1 - beta): росте з n, але дедалі повільніше,
    #бо нові знімки частого класу вже мало додають нового
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    #вага обернена до ефективної кількості
    w = 1.0 / eff
    #нормування: сума ваг дорівнює кількості класів, тобто середня вага 1
    w = len(counts) * w / w.sum()
    return w.tolist()


#функція втрат CA-CB-Focal; з різними параметрами вона ж дає BCE, Focal і CB-Focal,
#тому для абляції в train.py використовується лише цей клас
class CACBFocalLoss(nn.Module):
    #class_weights — w_c, k_coeffs — k_c, gamma0 — базове фокусування, lam — λ
    def __init__(self, class_weights, k_coeffs, gamma0=2.0, lam=0.5):
        super().__init__()
        w = torch.as_tensor(class_weights, dtype=torch.float32)
        k = torch.as_tensor(k_coeffs, dtype=torch.float32)
        #buffer — незмінні під час навчання числа, які переїжджають на GPU разом із модулем
        self.register_buffer('w', w)
        self.register_buffer('k', k)
        #класозалежне фокусування gamma_c = gamma0 * (1 + λ (k_c - 1)):
        #для критичніших класів фокусування сильніше
        self.register_buffer('gamma_c', gamma0 * (1.0 + lam * (k - 1.0)))

    #обчислює одне число — середню помилку на пакеті;
    #logits — сирі виходи моделі, targets — правильні мітки 0 або 1
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        #p_t — ймовірність, яку модель дала правильній відповіді
        p_t = targets * p + (1 - targets) * (1 - p)
        #бінарна крос-ентропія окремо для кожного знімка і класу
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        #фокальний множник: для впевнених правильних відповідей (p_t близько 1) він малий,
        #тож модель зосереджується на складних прикладах; за gamma_c = 0 він дорівнює 1
        focal = (1 - p_t).clamp(min=1e-8) ** self.gamma_c
        return (self.k * self.w * focal * ce).mean()
