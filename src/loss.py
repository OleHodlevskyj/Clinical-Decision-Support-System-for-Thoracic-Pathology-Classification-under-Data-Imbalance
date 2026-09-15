#Клінічно-адаптивна класо-збалансована фокальна функція втрат (CA-CB-Focal)
#формула (2.6)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(class_counts, beta=0.9999):
    #Ваги w_c за ефективною кількістю зразків (формула 2.3)
    counts = np.asarray(class_counts, dtype=np.float64)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / eff
    w = len(counts) * w / w.sum()
    return w.tolist()


class CACBFocalLoss(nn.Module):
    #Формула (2.6)
    def __init__(self, class_weights, k_coeffs, gamma0=2.0, lam=0.5):
        super().__init__()
        w = torch.as_tensor(class_weights, dtype=torch.float32)
        k = torch.as_tensor(k_coeffs, dtype=torch.float32)
        self.register_buffer('w', w)
        self.register_buffer('k', k)
        #класозалежне фокусування (формула 2.5)
        self.register_buffer('gamma_c', gamma0 * (1.0 + lam * (k - 1.0)))

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal = (1 - p_t).clamp(min=1e-8) ** self.gamma_c
        return (self.k * self.w * focal * ce).mean()
