#функція втрат CA-CB-Focal, класозалежне фокусування gamma_c і ваги класів за ефективною кількістю
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from loss import CACBFocalLoss, effective_number_weights


@pytest.fixture
def batch():
    #невеликий пакет для тестів: 8 знімків, 3 класи, випадкові виходи мережі й мітки.
    #зерно 0 робить пакет однаковим під час кожного запуску
    torch.manual_seed(0)
    return torch.randn(8, 3), torch.randint(0, 2, (8, 3)).float()


def test_neutral_parameters_equal_bce(batch):
    #без ваг, критичності та фокусування втрата має збігатися зі звичайною BCE.
    #так перевіряється, що формула не має прихованих помилок: у граничному випадку
    #вона дає відомий результат стандартної функції PyTorch
    logits, targets = batch
    loss = CACBFocalLoss([1, 1, 1], [1, 1, 1], gamma0=0.0)(logits, targets)
    assert torch.allclose(loss, F.binary_cross_entropy_with_logits(logits, targets))


def test_class_dependent_focusing():
    #перевіряє формулу γ_c = γ0 · (1 + λ(k_c − 1)): для k_c = 1, 2, 3
    #за γ0 = 2 і λ = 0,5 сила фокусування має дорівнювати 2, 3 і 4
    loss = CACBFocalLoss([1, 1, 1], [1.0, 2.0, 3.0], gamma0=2.0, lam=0.5)
    assert torch.allclose(loss.gamma_c, torch.tensor([2.0, 3.0, 4.0]))


def test_zero_lambda_gives_same_focusing_for_all_classes():
    #за λ = 0 критичність не впливає на фокусування: усі класи отримують γ = γ0,
    #навіть якщо їхні коефіцієнти k_c різні
    loss = CACBFocalLoss([1, 1, 1], [1.0, 2.0, 3.0], gamma0=2.0, lam=0.0)
    assert torch.allclose(loss.gamma_c, torch.full((3,), 2.0))


def test_focusing_suppresses_easy_examples_more_than_hard():
    #легкий приклад: модель упевнено права; важкий: упевнено помиляється.
    #порівнює, у скільки разів фокусування зменшує втрату порівняно з BCE: легкий приклад
    #має зменшуватися сильніше, а жоден приклад не повинен важити більше, ніж у BCE
    loss = CACBFocalLoss([1], [1], gamma0=2.0)
    target = torch.ones(1, 1)
    easy, hard = torch.tensor([[4.0]]), torch.tensor([[-4.0]])
    ratio_easy = loss(easy, target) / F.binary_cross_entropy_with_logits(easy, target)
    ratio_hard = loss(hard, target) / F.binary_cross_entropy_with_logits(hard, target)
    assert ratio_easy < ratio_hard <= 1.0


def test_higher_criticality_increases_loss(batch):
    #якщо підняти k_c одного класу, загальна втрата має зрости:
    #так метод сильніше штрафує модель за помилки на небезпечній патології
    logits, targets = batch
    base = CACBFocalLoss([1, 1, 1], [1, 1, 1], gamma0=0.0)(logits, targets)
    critical = CACBFocalLoss([1, 1, 1], [1, 1, 3], gamma0=0.0)(logits, targets)
    assert critical > base


def test_class_weights_scale_loss_linearly(batch):
    #подвоєння всіх ваг w_c має рівно вдвічі збільшити втрату:
    #ваги лише множать внесок класу й не змінюють решту формули
    logits, targets = batch
    one = CACBFocalLoss([1, 1, 1], [1, 2, 3])(logits, targets)
    two = CACBFocalLoss([2, 2, 2], [1, 2, 3])(logits, targets)
    assert torch.allclose(two, 2 * one)


def test_extreme_logits_stay_finite():
    #дуже великі виходи мережі (±50), коли модель абсолютно впевнена в правильній
    #або хибній відповіді: втрата й градієнти мають лишатися звичайними числами,
    #без nan чи нескінченності, інакше навчання зламалося б
    logits = torch.tensor([[50.0, -50.0, 50.0, -50.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    loss = CACBFocalLoss([1] * 4, [1, 2, 3, 3])(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_rare_class_gets_larger_weight():
    #ваги за ефективною кількістю зразків: що рідший клас, то більша його вага
    w = effective_number_weights([50000, 5000, 200])
    assert w[0] < w[1] < w[2]


def test_weights_are_normalised_to_number_of_classes():
    #після нормування сума ваг дорівнює кількості класів, тобто середня вага дорівнює 1
    counts = [50000, 5000, 200, 10]
    assert sum(effective_number_weights(counts)) == pytest.approx(len(counts))


@pytest.mark.parametrize('counts, beta', [([300, 300, 300], 0.9999), ([50000, 10], 0.0)])
def test_no_reweighting_cases(counts, beta):
    #два випадки, коли перезважування не потрібне, тому всі ваги мають дорівнювати 1:
    #рівні класи або beta=0 (кожен зразок «ефективний» лише один раз)
    assert np.allclose(effective_number_weights(counts, beta=beta), 1.0)
