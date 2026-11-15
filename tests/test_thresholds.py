#пороги класифікації: індекс Юдена, максимум F1-міри, межі сили сигналу
import json

import numpy as np
import pytest

import make_thresholds as mt


@pytest.fixture
def separable():
    #хворі й здорові розділені повністю: на таких даних правильний поріг очевидний
    return np.array([0, 0, 0, 1, 1, 1]), np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])


@pytest.fixture
def noisy():
    #реалістичніший випадок: 400 знімків, близько 15% хворих,
    #а ймовірності хворих і здорових частково перекриваються
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.15).astype(int)
    return y, np.clip(0.3 * y + rng.normal(0.3, 0.15, 400), 0, 1)


@pytest.mark.parametrize('crit', [mt.youden, mt.f1_max])
def test_threshold_splits_separable_classes(separable, crit):
    #обидва критерії на повністю розділених даних мають дати поріг між
    #найвищою ймовірністю здорового і найнижчою ймовірністю хворого
    y, p = separable
    t = crit(y, p)
    assert p[y == 0].max() < t <= p[y == 1].min()


def _j(y, p, t):
    #допоміжна: індекс Юдена для порога t, порахований вручну, незалежно від коду,
    #який перевіряємо: частка знайдених хворих мінус частка здорових, позначених як хворі
    pred = p >= t
    return pred[y == 1].mean() - pred[y == 0].mean()


def _f1(y, p, t):
    #допоміжна: F1-міра для порога t, порахована вручну за формулою
    pred = p >= t
    tp = (pred & (y == 1)).sum()
    return 2 * tp / (pred.sum() + (y == 1).sum())


@pytest.mark.parametrize('crit, score', [(mt.youden, _j), (mt.f1_max, _f1)])
def test_threshold_is_optimal_among_all_candidates(noisy, crit, score):
    #перебором усіх можливих порогів: кращого значення критерію не існує.
    #порівнює поріг, знайдений make_thresholds.py, з найкращим результатом перебору
    y, p = noisy
    best = max(score(y, p, t) for t in np.unique(p))
    assert score(y, p, crit(y, p)) == pytest.approx(best)


def test_bands_are_ordered_and_signal_starts_above_threshold(noisy):
    #межі сили сигналу (слабкий, помірний, виражений) мають іти за зростанням
    #і починатися не нижче 1, тобто не нижче самого порога
    y, p = noisy
    Y = np.stack([y, np.roll(y, 7)], 1)
    P = np.stack([p, np.roll(p, 7)], 1)
    weak, moderate, strong = mt.bands(Y, P, [0.5, 0.5])
    assert 1.0 <= weak <= moderate <= strong


def test_weak_band_never_below_one():
    #якщо у здорових ймовірності далеко під порогом, слабкий сигнал однаково починається з 1
    Y = np.array([[0], [0], [1], [1]])
    P = np.array([[0.01], [0.02], [0.9], [0.95]])
    assert mt.bands(Y, P, [0.5])[0] == 1.0


@pytest.fixture
def predictions(tmp_path, monkeypatch):
    #штучні валідаційні прогнози двох моделей у тимчасовій папці. monkeypatch на час тесту
    #перенаправляє туди make_thresholds.py, тож справжні файли в models не змінюються.
    #у класу 13 (грижа) навмисно немає жодного хворого
    rng = np.random.default_rng(2)
    Y = (rng.random((300, 14)) < 0.2).astype(int)
    Y[:, 13] = 0
    for name in ('CA-CB-Focal', 'BCE'):
        np.save(tmp_path / f'Pv_{name}.npy', np.clip(0.4 * Y + rng.random((300, 14)) * 0.6, 0, 1))
    np.save(tmp_path / 'Yv.npy', Y)
    monkeypatch.setattr(mt, 'PRED', str(tmp_path))
    monkeypatch.setattr(mt, 'OUT', str(tmp_path / 'thresholds.json'))
    return tmp_path


def test_main_writes_thresholds_for_every_configuration(predictions):
    #повний запуск make_thresholds.py: у thresholds.json мають з'явитися обидва критерії
    #для обох моделей, по 14 порогів і по 3 межі сили сигналу
    mt.main()
    res = json.loads((predictions / 'thresholds.json').read_text(encoding='utf-8'))
    assert set(res) == {'youden', 'f1', 'bands'}
    for crit in ('youden', 'f1'):
        assert set(res[crit]) == {'CA-CB-Focal', 'BCE'}
        assert all(len(v) == 14 for v in res[crit].values())
        #у класу без жодного хворого поріг не обчислюється — нейтральні 0.5
        assert res[crit]['CA-CB-Focal'][13] == 0.5
        assert len(res['bands'][crit]) == 3


def test_main_stops_without_validation_labels(tmp_path, monkeypatch):
    #якщо немає файлу з мітками валідаційної вибірки Yv.npy, скрипт має зупинитися
    #зі зрозумілим повідомленням, а не впасти з незрозумілою помилкою
    monkeypatch.setattr(mt, 'PRED', str(tmp_path))
    with pytest.raises(SystemExit):
        mt.main()
