#логіка вкладок застосунку: замість нейромережі — заглушка з наперед заданими ймовірностями
import numpy as np
import pytest
from PIL import Image

import app
from config import CLASS_NAMES, K_CRITICALITY
from conftest import STUB_THR, STUB_BANDS

YOUDEN, F1 = list(app.MODES)
UA = dict(zip(CLASS_NAMES, app.UA))
IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def _p(**ratios):
    #ймовірності як кратність порога Юдена: _p(Hernia=2.0) — грижа вдвічі вище порога.
    #решта патологій отримує ймовірність у 10 разів нижчу за поріг
    p = np.full(14, STUB_THR['youden'] * 0.1)
    for c, r in ratios.items():
        p[IDX[c]] = STUB_THR['youden'] * r
    return p


@pytest.fixture
def stub(monkeypatch):
    #заглушка моделі: повертає задані ймовірності й не рахує Grad-CAM.
    #monkeypatch підміняє функції лише на час тесту, тому результат залежить
    #тільки від заданих чисел і його можна передбачити наперед
    probs = {}
    monkeypatch.setattr(app, '_probs', lambda model, pil: probs[model])
    monkeypatch.setattr(app, '_gradcams', lambda model, pil, classes: ['CAM:%d' % c for c in classes])
    monkeypatch.setattr(app, 'M_BCE', 'BCE')
    probs[app.M_CA] = _p()
    probs['BCE'] = _p()
    return probs


def test_thresholds_follow_selected_mode():
    #пороги й межі сили сигналу мають братися з вибраного режиму (Юдена чи F1)
    #і для потрібної моделі
    assert np.allclose(app._thr(YOUDEN), STUB_THR['youden'])
    assert np.allclose(app._thr(F1, 'BCE'), STUB_THR['f1'])
    assert app._bands(F1) == STUB_BANDS


def test_fallback_without_thresholds_file(monkeypatch):
    #якщо файлу thresholds.json немає, застосунок не падає:
    #бере нейтральний поріг 0,5 і стандартні межі сили сигналу
    monkeypatch.setattr(app, '_THR_ALL', {})
    assert np.allclose(app._thr(YOUDEN), 0.5)
    assert len(app._bands(YOUDEN)) == 3


def test_ranked_labels_are_unique_and_bars_capped():
    #підписи для списку ймовірностей: 14 різних рядків, бо однакових Gradio не прийме,
    #а довжина смужки від 0 до 1; смужка заповнюється повністю від кратності 3
    p = _p(Hernia=6.0, Edema=1.5)
    out = app._ranked(p, app._thr(YOUDEN))
    assert len(out) == 14
    assert all(0 <= v <= 1 for v in out.values())
    hernia = next(k for k in out if k.startswith(UA['Hernia']))
    edema = next(k for k in out if k.startswith(UA['Edema']))
    assert out[hernia] == 1.0
    assert out[edema] == pytest.approx(1.5 / 3)


@pytest.mark.parametrize('ratio, head', [
    (1.0, 'Виражених ознак не виявлено'),
    (1.5, 'Слабкий сигнал'),
    (2.0, 'Помірний сигнал'),
    (3.0, 'Виражений сигнал'),
])
def test_diagnose_signal_strength(stub, ratio, head):
    #чотири рівні висновку залежно від того, у скільки разів ймовірність пневмотораксу
    #перевищує поріг; теплову карту будують саме для пневмотораксу, а коли сигналу
    #немає, не будують зовсім, щоб не показувати лікареві випадкову ділянку
    stub[app.M_CA] = _p(Pneumothorax=ratio)
    labels, cam, txt = app.diagnose(object(), YOUDEN)
    assert txt.startswith('### ' + head)
    assert len(labels) == 14
    if ratio >= STUB_BANDS[0]:
        assert UA['Pneumothorax'] in txt
        assert cam == [('CAM:%d' % IDX['Pneumothorax'], '%s x%s' % (UA['Pneumothorax'], app._num(ratio)))]
    else:
        assert 'Жодна патологія не перевищує' in txt
        assert 'теплова карта не будується' in txt
        assert cam is None


def test_diagnose_lists_all_signals_strongest_first(stub):
    #усі патології вище межі сигналу перелічуються в тексті, найсильніша — першою
    stub[app.M_CA] = _p(Edema=1.6, Mass=2.5)
    _, cam, txt = app.diagnose(object(), YOUDEN)
    listed = txt.split('**')[-2]
    assert listed == '%s, %s' % (UA['Mass'], UA['Edema'])
    #теплова карта є для кожної патології з висновку, у тому самому порядку, з назвою і кратністю
    assert [img for img, _ in cam] == ['CAM:%d' % IDX['Mass'], 'CAM:%d' % IDX['Edema']]
    assert [cap for _, cap in cam] == ['%s x2,50' % UA['Mass'], '%s x1,60' % UA['Edema']]


def test_mode_changes_decision(stub):
    #той самий знімок: за порогом Юдена сигнал є, за суворішим порогом F1 — немає
    stub[app.M_CA] = _p(Effusion=1.8)
    assert 'Помірний' in app.diagnose(object(), YOUDEN)[2]
    assert 'не виявлено' in app.diagnose(object(), F1)[2]


def test_priority_alarms_first_then_criticality(stub):
    #правило сортування на вкладці «Критичність»: спершу патології вище свого порога,
    #а серед них вирішує пріоритет (p / поріг) · k_c.
    #інфільтрація вище порога сильніше, але пневмоторакс критичніший — він перший;
    #грижа має найвищий k_c, але без ознак, тому йде лише після патологій з ознаками
    stub[app.M_CA] = _p(Infiltration=1.5, Pneumothorax=1.25, Hernia=0.95)
    ks = [K_CRITICALITY[c] for c in CLASS_NAMES]
    ks[IDX['Hernia']] = 3.0
    rows = app.priority(object(), YOUDEN, *ks)
    assert [r[0] for r in rows[:3]] == [UA['Pneumothorax'], UA['Infiltration'], UA['Hernia']]
    assert [r[-1] for r in rows[:3]] == ['Позитивний', 'Позитивний', '—']
    assert len(rows) == 14
    #числа в таблиці записано з десятковою комою
    assert float(rows[0][4].replace(',', '.')) == pytest.approx(1.25 * 3.0)


def test_compare_splits_alarms_between_models(stub):
    #моделі CA-CB-Focal і BCE знаходять різні патології: текст має правильно розкласти
    #їх на «в обох», «лише в CA-CB-Focal» і «лише в базовій BCE»
    stub[app.M_CA] = _p(Hernia=1.2, Edema=1.2)
    stub['BCE'] = _p(Edema=1.2, Nodule=1.2)
    lbl_ca, lbl_bce, txt = app.compare(object(), YOUDEN)
    both, only_ca, only_bce = [line.split(':** ')[1] for line in txt.split('\n\n')]
    assert (both, only_ca, only_bce) == (UA['Edema'], UA['Hernia'], UA['Nodule'])
    assert len(lbl_ca) == len(lbl_bce) == 14


def test_compare_without_baseline_model(stub, monkeypatch):
    #якщо файлу базової моделі немає, вкладка порівняння пояснює це, а не падає
    monkeypatch.setattr(app, 'M_BCE', None)
    assert 'model_BCE.pth' in app.compare(object(), YOUDEN)[2]


@pytest.mark.parametrize('fn, args', [(app.diagnose, ()), (app.compare, ()), (app.priority, (1.0,) * 14)])
def test_no_image_gives_hint(fn, args):
    #кнопку натиснули без знімка: кожна вкладка просить завантажити знімок.
    #текст шукається без першої літери, бо вона буває і великою, і малою
    out = fn(None, YOUDEN, *args)
    assert 'авантажте знімок' in str(out)


def test_real_inference_path_on_stub_weights(xray):
    #справжні _probs і _gradcam на моделі з випадковими вагами: перевіряється форма виходу.
    #заглушок тут немає, тож тест показує, що справжні функції виконуються без помилок,
    #дають 14 ймовірностей від 0 до 1 і будують зображення теплової карти
    p = app._probs(app.M_CA, xray)
    assert p.shape == (14,) and ((p >= 0) & (p <= 1)).all()
    cam = app._gradcam(app.M_CA, xray, int(np.argmax(p)))
    assert isinstance(cam, Image.Image) and cam.width > 0
    #кілька карт за один прохід: по одній на кожен клас, і для різних класів вони різні
    maps = app._gradcams(app.M_CA, xray, [0, 13])
    assert len(maps) == 2 and all(isinstance(m, Image.Image) for m in maps)
    assert not np.array_equal(np.asarray(maps[0]), np.asarray(maps[1]))
