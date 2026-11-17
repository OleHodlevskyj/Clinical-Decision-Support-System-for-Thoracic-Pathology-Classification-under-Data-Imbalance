#увесь ланцюжок зі справжніми файлами з папки models; без них тести пропускаються
import os
import json

import numpy as np
import pytest
import torch
from PIL import Image

import app
import make_thresholds as mt
from conftest import REAL_MODELS, write_dicom
from model import build_model

pytestmark = pytest.mark.integration
WEIGHTS = os.path.join(REAL_MODELS, 'model_CA-CB-Focal.pth')
THR_FILE = os.path.join(REAL_MODELS, 'thresholds.json')
PRED = os.path.join(REAL_MODELS, 'predictions')


@pytest.fixture(scope='module')
def real_model():
    #справжня навчена модель із models/model_CA-CB-Focal.pth. scope='module' означає,
    #що вона завантажується один раз на всі тести файлу. якщо файлу немає, наприклад
    #після клонування репозиторію, тести пропускаються
    if not os.path.exists(WEIGHTS):
        pytest.skip('немає models/model_CA-CB-Focal.pth')
    m = build_model(pretrained=False, arch='resnet50').to(app.device)
    m.load_state_dict(torch.load(WEIGHTS, map_location=app.device))
    return m.eval()


@pytest.fixture
def real_app(real_model, monkeypatch):
    #підставляє в застосунок справжню модель і справжні пороги замість заглушок
    if not os.path.exists(THR_FILE):
        pytest.skip('немає models/thresholds.json')
    with open(THR_FILE, encoding='utf-8') as fh:
        monkeypatch.setattr(app, '_THR_ALL', json.load(fh))
    monkeypatch.setattr(app, 'M_CA', real_model)
    return app


def test_dicom_to_diagnosis(real_app, tmp_path, gradient):
    #увесь шлях від файлу DICOM до висновку в обох режимах порога:
    #читання файлу, діагностика з тепловою картою і список пріоритетів
    img, _ = real_app.load_dicom(write_dicom(str(tmp_path / 'x.dcm'), gradient))
    for mode in real_app.MODES:
        labels, cam, txt = real_app.diagnose(img, mode)
        assert len(labels) == 14
        assert cam and all(isinstance(m, Image.Image) for m, _ in cam)
        assert txt.startswith('### ')
        rows = real_app.priority(img, mode, *[1.0] * 14)
        assert len(rows) == 14


def test_prediction_is_deterministic(real_app, xray):
    #той самий знімок двічі має давати однакові ймовірності:
    #результат не залежить від випадковості
    assert np.array_equal(real_app._probs(real_app.M_CA, xray), real_app._probs(real_app.M_CA, xray))


def test_saved_thresholds_reproduce_from_validation_predictions():
    #thresholds.json має збігатися з тим, що make_thresholds.py рахує з валідаційних прогнозів.
    #так перевіряється, що файл порогів не застарів, наприклад після перенавчання моделі
    yv, pv = os.path.join(PRED, 'Yv.npy'), os.path.join(PRED, 'Pv_CA-CB-Focal.npy')
    if not (os.path.exists(yv) and os.path.exists(pv) and os.path.exists(THR_FILE)):
        pytest.skip('немає валідаційних прогнозів або thresholds.json')
    with open(THR_FILE, encoding='utf-8') as fh:
        saved = json.load(fh)
    Y, P = np.load(yv), np.load(pv)
    for crit, fn in (('youden', mt.youden), ('f1', mt.f1_max)):
        fresh = [fn(Y[:, c], P[:, c]) for c in range(Y.shape[1])]
        assert np.allclose(saved[crit]['CA-CB-Focal'], fresh)
        assert saved['bands'][crit] == mt.bands(Y, P, fresh)
