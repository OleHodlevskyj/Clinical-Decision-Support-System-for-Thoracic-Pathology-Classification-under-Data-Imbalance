#спільні налаштування тестів: шляхи до модулів, заглушка папки моделей, штучні DICOM
import os
import sys
import json
import shutil
import tempfile

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REAL_MODELS = os.path.join(ROOT, 'models')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'app'))

#пороги заглушки: однакові для всіх класів, щоб очікувані результати рахувались вручну
STUB_THR = {'youden': 0.2, 'f1': 0.4}
STUB_BANDS = [1.35, 1.75, 2.2]
_stub_dir = None


def pytest_configure(config):
    #pytest викликає цю функцію сам, ще до запуску тестів.
    #app.py завантажує моделі під час імпорту, тому папку-заглушку треба підготувати
    #до збирання тестів: модель на випадкових вагах і штучний thresholds.json.
    #так модульні тести не залежать від справжніх ваг і не змінюють їх
    global _stub_dir
    import torch
    from model import build_model
    _stub_dir = tempfile.mkdtemp(prefix='stub_models_')
    torch.manual_seed(0)
    torch.save(build_model(pretrained=False, arch='resnet50').state_dict(),
               os.path.join(_stub_dir, 'model_CA-CB-Focal.pth'))
    thr = {crit: {cfg: [v] * 14 for cfg in ('CA-CB-Focal', 'BCE')} for crit, v in STUB_THR.items()}
    thr['bands'] = {crit: STUB_BANDS for crit in STUB_THR}
    with open(os.path.join(_stub_dir, 'thresholds.json'), 'w', encoding='utf-8') as fh:
        json.dump(thr, fh)
    os.environ['MODELS_DIR'] = _stub_dir


def pytest_unconfigure(config):
    #pytest викликає її сам після всіх тестів: видаляє тимчасову папку із заглушкою,
    #щоб на диску не лишалися зайві файли
    if _stub_dir:
        shutil.rmtree(_stub_dir, ignore_errors=True)


def write_dicom(path, pixels, photometric='MONOCHROME2', **fields):
    #мінімальний коректний файл DICOM із заданими пікселями та полями заголовка.
    #потрібен тестам читання DICOM і замірам швидкодії, бо справжніх знімків пацієнтів
    #брати не можна; поля на кшталт імені пацієнта чи проєкції передаються як іменовані аргументи
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage
    pixels = np.asarray(pixels, dtype=np.uint16)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PhotometricInterpretation = photometric
    rgb = photometric == 'RGB'
    ds.SamplesPerPixel = 3 if rgb else 1
    if rgb:
        ds.PlanarConfiguration = 0
    frames = pixels.shape[0] if pixels.ndim == 3 and not rgb else 1
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.Rows, ds.Columns = pixels.shape[-3:-1] if rgb else pixels.shape[-2:]
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = 0
    for k, v in fields.items():
        setattr(ds, k, v)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def gradient():
    #штучний знімок 64 на 64 для файлів DICOM: ліворуч темно, праворуч світло.
    #за ним легко перевірити, чи не переплутано напрямок яскравості після читання
    return np.tile(np.linspace(0, 4000, 64), (64, 1))


@pytest.fixture
def xray():
    #звичайне зображення з випадковим шумом замість справжнього рентгена:
    #потрібне там, де важлива лише форма результату, а не сам діагноз
    from PIL import Image
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (256, 256), dtype=np.uint8)).convert('RGB')
