#читання файлів DICOM з рентген-апарата: штучні файли створюються в тимчасовій папці
import os

import numpy as np
import pydicom
import pytest
from PIL import Image

import app
from conftest import write_dicom


def _gray(img):
    #допоміжна: переводить зображення у відтінки сірого як масив чисел, щоб порівнювати яскравість
    return np.asarray(img.convert('L'), dtype=float)


def test_monochrome2_becomes_rgb_image(tmp_path, gradient):
    #звичайний знімок MONOCHROME2 має стати зображенням RGB того самого розміру, бо такого
    #чекає модель. напрямок яскравості зберігається, значення розтягуються на весь
    #діапазон від 0 до 255, а попереджень немає
    img, txt = app.load_dicom(write_dicom(str(tmp_path / 'a.dcm'), gradient))
    assert isinstance(img, Image.Image) and img.mode == 'RGB'
    assert img.size == (64, 64)
    g = _gray(img)
    assert g[:, 0].mean() < g[:, -1].mean()
    assert g.min() == 0 and g.max() == 255
    assert 'завантажено' in txt and 'Увага' not in txt


def test_monochrome1_is_inverted(tmp_path, gradient):
    #у MONOCHROME1 повітря світле, а модель навчалась на темному повітрі,
    #тому після читання такий знімок має бути інвертованим варіантом MONOCHROME2
    m2, _ = app.load_dicom(write_dicom(str(tmp_path / 'm2.dcm'), gradient))
    m1, _ = app.load_dicom(write_dicom(str(tmp_path / 'm1.dcm'), gradient, 'MONOCHROME1'))
    assert np.allclose(_gray(m1), 255 - _gray(m2), atol=1)


def test_window_from_file_is_applied(tmp_path, gradient):
    #вікно яскравості охоплює лише темну половину — світла частина насичується.
    #перевіряє, що застосунок бере вікно з самого файлу, як рентген-апарат
    img, _ = app.load_dicom(write_dicom(str(tmp_path / 'w.dcm'), gradient,
                                        WindowCenter=1000, WindowWidth=2000))
    g = _gray(img)
    assert (g[:, -10:] == 255).all()
    assert g[:, 0].mean() < 10


def test_rgb_file_is_converted_to_gray(tmp_path, gradient):
    #кольоровий файл DICOM має стати сірим: усі три канали однакові
    rgb = np.repeat(gradient[..., None], 3, -1)
    img, _ = app.load_dicom(write_dicom(str(tmp_path / 'rgb.dcm'), rgb, 'RGB', BitsStored=16, HighBit=15))
    g = np.asarray(img)
    assert (g[..., 0] == g[..., 1]).all() and (g[..., 1] == g[..., 2]).all()


def test_multiframe_uses_first_frame(tmp_path, gradient):
    #файл із двома кадрами, де другий віддзеркалений: застосунок має взяти
    #саме перший кадр і попередити лікаря, що кадрів кілька
    frames = np.stack([gradient, gradient[:, ::-1]])
    img, txt = app.load_dicom(write_dicom(str(tmp_path / 'mf.dcm'), frames))
    g = _gray(img)
    assert g[:, 0].mean() < g[:, -1].mean()
    assert 'кілька кадрів' in txt


@pytest.mark.parametrize('field, value, warning', [
    ('ViewPosition', 'LL', 'бічна проєкція'),
    ('BurnedInAnnotation', 'YES', 'надруковані дані пацієнта'),
])
def test_warnings_from_header(tmp_path, gradient, field, value, warning):
    #бічна проєкція та надруковані на знімку дані пацієнта мають давати
    #попередження з позначкою «Увага:»
    _, txt = app.load_dicom(write_dicom(str(tmp_path / 'w.dcm'), gradient, **{field: value}))
    assert warning in txt and '**Увага:**' in txt


def test_personal_data_is_not_shown_and_temp_copy_is_removed(tmp_path, gradient):
    #перевіряє обіцянку щодо особистих даних: ім'я, номер картки й дата народження
    #з файлу не з'являються в повідомленні, а тимчасова копія файлу видаляється.
    #ISO_IR 192 — кодування UTF-8, інакше кирилиця в імені запишеться знаками-замінниками
    path = write_dicom(str(tmp_path / 'p.dcm'), gradient, SpecificCharacterSet='ISO_IR 192',
                       PatientName='Іваненко^Петро', PatientID='CARD-0042', PatientBirthDate='19700101')
    assert str(pydicom.dcmread(path).PatientName) == 'Іваненко^Петро'
    _, txt = app.load_dicom(path)
    for secret in ('Іваненко', 'Петро', 'CARD-0042', '19700101'):
        assert secret not in txt
    assert not os.path.exists(path)


def test_not_a_dicom_file(tmp_path):
    #файл, що не є DICOM (наприклад, звичайне фото): зображення в полі не змінюється,
    #лікар бачить зрозуміле повідомлення, а тимчасова копія однаково видаляється
    path = tmp_path / 'photo.jpg'
    path.write_bytes(b'not a dicom at all')
    upd, txt = app.load_dicom(str(path))
    assert upd == app.gr.update()
    assert 'Не вдалося прочитати' in txt
    assert not path.exists()


def test_empty_upload_changes_nothing():
    #поле файлу очистили: нічого не змінюється й жодного тексту не з'являється
    upd, txt = app.load_dicom(None)
    assert upd == app.gr.update() and txt == ''
