#інтерфейс застосунку: які функції запускає кожна подія, які написи бачить лікар
#і чи прибираються тимчасові файли Gradio після закриття
import os
import re

import pytest
from PIL import Image

import app

#події, після яких старі результати мають зникнути з усіх вкладок
CHANGES = {'новий знімок': (app.inp, 'change'),
           'зміна режиму порога': (app.mode, 'change')}


def _handlers(block, event):
    #назви функцій, які Gradio викличе після події event на елементі block
    return {f.fn.__name__ for f in app.demo.fns.values()
            if (block._id, event) in f.targets}


def test_analysis_only_by_button():
    #аналіз запускає лише кнопка «Аналізувати», і вона оновлює всі три вкладки
    assert {'diagnose', 'priority', 'compare'} <= _handlers(app.btn, 'click')


@pytest.mark.parametrize('name', CHANGES)
def test_change_clears_old_results(name):
    #новий знімок або інший режим не запускають аналіз, а прибирають старий результат:
    #раніше «Діагностика» показувала результат попереднього знімка біля нового
    found = _handlers(*CHANGES[name])
    assert '_clear_results' in found
    assert not found & {'diagnose', 'priority', 'compare'}


def test_clear_results_covers_every_output():
    #очищення стосується всіх семи виходів трьох вкладок і підказує, що робити далі
    seven = len([app.lbl, app.cam, app.info, app.out, app.lbl_ca, app.lbl_bce, app.cmp_info])
    out = app._clear_results(object())
    assert len(out) == seven
    assert out[0] == {} and out[1] is None and out[4] == {} and out[5] == {}
    assert out[2] == out[6] == 'Натисніть «Аналізувати».'
    assert app._clear_results(None)[2] == 'Завантажте знімок.'


def test_new_image_clears_dicom_field():
    #лікар завантажив звичайний знімок або прибрав його: назва попереднього файлу DICOM
    #і попередження до нього зникають. очищення прив'язане до upload і clear, а не до change,
    #бо change спрацьовує й тоді, коли знімок у поле підставляє сам load_dicom
    for event in ('upload', 'clear'):
        assert '_reset_dicom' in _handlers(app.inp, event)
    assert '_reset_dicom' not in _handlers(app.inp, 'change')
    assert app._reset_dicom() == (None, '')


def test_image_field_keeps_pixels(xray):
    #знімок з DICOM Gradio зберігає у форматі поля знімка і вже з цього файлу передає моделі.
    #типовий webp стискав з втратами: на знімку VinDr головний результат змінювався з
    #кардіомегалії на грижу. формат поля має повертати ті самі пікселі
    from gradio.processing_utils import encode_pil_to_bytes
    import io
    import numpy as np
    img = xray.convert('RGB')
    back = Image.open(io.BytesIO(encode_pil_to_bytes(img, app.inp.format))).convert('RGB')
    assert np.array_equal(np.array(img), np.array(back))


def test_api_names_kept_for_benchmark():
    #benchmark.py звертається до функцій за іменами /diagnose, /priority, /compare;
    #вони мають лишитися після зміни порядку подій
    names = {f.api_name for f in app.demo.fns.values()}
    assert {'diagnose', 'priority', 'compare'} <= names


def _ui_texts():
    #усі написи інтерфейсу: підписи елементів, варіанти режиму, тексти Markdown і кнопок
    out = list(app.MODES)
    for block in app.demo.blocks.values():
        for attr in ('label', 'value'):
            v = getattr(block, attr, None)
            if isinstance(v, str):
                out.append(v)
    return out


def test_no_alarm_wording(stub_compare):
    #у застосунку не вживається слово «тривога»: замість нього — «позитивний результат».
    #перевіряються і постійні написи, і текст, який формує вкладка порівняння
    texts = _ui_texts() + [stub_compare]
    assert not [t for t in texts if 'тривог' in t.lower()]


@pytest.fixture
def stub_compare(monkeypatch):
    #текст вкладки порівняння з обома моделями-заглушками, у яких усі патології вище порога
    import numpy as np
    monkeypatch.setattr(app, 'M_BCE', 'BCE')
    monkeypatch.setattr(app, '_probs', lambda model, pil: np.ones(14))
    return app.compare(object(), list(app.MODES)[0])[2]


def test_numbers_use_decimal_comma(stub_compare, monkeypatch):
    #усі числа, які бачить лікар, записано з десятковою комою: «x2,85», «p=0,69».
    #шукаються дві цифри через крапку, тобто число з крапкою; теплова карта тут не потрібна
    monkeypatch.setattr(app, '_gradcam', lambda model, pil, cls: None)
    labels, _, txt = app.diagnose(object(), list(app.MODES)[0])
    texts = list(labels) + [txt, stub_compare]
    texts += [cell for row in app.priority(object(), list(app.MODES)[0], *[1.0] * 14) for cell in row]
    assert not [t for t in texts if re.search(r'\d\.\d', str(t))]
    assert app._num(0.6912) == '0,69' and app._num(2.85, 1) == '2,9'


def test_bar_percentages_hidden():
    #Gradio підписує довжину смужки відсотками, і лікар прочитав би їх як ймовірність.
    #відсотки приховано стилем, а великий заголовок над списком, що дублює перший рядок, вимкнено
    assert re.search(r'\.confidence\s*\{\s*display:\s*none', app._CSS)
    for lbl in (app.lbl, app.lbl_ca, app.lbl_bce):
        assert lbl.show_heading is False


def test_priority_table_fits():
    #таблиця «Критичність» вміщується без горизонтальної прокрутки: ширини колонок
    #задано для кожної з шести колонок і разом дають 100 %, довгі заголовки переносяться
    widths = app.out.column_widths
    assert len(widths) == len(app.out.headers) == 6
    assert sum(float(w.rstrip('%')) for w in widths) == 100
    assert app.out.wrap is True


def test_window_opens_light_theme():
    #вікно відкривається у світлій темі незалежно від налаштувань Windows
    with open(app.__file__, encoding='utf-8') as fh:
        assert "demo.local_url + '?__theme=light'" in fh.read()


def test_works_without_internet():
    #застосунок не звертається до інтернету: службова статистика Gradio вимкнена,
    #а шрифти теми або системні, або вбудовані в Gradio, без завантаження з Google Fonts
    from gradio.themes import GoogleFont
    assert os.environ.get('GRADIO_ANALYTICS_ENABLED') == 'False'
    fonts = list(app._THEME._font) + list(app._THEME._font_mono)
    assert not [f for f in fonts if isinstance(f, GoogleFont)]


def test_cache_is_cleared_on_close(xray, tmp_path):
    #справжній сервер Gradio: знімок надсилається так само, як із вікна застосунку.
    #копія знімка і картинка Grad-CAM потрапляють у тимчасову папку, а після закриття
    #застосунку мають зникнути
    from gradio_client import Client, handle_file
    png = str(tmp_path / 'xray.png')
    xray.save(png)
    app.demo.launch(prevent_thread_lock=True, quiet=True, share=False)
    try:
        Client(app.demo.local_url, verbose=False, download_files=False).predict(
            handle_file(png), list(app.MODES)[0], api_name='/diagnose')
        cached = set().union(*app.demo.temp_file_sets)
        assert cached and all(os.path.exists(f) for f in cached)
    finally:
        app.demo.close()
    assert not [f for f in cached if os.path.exists(f)]


@pytest.mark.parametrize('x, shown', [(0.69, '0,69'), (1.0, '1,00'), (0.052, '0,052'),
                                      (0.00212, '0,0021'), (0.00001, '<0,0001')])
def test_small_probabilities_not_shown_as_zero(x, shown):
    #у базової моделі BCE пороги рідкісних патологій близько 0,002: із двома знаками після коми
    #лікар побачив би «p=0,00 / поріг 0,00», тому знаків стільки, щоб було видно дві значущі цифри
    assert app._prob(x) == shown
