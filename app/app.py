#застосунок лікаря: вікно, у якому завантажують рентген грудної клітки і отримують
#ймовірності 14 патологій, теплову карту Grad-CAM, пріоритети за критичністю
#і порівняння з базовою моделлю BCE.
import os
import sys
import glob
import json

#папка src лежить поруч із app; додаємо її до шляхів пошуку, щоб працювали import config і model
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
#Gradio типово надсилає розробникам службову статистику використання; застосунок має
#працювати без інтернету, тому вимикаємо її до імпорту gradio
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import numpy as np
import torch
import gradio as gr
import pydicom
from pydicom.pixels import apply_voi_lut
from torchvision import transforms as T
from PIL import Image
import matplotlib
#Agg малює картинки в пам'ять без відкриття окремих вікон matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#спільні налаштування проєкту: назви класів, українські підписи, критичність k_c,
#параметри нормалізації і розмір зображення
from config import (CLASS_NAMES, UA_LABELS, K_CRITICALITY,
                    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE)
from model import build_model

#відеокарта NVIDIA, якщо вона є, інакше центральний процесор
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#папку з моделями задати змінною середовища MODELS_DIR
MODELS_DIR = os.environ.get('MODELS_DIR',
                            os.path.join(os.path.dirname(__file__), '..', 'models'))
#українські назви 14 патологій у тому самому порядку, що й виходи моделі
UA = [UA_LABELS[c] for c in CLASS_NAMES]

#завантаження моделей
#словник «назва конфігурації -> шлях до файлу»: з model_BCE.pth виходить BCE
#([6:-4] відрізає «model_» на початку і «.pth» у кінці)
_mfiles = {os.path.basename(f)[6:-4]: f
           for f in glob.glob(os.path.join(MODELS_DIR, 'model_*.pth'))}
#без жодного файлу ваг застосунок не має сенсу, тому зупиняємось з поясненням
assert _mfiles, f'Моделі не знайдено в {MODELS_DIR} (потрібні model_*.pth)'


#створює порожню ResNet-50 і заповнює її навченими вагами з файлу
def _load(name):
    #архітектура задана явно: Grad-CAM нижче чіпляється до layer4, це шар ResNet
    m = build_model(pretrained=False, arch='resnet50').to(device)
    #map_location дозволяє відкрити ваги, збережені на GPU, на комп'ютері без відеокарти
    m.load_state_dict(torch.load(_mfiles[name], map_location=device))
    #режим використання: нормалізація пакетів бере середні, збережені під час навчання,
    #тож результат для знімка не залежить від того, що аналізувалося перед ним
    m.eval()#перемикає модель у режим використання, а не навчання
    return m


#основна модель обов'язкова; базова BCE потрібна лише для вкладки порівняння
M_CA = _load('CA-CB-Focal')
M_BCE = _load('BCE') if 'BCE' in _mfiles else None

#по-класові пороги за двома критеріями (src/make_thresholds.py, валідаційна вибірка):
#індекс Юдена — на користь чутливості, максимум F1-міри — на користь точності
_THR_FILE = os.path.join(MODELS_DIR, 'thresholds.json')
#якщо файлу порогів немає, словник лишається порожнім і діють запасні значення нижче
_THR_ALL = {}
if os.path.exists(_THR_FILE):
    with open(_THR_FILE, encoding='utf-8') as _fh:
        _THR_ALL = json.load(_fh)

#підпис режиму в інтерфейсі -> ключ у thresholds.json
MODES = {'Не пропустити хворого (індекс Юдена)': 'youden',
         'Менше хибнопозитивних результатів (максимум F1-міри)': 'f1'}
#типово вибрано перший режим, тобто індекс Юдена
MODE_DEFAULT = list(MODES)[0]


#повертає 14 порогів для вибраного режиму і моделі; без файлу порогів — 0,5 для всіх
def _thr(mode_label, cfg='CA-CB-Focal'):
    return np.array(_THR_ALL.get(MODES[mode_label], {}).get(cfg, [0.5] * 14))


#повертає три межі сили сигналу (слабкий, помірний, виражений) для вибраного режиму
def _bands(mode_label):
    #межі сили сигналу — з розподілу максимальної кратності p/поріг на валідації
    return _THR_ALL.get('bands', {}).get(MODES[mode_label], [1.35, 1.75, 2.2])


#підготовка знімка так само, як під час навчання: розмір 224 x 224, перетворення
#на тензор (числа 0..1) і нормалізація середнім та відхиленням ImageNet
_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                 T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


#проганяє один знімок через модель і повертає 14 ймовірностей від 0 до 1
def _probs(model, pil):
    #RGB, бо модель чекає три канали; unsqueeze(0) робить «пакет» з одного знімка
    x = _tf(pil.convert('RGB')).unsqueeze(0).to(device)
    #без запам'ятовування проміжних значень для градієнтів: швидше й менше пам'яті
    with torch.no_grad():
        #сигмоїда перетворює вихід моделі на ймовірності; [0] — перший і єдиний знімок
        return torch.sigmoid(model(x))[0].cpu().numpy()


#будує теплову карту Grad-CAM: які ділянки знімка найбільше вплинули на клас cls
def _gradcam(model, pil, cls):
    x = _tf(pil.convert('RGB')).unsqueeze(0).to(device)
    store = {}
    #хук на останньому блоці layer4: перший запам'ятовує його вихід (карти ознак),
    #другий — градієнт, тобто наскільки кожна карта впливає на відповідь
    h1 = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__('a', o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: store.__setitem__('g', go[0]))
    out = model(x)#прохід уперед: отримуємо 14 логітів, хук 1 зберігає карти
    model.zero_grad()#стерти старі градієнти, якщо вони лишилися
    #зворотний прохід від виходу потрібного класу заповнює градієнт у store
    out[0, cls].backward()
    A, G = store['a'][0], store['g'][0]
    #вага кожної карти — середній градієнт по ній; зважена сума карт дає теплову карту,
    #relu лишає лише ділянки, що підсилюють відповідь
    cam = torch.relu((G.mean(dim=(1, 2))[:, None, None] * A).sum(0))
    #зводимо до діапазону 0..1; 1e-8 захищає від ділення на нуль
    cam = cam / (cam.max() + 1e-8)
    #хуки знімаємо, щоб вони не спрацьовували під час наступних звичайних прогнозів
    h1.remove(); h2.remove()
    #карта має розмір 7 x 7, тому розтягуємо її до розміру знімка
    cam = np.array(Image.fromarray((cam.detach().cpu().numpy() * 255).astype('uint8'))
                   .resize((IMG_SIZE, IMG_SIZE))) / 255.
    base = np.array(pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    #накладаємо напівпрозору кольорову карту на знімок; subplots_adjust прибирає білі поля навколо
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(base); ax.imshow(cam, cmap='jet', alpha=0.45); ax.axis('off')
    #малюємо рисунок у пам'яті й перетворюємо його на картинку для інтерфейсу
    fig.canvas.draw()
    out_img = Image.fromarray(np.array(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return out_img


#число з десятковою комою
def _num(x, digits=2):
    return ('%.*f' % (digits, x)).replace('.', ',')


#ймовірність або поріг: у базової моделі BCE пороги рідкісних патологій бувають близько 0,002,
#і з двома знаками після коми перетворилися б на «0,00». тому знаків стільки,
#щоб було видно дві значущі цифри: 0,69; 0,052; 0,0021
def _prob(x):
    if x < 1e-4:
        return '<0,0001'
    return _num(x, max(2, 1 - int(np.floor(np.log10(x)))))


#готує дані для списку ймовірностей: підпис кожної патології і довжину її смужки
def _ranked(p, thr):
    #для gr.Label: підпис із кратністю і довжина смужки; ключі мають бути унікальні
    ratio = p / thr
    #смужка заповнюється повністю, коли ймовірність утричі перевищує поріг.
    #Gradio підписує довжину смужки відсотками, але це не ймовірність, тому відсотки
    #приховано стилем .confidence нижче, а ймовірність p написано в самому підписі
    return {'%s  x%s  (p=%s / поріг %s)' % (UA[i], _num(ratio[i]), _prob(p[i]), _prob(thr[i])):
            float(min(ratio[i] / 3, 1.0)) for i in range(14)}


#відкриває файл DICOM з рентген-апарата, бере з нього лише зображення
#і повертає його разом із текстом-попередженням
def load_dicom(path):
    #з файлу береться лише зображення: ім'я пацієнта, дата народження, номер картки
    #та інші поля DICOM не читаються й ніде не показуються
    if not path:
        #gr.update() без аргументів означає «нічого не змінювати в полі знімка»
        return gr.update(), ""
    try:
        ds = pydicom.dcmread(path)
        #pixel_array — самі пікселі; стиснуті JPEG і JPEG 2000 розпаковує pylibjpeg
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        return gr.update(), f"**Не вдалося прочитати файл як DICOM:** {type(e).__name__}"
    finally:
        #копію з особистими даними, яку Gradio зберіг у тимчасовій папці, прибираємо
        try:
            os.remove(path)
        except OSError:
            pass

    notes = []
    #кольорове зображення (3 або 4 канали) переводимо в сіре усередненням каналів
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(-1)
    #кілька кадрів в одному файлі: беремо перший
    elif arr.ndim == 3:
        arr = arr[0]
        notes.append("файл містить кілька кадрів — взято перший")
    #вікно яскравості з самого файлу: без нього знімок виходить надто темним або блідим
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass
    #MONOCHROME1 — кістки темні, повітря світле; модель навчалась на протилежному
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = arr.max() - arr
    #зводимо яскравість до діапазону 0..255, як у звичайній картинці
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    img = Image.fromarray(arr.astype('uint8')).convert('RGB')

    #проєкцію знімка беремо з поля ViewPosition; бічні моделі не знайомі
    view = str(getattr(ds, 'ViewPosition', '')).upper()
    if view in ('LL', 'RL', 'LAT', 'LATERAL'):
        notes.append("бічна проєкція — модель навчалась лише на прямих, результат ненадійний")
    #апарат позначає, чи надруковано текст прямо на зображенні
    if str(getattr(ds, 'BurnedInAnnotation', '')).upper() == 'YES':
        notes.append("на самому зображенні можуть бути надруковані дані пацієнта — "
                     "перевірте, перш ніж показувати стороннім")
    txt = "Файл DICOM завантажено, особисті дані з нього не використовуються."
    if notes:
        #порожній рядок між попередженнями: одиничний перенос Markdown злив би їх в один абзац
        txt += "".join(chr(10) + chr(10) + "**Увага:** " + n for n in notes)
    return img, txt


#очищає поле файлу DICOM і текст під ним
def _reset_dicom():
    return None, ""


#новий знімок або інший режим порога: результати попереднього аналізу прибираються
#з усіх вкладок, щоб біля нового знімка не стояв чужий результат.
#порядок виходів: діагностика (список, карта, текст), таблиця критичності, порівняння (два списки, текст)
def _clear_results(pil):
    hint = "Завантажте знімок." if pil is None else "Натисніть «Аналізувати»."
    row = [["—", hint, "—", "—", "—", ""]]
    return {}, None, hint, row, {}, {}, hint


#вкладка «Діагностика»: ймовірності, сила сигналу і Grad-CAM для найсильнішого класу
def diagnose(pil, mode_label):
    if pil is None:
        return {}, None, "Завантажте знімок."
    model = M_CA
    thr = _thr(mode_label)
    b_weak, b_moderate, b_strong = _bands(mode_label)
    p = _probs(model, pil)
    #кратність: у скільки разів ймовірність більша за поріг свого класу
    ratio = p / thr
    #клас з найбільшою кратністю визначає заголовок і теплову карту
    top = int(np.argmax(ratio))
    r = ratio[top]

    #заголовок і пояснення залежать від того, між якими межами сигналу лежить кратність
    if r < b_weak:
        head, note = ("Виражених ознак не виявлено",
                      "Найвищий сигнал x%s — нижче межі, типовий для знімка без патології. "
                      "Модель не може підтвердити відсутність патології, лише не бачить ознак." % _num(r, 1))
    elif r < b_moderate:
        head, note = ("Слабкий сигнал: %s" % UA[top],
                      "Кратність x%s — потребує перевірки лікарем." % _num(r, 1))
    elif r < b_strong:
        head, note = ("Помірний сигнал: %s" % UA[top],
                      "Кратність x%s — вище типового рівня." % _num(r, 1))
    else:
        head, note = ("Виражений сигнал: %s" % UA[top],
                      "Кратність x%s — значно вище порога класу." % _num(r, 1))

    #без жодного сигналу карта показувала б випадкову ділянку, наприклад живіт,
    #і могла б ввести лікаря в оману, тому вона будується лише тоді, коли є що показати
    cam = _gradcam(model, pil, top) if r >= b_weak else None
    #усі патології, що перевищують межу слабкого сигналу, від найсильнішої до найслабшої
    above = [UA[i] for i in np.argsort(ratio)[::-1] if ratio[i] >= b_weak]
    #chr(10) — символ нового рядка
    nl = chr(10)
    txt = "### %s" % head + nl + note + nl + nl
    txt += ("Сигнал вище межі (кратність від %s): **%s**" % (_num(b_weak), ', '.join(above))
            if above else "_Жодна патологія не перевищує межі сигналу, тому теплова карта не будується._")
    return _ranked(p, thr), cam, txt


#вкладка «Критичність»: таблиця патологій, упорядкована за пріоритетом (p / поріг) x k_c;
#ks — значення 14 повзунків критичності
def priority(pil, mode_label, *ks):
    if pil is None:
        return [["—", "завантажте знімок на вкладці «Діагностика»", "—", "—", "—", ""]]
    thr = _thr(mode_label)
    p = _probs(M_CA, pil)
    rows = []
    for i, c in enumerate(CLASS_NAMES):
        #пріоритет — у скільки разів ймовірність перевищує власний поріг класу,
        #підсилене клінічною критичністю: небезпечні патології піднімаються вище.
        #позитивний результат лишається прив'язаним до порога класу, інакше спрацьовують майже всі 14
        rank = p[i] / thr[i] * ks[i]
        alarm = p[i] >= thr[i]
        #перші два значення потрібні лише для сортування, у таблицю вони не потрапляють
        rows.append([alarm, rank, UA[i], _prob(p[i]), _prob(thr[i]), _num(ks[i], 1),
                     _num(rank), "Позитивний" if alarm else "—"])
    #спершу патології з ознаками, і вже серед них критичність вирішує порядок:
    #інакше клас без ознак із високим k_c обганяв би клас, де ознаки є
    rows.sort(key=lambda r: (not r[0], -r[1]))
    return [r[2:] for r in rows]


#вкладка «Порівняння моделей»: ймовірності CA-CB-Focal і BCE та список розбіжностей
def compare(pil, mode_label):
    if pil is None:
        return {}, {}, "Завантажте знімок на вкладці «Діагностика»."
    if M_BCE is None:
        return {}, {}, "Немає файлу model_BCE.pth — порівняння недоступне."
    #кожна модель оцінюється за власними порогами: у них різний масштаб ймовірностей,
    #тож спільний поріг поставив би базову модель у програшне становище
    thr_ca, thr_bce = _thr(mode_label, 'CA-CB-Focal'), _thr(mode_label, 'BCE')
    p_ca, p_bce = _probs(M_CA, pil), _probs(M_BCE, pil)
    #позитивний результат кожної моделі для кожної з 14 патологій (True або False)
    al_ca, al_bce = p_ca >= thr_ca, p_bce >= thr_bce

    #де моделі згодні, і де позитивний результат дала лише одна з них
    both = [UA[i] for i in range(14) if al_ca[i] and al_bce[i]]
    only_ca = [UA[i] for i in range(14) if al_ca[i] and not al_bce[i]]
    only_bce = [UA[i] for i in range(14) if al_bce[i] and not al_ca[i]]
    nl = chr(10)
    txt = ("**Позитивний результат в обох моделях:** " + (', '.join(both) or '—') + nl + nl +
           "**Лише в CA-CB-Focal:** " + (', '.join(only_ca) or '—') + nl + nl +
           "**Лише в базовій BCE:** " + (', '.join(only_bce) or '—'))
    return _ranked(p_ca, thr_ca), _ranked(p_bce, thr_bce), txt


#зовнішній вигляд: готова тема Soft з бірюзовими кнопками і заокругленими полями.
#шрифт системний (Segoe UI є в кожній Windows і має українські літери), а не з Google Fonts:
#інакше сторінка щоразу завантажувала б шрифт з інтернету
_THEME = gr.themes.Soft(
    primary_hue="teal", secondary_hue="blue", neutral_hue="slate",
    font=["Segoe UI", "system-ui", "sans-serif"],
).set(
    block_border_width="1px",
    #block_shadow="0 1px 2px rgba(0,0,0,0.05)",
    block_label_background_fill="*neutral_100",
    block_label_background_fill_dark="*neutral_800",
    block_label_text_weight="600",
    button_large_radius="12px",
    input_radius="10px",
)

#власні стилі: смуга прокрутки завжди на місці (сторінка не «стрибає» між вкладками),
#ширина вмісту до 1150 пікселів, прихований нижній підпис Gradio і відсотки біля смужок
#у списках ймовірностей (.confidence): там Gradio показує довжину смужки, а не ймовірність
_CSS = """
html {overflow-y: scroll; scrollbar-gutter: stable;}
.gradio-container {width: 100% !important; max-width: 1150px !important; margin: 0 auto !important; padding-top: 14px;}
.gradio-container .tabs, .gradio-container .tabitem {width: 100% !important;}
footer {display: none !important;}
.confidence {display: none !important;}
.tabitem {padding-top: 12px;}
.block {overflow: hidden;}
"""

#тимчасові файли Gradio (копії знімків, картинки Grad-CAM): раз на годину видаляються ті,
#яким більше доби, а під час закриття застосунку — усі. доба, а не кілька хвилин, бо поки
#знімок відкрито, Gradio щоразу читає його з копії, і передчасне видалення зламало б аналіз
CACHE_CHECK_S, CACHE_AGE_S = 3600, 86400

#опис інтерфейсу: кожен блок with — це контейнер (вкладка, рядок, стовпець, група),
#а все, що створено всередині, з'являється в ньому
with gr.Blocks(title="Аналіз рентгенів", delete_cache=(CACHE_CHECK_S, CACHE_AGE_S)) as demo:
    with gr.Tab("Діагностика"):
        with gr.Row(equal_height=False):
            #ліва частина: вибір режиму, знімок, файл DICOM і кнопка
            with gr.Column(scale=5):
                with gr.Group():
                    mode = gr.Radio(list(MODES), value=MODE_DEFAULT, label="Режим порога")
                    #type="pil": функції отримують знімок як зображення Pillow
                    #format="png": знімок з DICOM Gradio зберігає у форматі поля, а типовий webp
                    #стискає з втратами, і модель отримувала б уже змінене зображення
                    inp = gr.Image(type="pil", format="png", label="Рентген грудної клітки", height=380)
                    #без обмеження розширень: на дисках від апаратів файли DICOM часто без .dcm
                    #height: порожнє поле не таке високе, і кнопка «Аналізувати» видна без прокрутки
                    dcm = gr.File(label="Або файл DICOM з рентген-апарата", type="filepath", height=180)
                    dcm_info = gr.Markdown()
                #аналіз запускається лише цією кнопкою: лікар спершу вибирає знімок і режим порога
                btn = gr.Button("Аналізувати", variant="primary", size="lg")
            #права частина: результати
            with gr.Column(scale=6):
                with gr.Group():
                    #показує 6 патологій з найдовшими смужками
                    lbl = gr.Label(num_top_classes=6, label="Ймовірності патологій", show_heading=False)
                with gr.Group():
                    cam = gr.Image(label="Grad-CAM — область уваги моделі", height=300)
                info = gr.Markdown()
        #після вибору файлу DICOM його зображення потрапляє в поле знімка
        dcm.upload(load_dicom, dcm, [inp, dcm_info])
        #лікар сам завантажив інший знімок або прибрав його: назва попереднього файлу DICOM
        #і попередження до нього вже не стосуються того, що в полі, тому їх прибираємо.
        #upload спрацьовує лише на дію лікаря, а не тоді, коли знімок підставляє load_dicom
        inp.upload(_reset_dicom, None, [dcm, dcm_info])
        inp.clear(_reset_dicom, None, [dcm, dcm_info])

    with gr.Tab("Критичність (k_c)"):
        gr.Markdown("Базовий поріг кожної патології підібрано на валідаційній вибірці за режимом, "
                    "обраним на вкладці «Діагностика». Лікар задає **критичність**, і вона визначає "
                    "порядок перегляду: **пріоритет = (ймовірність ÷ базовий поріг) × k_c**. "
                    "Небезпечні патології піднімаються вгору списку навіть за нижчої ймовірності. "
                    "*(спершу проаналізуйте знімок на вкладці «Діагностика»)*")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    #14 повзунків від 1 до 3 з кроком 0,5; початкові значення — з config.py
                    ks = [gr.Slider(1.0, 3.0, K_CRITICALITY[c], step=0.5, label=UA_LABELS[c])
                          for c in CLASS_NAMES]
                prio_btn = gr.Button("Оцінити пріоритети", variant="primary")
            with gr.Column(scale=6):
                #ширини колонок у відсотках, а wrap переносить довгі заголовки на другий рядок:
                #без цього «Базовий поріг» обрізався, а «Результат» ховався за прокруткою
                out = gr.Dataframe(headers=["Патологія", "Ймовірність", "Базовий поріг",
                                            "k_c", "Пріоритет", "Результат"],
                                   column_widths=["26%", "15%", "15%", "10%", "15%", "19%"], wrap=True)
        #таблиця перераховується за кнопкою і щоразу, коли лікар відпускає повзунок
        prio_btn.click(priority, [inp, mode] + ks, out)
        for k in ks:
            k.release(priority, [inp, mode] + ks, out)

    with gr.Tab("Порівняння моделей"):
        gr.Markdown("Той самий знімок аналізують дві моделі: **CA-CB-Focal** (запропонований метод) "
                    "і **базова BCE** (звичайна функція втрат). Кожна оцінюється за власними порогами "
                    "в обраному режимі. *(спершу завантажте знімок на вкладці «Діагностика»)*")
        with gr.Row():
            with gr.Group():
                lbl_ca = gr.Label(num_top_classes=6, label="CA-CB-Focal", show_heading=False)
            with gr.Group():
                lbl_bce = gr.Label(num_top_classes=6, label="Базова BCE", show_heading=False)
        cmp_info = gr.Markdown()

    #запис подія(функція, входи, виходи): коли сталася подія, взяти значення входів,
    #викликати функцію і її результат показати у виходах.
    #кнопка «Аналізувати» запускає всі три обчислення, і кожне оновлює свою вкладку
    btn.click(diagnose, [inp, mode], [lbl, cam, info])
    btn.click(priority, [inp, mode] + ks, out)
    btn.click(compare, [inp, mode], [lbl_ca, lbl_bce, cmp_info])
    #новий знімок або інший режим: старі результати прибираються з усіх вкладок
    #до наступного натискання кнопки, інакше вони стояли б біля нового знімка
    for trigger in (inp.change, mode.change):
        trigger(_clear_results, inp, [lbl, cam, info, out, lbl_ca, lbl_bce, cmp_info])


#виконується лише під час прямого запуску файлу; тести імпортують app без відкриття вікна
if __name__ == '__main__':
    import webview
    #вебсервер Gradio лише на цьому комп'ютері (share=False — без публічного посилання);
    #prevent_thread_lock не блокує програму, щоб далі можна було відкрити вікно
    demo.launch(share=False, inbrowser=False, prevent_thread_lock=True, theme=_THEME, css=_CSS)
    #власне вікно застосунку показує сторінку сервера замість браузера;
    #__theme=light вмикає світлу тему незалежно від налаштувань Windows;
    #zoomable дозволяє змінювати масштаб клавішами Ctrl і плюс або мінус
    webview.create_window("Аналіз рентгенів", demo.local_url + '?__theme=light',
                          width=1400, height=900, min_size=(1000, 700), zoomable=True)
    #працює, доки вікно відкрите; після закриття зупиняємо сервер
    webview.start()
    demo.close()
