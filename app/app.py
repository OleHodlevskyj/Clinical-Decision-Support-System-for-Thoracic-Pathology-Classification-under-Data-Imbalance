import os
import sys
import glob
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import torch
import gradio as gr
import pydicom
from pydicom.pixels import apply_voi_lut
from torchvision import transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (CLASS_NAMES, UA_LABELS, K_CRITICALITY,
                    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE)
from model import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = os.environ.get('MODELS_DIR',
                            os.path.join(os.path.dirname(__file__), '..', 'models'))
UA = [UA_LABELS[c] for c in CLASS_NAMES]

#завантаження моделей
_mfiles = {os.path.basename(f)[6:-4]: f
           for f in glob.glob(os.path.join(MODELS_DIR, 'model_*.pth'))}
assert _mfiles, f'Моделі не знайдено в {MODELS_DIR} (потрібні model_*.pth)'


def _load(name):
    #архітектура задана явно: Grad-CAM нижче чіпляється до layer4, це шар ResNet
    m = build_model(pretrained=False, arch='resnet50').to(device)
    m.load_state_dict(torch.load(_mfiles[name], map_location=device))
    m.eval()
    return m


M_CA = _load('CA-CB-Focal')
M_BCE = _load('BCE') if 'BCE' in _mfiles else None

#по-класові пороги за двома критеріями (src/make_thresholds.py, валідаційна вибірка):
#індекс Юдена — на користь чутливості, максимум F1-міри — на користь точності тривог
_THR_FILE = os.path.join(MODELS_DIR, 'thresholds.json')
_THR_ALL = {}
if os.path.exists(_THR_FILE):
    with open(_THR_FILE, encoding='utf-8') as _fh:
        _THR_ALL = json.load(_fh)

MODES = {'Не пропустити хворого (індекс Юдена)': 'youden',
         'Менше хибних тривог (максимум F1-міри)': 'f1'}
MODE_DEFAULT = list(MODES)[0]


def _thr(mode_label, cfg='CA-CB-Focal'):
    return np.array(_THR_ALL.get(MODES[mode_label], {}).get(cfg, [0.5] * 14))


def _bands(mode_label):
    #межі сили сигналу — з розподілу максимальної кратності p/поріг на валідації
    return _THR_ALL.get('bands', {}).get(MODES[mode_label], [1.35, 1.75, 2.2])


_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                 T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _probs(model, pil):
    x = _tf(pil.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(x))[0].cpu().numpy()


def _gradcam(model, pil, cls):
    x = _tf(pil.convert('RGB')).unsqueeze(0).to(device)
    store = {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__('a', o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: store.__setitem__('g', go[0]))
    out = model(x)
    model.zero_grad()
    out[0, cls].backward()
    A, G = store['a'][0], store['g'][0]
    cam = torch.relu((G.mean(dim=(1, 2))[:, None, None] * A).sum(0))
    cam = cam / (cam.max() + 1e-8)
    h1.remove(); h2.remove()
    cam = np.array(Image.fromarray((cam.detach().cpu().numpy() * 255).astype('uint8'))
                   .resize((IMG_SIZE, IMG_SIZE))) / 255.
    base = np.array(pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(base); ax.imshow(cam, cmap='jet', alpha=0.45); ax.axis('off')
    fig.canvas.draw()
    out_img = Image.fromarray(np.array(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return out_img


def _ranked(p, thr):
    #для gr.Label: підпис із кратністю і довжина смужки; ключі мають бути унікальні
    ratio = p / thr
    return {'%s  x%.2f  (p=%.2f / поріг %.2f)' % (UA[i], ratio[i], p[i], thr[i]):
            float(min(ratio[i] / 3, 1.0)) for i in range(14)}


def load_dicom(path):
    #з файлу береться лише зображення: ім'я пацієнта, дата народження, номер картки
    #та інші поля DICOM не читаються й ніде не показуються
    if not path:
        return gr.update(), ""
    try:
        ds = pydicom.dcmread(path)
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
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(-1)
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
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    img = Image.fromarray(arr.astype('uint8')).convert('RGB')

    view = str(getattr(ds, 'ViewPosition', '')).upper()
    if view in ('LL', 'RL', 'LAT', 'LATERAL'):
        notes.append("бічна проекція — модель навчалась лише на прямих, результат ненадійний")
    if str(getattr(ds, 'BurnedInAnnotation', '')).upper() == 'YES':
        notes.append("на самому зображенні можуть бути надруковані дані пацієнта — "
                     "перевірте, перш ніж показувати стороннім")
    txt = "Файл DICOM завантажено, особисті дані з нього не використовуються."
    if notes:
        #порожній рядок між попередженнями: одиничний перенос Markdown злив би їх в один абзац
        txt += "".join(chr(10) + chr(10) + "**Увага:** " + n for n in notes)
    return img, txt


def diagnose(pil, mode_label):
    if pil is None:
        return {}, None, "Завантажте знімок."
    model = M_CA
    thr = _thr(mode_label)
    b_weak, b_moderate, b_strong = _bands(mode_label)
    p = _probs(model, pil)
    ratio = p / thr
    top = int(np.argmax(ratio))
    r = ratio[top]

    if r < b_weak:
        head, note = ("Виражених ознак не виявлено",
                      "Найвищий сигнал x%.1f — нижче межі, типовий для знімка без патології. "
                      "Модель не може підтвердити відсутність патології, лише не бачить ознак." % r)
    elif r < b_moderate:
        head, note = ("Слабкий сигнал: %s" % UA[top],
                      "Кратність x%.1f — потребує перевірки лікарем." % r)
    elif r < b_strong:
        head, note = ("Помірний сигнал: %s" % UA[top],
                      "Кратність x%.1f — вище типового рівня." % r)
    else:
        head, note = ("Виражений сигнал: %s" % UA[top],
                      "Кратність x%.1f — значно вище порога класу." % r)

    cam = _gradcam(model, pil, top)
    above = [UA[i] for i in np.argsort(ratio)[::-1] if ratio[i] >= b_weak]
    nl = chr(10)
    txt = "### %s" % head + nl + note + nl + nl
    txt += ("Сигнал вище межі (кратність від %.2f): **%s**" % (b_weak, ', '.join(above))
            if above else "_Жодна патологія не перевищує межі сигналу._")
    return _ranked(p, thr), cam, txt


def priority(pil, mode_label, *ks):
    if pil is None:
        return [["—", "завантажте знімок на вкладці «Діагностика»", "—", "—", "—", ""]]
    thr = _thr(mode_label)
    p = _probs(M_CA, pil)
    rows = []
    for i, c in enumerate(CLASS_NAMES):
        #пріоритет — у скільки разів ймовірність перевищує власний поріг класу,
        #підсилене клінічною критичністю: небезпечні патології піднімаються вище.
        #тривога лишається прив'язаною до порога класу, інакше спрацьовують майже всі 14
        rank = p[i] / thr[i] * ks[i]
        alarm = p[i] >= thr[i]
        rows.append([alarm, rank, UA[i], f"{p[i]:.2f}", f"{thr[i]:.2f}", f"{ks[i]:.1f}",
                     f"{rank:.2f}", "ТРИВОГА" if alarm else "—"])
    #спершу патології з ознаками, і вже серед них критичність вирішує порядок:
    #інакше клас без ознак із високим k_c обганяв би клас, де ознаки є
    rows.sort(key=lambda r: (not r[0], -r[1]))
    return [r[2:] for r in rows]


def compare(pil, mode_label):
    if pil is None:
        return {}, {}, "Завантажте знімок на вкладці «Діагностика»."
    if M_BCE is None:
        return {}, {}, "Немає файлу model_BCE.pth — порівняння недоступне."
    #кожна модель оцінюється за власними порогами: у них різний масштаб ймовірностей,
    #тож спільний поріг поставив би базову модель у програшне становище
    thr_ca, thr_bce = _thr(mode_label, 'CA-CB-Focal'), _thr(mode_label, 'BCE')
    p_ca, p_bce = _probs(M_CA, pil), _probs(M_BCE, pil)
    al_ca, al_bce = p_ca >= thr_ca, p_bce >= thr_bce

    both = [UA[i] for i in range(14) if al_ca[i] and al_bce[i]]
    only_ca = [UA[i] for i in range(14) if al_ca[i] and not al_bce[i]]
    only_bce = [UA[i] for i in range(14) if al_bce[i] and not al_ca[i]]
    nl = chr(10)
    txt = ("**Тривога в обох моделях:** " + (', '.join(both) or '—') + nl + nl +
           "**Лише в CA-CB-Focal:** " + (', '.join(only_ca) or '—') + nl + nl +
           "**Лише в базовій BCE:** " + (', '.join(only_bce) or '—'))
    return _ranked(p_ca, thr_ca), _ranked(p_bce, thr_bce), txt


_THEME = gr.themes.Soft(
    primary_hue="teal", secondary_hue="blue", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    block_border_width="1px",
    #block_shadow="0 1px 2px rgba(0,0,0,0.05)",
    block_label_background_fill="*neutral_100",
    block_label_background_fill_dark="*neutral_800",
    block_label_text_weight="600",
    button_large_radius="12px",
    input_radius="10px",
)

_CSS = """
html {overflow-y: scroll; scrollbar-gutter: stable;}
.gradio-container {width: 100% !important; max-width: 1150px !important; margin: 0 auto !important; padding-top: 14px;}
.gradio-container .tabs, .gradio-container .tabitem {width: 100% !important;}
footer {display: none !important;}
.tabitem {padding-top: 12px;}
.block {overflow: hidden;}
"""

with gr.Blocks(title="Аналіз рентгенів") as demo:
    with gr.Tab("Діагностика"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Group():
                    mode = gr.Radio(list(MODES), value=MODE_DEFAULT, label="Режим порога")
                    inp = gr.Image(type="pil", label="Рентген грудної клітки", height=380)
                    #без обмеження розширень: на дисках від апаратів файли DICOM часто без .dcm
                    dcm = gr.File(label="Або файл DICOM з рентген-апарата", type="filepath")
                    dcm_info = gr.Markdown()
                btn = gr.Button("Аналізувати", variant="primary", size="lg")
            with gr.Column(scale=6):
                with gr.Group():
                    lbl = gr.Label(num_top_classes=6, label="Ймовірності патологій")
                with gr.Group():
                    cam = gr.Image(label="Grad-CAM — область уваги моделі", height=300)
                info = gr.Markdown()
        dcm.upload(load_dicom, dcm, [inp, dcm_info])

    with gr.Tab("Критичність (k_c)"):
        gr.Markdown("Базовий поріг кожної патології підібрано на валідаційній вибірці за режимом, "
                    "обраним на вкладці «Діагностика». Лікар задає **критичність**, і вона визначає "
                    "порядок перегляду: **пріоритет = (ймовірність ÷ базовий поріг) × k_c**. "
                    "Небезпечні патології піднімаються вгору списку навіть за нижчої ймовірності. "
                    "*(спершу проаналізуйте знімок на вкладці «Діагностика»)*")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    ks = [gr.Slider(1.0, 3.0, K_CRITICALITY[c], step=0.5, label=UA_LABELS[c])
                          for c in CLASS_NAMES]
                prio_btn = gr.Button("Оцінити пріоритети", variant="primary")
            with gr.Column(scale=6):
                out = gr.Dataframe(headers=["Патологія", "Ймовірність", "Базовий поріг",
                                            "k_c", "Пріоритет", "Статус"])
        prio_btn.click(priority, [inp, mode] + ks, out)
        for k in ks:
            k.release(priority, [inp, mode] + ks, out)

    with gr.Tab("Порівняння моделей"):
        gr.Markdown("Той самий знімок аналізують дві моделі: **CA-CB-Focal** (запропонований метод) "
                    "і **базова BCE** (звичайна функція втрат). Кожна оцінюється за власними порогами "
                    "в обраному режимі. *(спершу завантажте знімок на вкладці «Діагностика»)*")
        with gr.Row():
            with gr.Group():
                lbl_ca = gr.Label(num_top_classes=6, label="CA-CB-Focal")
            with gr.Group():
                lbl_bce = gr.Label(num_top_classes=6, label="Базова BCE")
        cmp_info = gr.Markdown()

    #усі вкладки оновлюються разом зі знімком і режимом порога, інакше показують
    #результати попереднього зображення або попереднього режиму
    btn.click(diagnose, [inp, mode], [lbl, cam, info])
    mode.change(diagnose, [inp, mode], [lbl, cam, info])
    for trigger in (btn.click, inp.change, mode.change):
        trigger(priority, [inp, mode] + ks, out)
        trigger(compare, [inp, mode], [lbl_ca, lbl_bce, cmp_info])


if __name__ == '__main__':
    import webview
    demo.launch(share=False, inbrowser=False, prevent_thread_lock=True, theme=_THEME, css=_CSS)
    webview.create_window("Аналіз рентгенів", demo.local_url, width=1400, height=900,
                          min_size=(1000, 700))
    webview.start()
    demo.close()
