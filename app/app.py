"""
Дві вкладки:
  1. Діагностика       — знімок → ймовірності 14 патологій + Grad-CAM;
  2. Панель критичності — лікар задає k_c, система показує пороги тривоги.
Потрібні файли моделей у ../models/ (model_CA-CB-Focal.pth).

"""
import os
import sys
import glob
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import torch
import gradio as gr
from torchvision import transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (CLASS_NAMES, UA_LABELS, K_CRITICALITY,
                    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE)
from model import build_resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = os.environ.get('MODELS_DIR',
                            os.path.join(os.path.dirname(__file__), '..', 'models'))
UA = [UA_LABELS[c] for c in CLASS_NAMES]

#завантаження моделей
_mfiles = {os.path.basename(f)[6:-4]: f
           for f in glob.glob(os.path.join(MODELS_DIR, 'model_*.pth'))}
assert _mfiles, f'Моделі не знайдено в {MODELS_DIR} (потрібні model_*.pth)'


def _load(name):
    m = build_resnet50(pretrained=False).to(device)
    m.load_state_dict(torch.load(_mfiles[name], map_location=device))
    m.eval()
    return m


M_CA = _load('CA-CB-Focal')

#варіанти моделі для перемикача: оригінальна + донавчена (якщо є файл)
MODEL_OPTIONS = {'Оригінальна (NIH, дорослі)': M_CA}
if 'pneumonia_adapted' in _mfiles:
    MODEL_OPTIONS['Донавчена (пневмонія, діти)'] = _load('pneumonia_adapted')

#по-класові пороги класифікації (індекс Юдена, формула 2.16) ---
_THR_FILE = os.path.join(MODELS_DIR, 'thresholds.json')
_THR_ALL = {}
if os.path.exists(_THR_FILE):
    with open(_THR_FILE, encoding='utf-8') as _fh:
        _THR_ALL = json.load(_fh)
_THR = np.array(_THR_ALL.get('CA-CB-Focal', [0.5] * 14))


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


def diagnose(pil, model_choice):
    if pil is None:
        return {}, None, "Завантажте знімок."
    model = MODEL_OPTIONS.get(model_choice, M_CA)
    p = _probs(model, pil)
    ratio = p / _THR
    top = int(np.argmax(ratio))
    r = ratio[top]

    #сила сигналу. пороги взято з розподілу максимальної кратності на тесті:
    #для знімків без патологій медіана 1.28, для знімків із патологією 1.42 —
    #розподіли сильно перекриваються, тому нижче 1.2 стверджувати щось не можна.
    if r < 1.2:
        head, note = ("Виражених ознак не виявлено",
                      "Найвищий сигнал слабкий (x%.1f) — типовий для знімка без патології. "
                      "Модель не може підтвердити відсутність патології, лише не бачить ознак." % r)
    elif r < 1.5:
        head, note = ("Слабкий сигнал: %s" % UA[top],
                      "Кратність x%.1f. На рівні типового знімка без патології — "
                      "потребує перевірки лікарем." % r)
    elif r < 2.0:
        head, note = ("Помірний сигнал: %s" % UA[top],
                      "Кратність x%.1f — вище типового рівня." % r)
    else:
        head, note = ("Виражений сигнал: %s" % UA[top],
                      "Кратність x%.1f — значно вище порога класу." % r)

    conf = {'%s  x%.2f  (p=%.2f / поріг %.2f)' % (UA[i], ratio[i], p[i], _THR[i]):
            float(min(ratio[i] / 3, 1.0)) for i in range(14)}
    cam = _gradcam(model, pil, top)
    above = [UA[i] for i in np.argsort(ratio)[::-1] if ratio[i] >= 1.2]
    nl = chr(10)
    txt = "### %s" % head + nl + note + nl + nl
    txt += ("Перевищують свій поріг у 1.2 раза й більше: **%s**" % ', '.join(above)
            if above else "_Жодна патологія не перевищує свого порога у 1.2 раза._")
    return conf, cam, txt


def priority(pil, base_thr, *ks):
    if pil is None:
        return [["—", "завантажте знімок на вкладці «Діагностика»", "—", "—", ""]]
    p = _probs(M_CA, pil)
    rows = []
    for i, c in enumerate(CLASS_NAMES):
        eff = min(base_thr / ks[i], 0.99)     # вищий k_c → нижчий поріг → чутливіше
        rows.append([UA[i], f"{p[i]:.2f}", f"{ks[i]:.1f}", f"{eff:.2f}",
                     "ТРИВОГА" if p[i] >= eff else "—"])
    return rows


_THEME = gr.themes.Soft(
    primary_hue="teal", secondary_hue="blue", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    block_border_width="1px",
    block_radius="14px",
    block_shadow="0 1px 2px rgba(0,0,0,0.05)",
    block_label_background_fill="*neutral_100",
    block_label_background_fill_dark="*neutral_800",
    block_label_text_weight="600",
    button_large_radius="12px",
    input_radius="10px",
)

_CSS = """
.gradio-container {max-width: 1150px !important; margin: 0 auto !important; padding-top: 14px;}
footer {display: none !important;}
.tabitem {padding-top: 12px;}
.block {overflow: hidden;}
"""

with gr.Blocks(title="Аналіз рентгенів", theme=_THEME, css=_CSS) as demo:
    with gr.Tab("Діагностика"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Group():
                    model_sel = gr.Dropdown(
                        list(MODEL_OPTIONS.keys()),
                        value=list(MODEL_OPTIONS.keys())[0],
                        label="Модель")
                    inp = gr.Image(type="pil", label="Рентген грудної клітки", height=380)
                btn = gr.Button("Аналізувати", variant="primary", size="lg")
            with gr.Column(scale=6):
                with gr.Group():
                    lbl = gr.Label(num_top_classes=6, label="Ймовірності патологій")
                with gr.Group():
                    cam = gr.Image(label="Grad-CAM — область уваги моделі", height=300)
                info = gr.Markdown()
        btn.click(diagnose, [inp, model_sel], [lbl, cam, info])

    with gr.Tab("Критичність (k_c)"):
        gr.Markdown("Лікар задає **критичність** патології. Вищий k_c → нижчий поріг "
                    "тривоги → чутливіше. *(спершу проаналізуйте знімок на вкладці «Діагностика»)*")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    thr = gr.Slider(0.1, 0.9, 0.5, label="Базовий поріг тривоги")
                    ks = [gr.Slider(1.0, 3.0, K_CRITICALITY[c], step=0.5, label=UA_LABELS[c])
                          for c in CLASS_NAMES]
                prio_btn = gr.Button("Оцінити пріоритети", variant="primary")
            with gr.Column(scale=6):
                out = gr.Dataframe(headers=["Патологія", "Ймовірність", "k_c", "Поріг", "Статус"])
        prio_btn.click(priority, [inp, thr] + ks, out)
        thr.release(priority, [inp, thr] + ks, out)
        for k in ks:
            k.release(priority, [inp, thr] + ks, out)


if __name__ == '__main__':
    demo.launch(share=True)
