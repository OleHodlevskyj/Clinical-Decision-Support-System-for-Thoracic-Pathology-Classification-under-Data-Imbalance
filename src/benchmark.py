#заміри швидкодії застосунку (п. 3.4): час відповіді, пропускна здатність, пам'ять і процесор.
#сервер запускається окремим процесом, запити йдуть справжнім HTTP через gradio_client,
#тому враховано все, що проходить реальний запит: мережевий рівень, чергу Gradio, модель.
#запуск: .venv\Scripts\python src\benchmark.py   (результат — results\benchmark.json)
import os
import sys
import json
import time
import ctypes
import shutil
import argparse
import platform
import tempfile
import threading
import subprocess
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
#папки src, app і tests на початок шляхів пошуку: звідти імпортуються app, config і conftest
sys.path[:0] = [HERE, os.path.join(ROOT, 'app'), os.path.join(ROOT, 'tests')]

import app
from config import CLASS_NAMES, K_CRITICALITY
from conftest import write_dicom
from gradio_client import Client, handle_file

#окремий порт, щоб не заважати застосунку, якщо він відкритий
PORT = 7871
URL = f'http://127.0.0.1:{PORT}/'
OUT = os.path.join(ROOT, 'results', 'benchmark.json')
LOG = os.path.join(ROOT, 'results', 'benchmark_server.log')
#режим порога і типові значення критичності, з якими йдуть запити
MODE = list(app.MODES)[0]
KS = [K_CRITICALITY[c] for c in CLASS_NAMES]
#скільки користувачів одночасно надсилають запити в сценаріях навантаження
CLIENTS = (1, 2, 4, 8)
#розмір знімка NIH ChestX-ray14 і типовий розмір цифрового рентгена у файлі DICOM
PNG_SIZE = 1024
DICOM_SHAPE = (2500, 2048)
#перші виміри повільніші (прогрів кешів і бібліотек), тому вони відкидаються
WARMUP = 3


#забороняє Windows присипляти комп'ютер, поки триває замір
def keep_awake():
    #сон ноутбука під час заміру зіпсував би час відповіді. заборона діє, доки працює скрипт;
    #закрита кришка однаково присипляє ноутбук. 0x80000000 — діяти постійно, 0x1 — не засинати
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


#зведення ряду вимірів: кількість, середнє, відхилення, медіана (p50), 95-й і 99-й процентилі, максимум
def stats(values):
    v = np.asarray(values, dtype=float)
    return {'n': int(v.size), 'mean': float(v.mean()), 'std': float(v.std()),
            'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)),
            'p99': float(np.percentile(v, 99)), 'max': float(v.max())}


class Sampler:
    #раз на пів секунди записує пам'ять і завантаження процесора заданого процесу
    def __init__(self, pid):
        self.proc = psutil.Process(pid)
        self.rss, self.cpu, self.system = [], [], []
        self.stop = threading.Event()

    #працює в окремому потоці, доки не встановлено self.stop
    def _run(self):
        #python.exe із .venv на Windows — лише запускалка, справжній Python є її дочірнім
        #процесом, тому рахується все дерево; об'єкти кешуються, бо cpu_percent міряє
        #різницю від попереднього виклику на тому самому об'єкті
        procs = [self.proc] + self.proc.children(recursive=True)
        for p in procs:
            p.cpu_percent(None)
        psutil.cpu_percent(None)
        while not self.stop.wait(0.5):
            self.rss.append(sum(p.memory_info().rss for p in procs))
            self.cpu.append(sum(p.cpu_percent(None) for p in procs) / psutil.cpu_count())
            #завантаження всієї системи: показує вплив сторонніх програм на заміри
            self.system.append(psutil.cpu_percent(None))

    #with Sampler(...) as s: запис починається на вході в блок і зупиняється на виході
    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    #середня і пікова пам'ять у мегабайтах, середнє і пікове завантаження процесора у відсотках
    def result(self):
        return {'rss_mb_mean': float(np.mean(self.rss)) / 2 ** 20 if self.rss else None,
                'rss_mb_peak': float(np.max(self.rss)) / 2 ** 20 if self.rss else None,
                'cpu_pct_mean': float(np.mean(self.cpu)) if self.cpu else None,
                'cpu_pct_peak': float(np.max(self.cpu)) if self.cpu else None,
                'system_cpu_pct_mean': float(np.mean(self.system)) if self.system else None}


#запускає застосунок окремим процесом без вікна і чекає, доки сервер почне відповідати;
#повертає процес і час запуску
def start_server():
    #час запуску — від старту процесу до першої відповіді сервера
    code = (f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'app')!r}); import app; "
            f"app.demo.launch(server_name='127.0.0.1', server_port={PORT}, share=False, "
            f"inbrowser=False, quiet=True, show_error=True)")
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED='False', PYTHONIOENCODING='utf-8')
    #журнал сервера: якщо запит завершиться помилкою, тут буде її точне місце
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, 'w', encoding='utf-8')
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT, env=env,
                            stdout=log, stderr=log)
    while True:
        if proc.poll() is not None:
            raise SystemExit('сервер завершився під час запуску')
        try:
            urllib.request.urlopen(URL, timeout=2)
            return proc, time.perf_counter() - t0
        except OSError:
            time.sleep(0.2)


#новий клієнт, що надсилає запити серверу так само, як вікно застосунку
def client():
    #download_files=False: не зберігати на диск картинку Grad-CAM, це не робота застосунку
    return Client(URL, verbose=False, download_files=False)


#один запит до функції name (diagnose, priority або compare) зі знімком png;
#чекає, доки прийде відповідь
def call(c, name, png):
    args = {'diagnose': (handle_file(png), MODE),
            'priority': (handle_file(png), MODE, *KS),
            'compare': (handle_file(png), MODE)}[name]
    c.predict(*args, api_name='/' + name)


#робить count копій знімка, що відрізняються одним пікселем
def make_files(pil, count, folder):
    #gradio_client передає файл із кожним запитом, а Gradio кладе його в папку, названу
    #за вмістом: однаковий файл від двох запитів одночасно записувався б поверх того,
    #що читає третій. тому в кожного джерела запитів свій файл, різниця — один піксель.
    #у вікні застосунку такого немає: знімок завантажується один раз на всі три обчислення
    base = np.asarray(pil).copy()
    out = []
    for i in range(count):
        base[0, 0] = i
        path = os.path.join(folder, f'xray_{i}.png')
        Image.fromarray(base).save(path)
        out.append(path)
    return out


def full_click(files, n):
    #одне натискання «Аналізувати»: три обчислення одночасно, чекаємо останнє.
    #верхня оцінка: тут файл передається тричі, у вікні застосунку — один раз
    names = ('diagnose', 'priority', 'compare')
    clients = {k: client() for k in names}
    with ThreadPoolExecutor(len(names)) as pool:
        def once():
            t0 = time.perf_counter()
            list(pool.map(lambda k, f: call(clients[k], k, f), names, files))
            return time.perf_counter() - t0
        #спершу прогрів, потім n вимірів
        for _ in range(WARMUP):
            once()
        return stats([once() for _ in range(n)])


def load_level(files, users, n):
    #users незалежних користувачів надсилають запити без пауз, n запитів разом
    latency, errors = [], []
    lock = threading.Lock()
    per_user = max(1, n // users)
    cs = [client() for _ in range(users)]
    #один запит до початку заміру, щоб прогріти сервер
    call(cs[0], 'diagnose', files[0])

    #робота одного користувача: per_user запитів підряд; помилки рахуються, а не зупиняють замір
    def work(c, png):
        for _ in range(per_user):
            t0 = time.perf_counter()
            try:
                call(c, 'diagnose', png)
                with lock:
                    latency.append(time.perf_counter() - t0)
            except Exception as e:
                with lock:
                    errors.append(f'{type(e).__name__}: {str(e)[:200]}')

    t0 = time.perf_counter()
    with ThreadPoolExecutor(users) as pool:
        list(pool.map(work, cs, files))
    elapsed = time.perf_counter() - t0
    #rps — пропускна здатність: скільки запитів сервер обробив за секунду
    return {'users': users, 'latency': stats(latency), 'errors': len(errors),
            'error_kinds': dict(Counter(errors)), 'elapsed_s': elapsed,
            'rps': len(latency) / elapsed}


#час виконання функції fn: прогрів, потім n вимірів
def timed(fn, n):
    for _ in range(WARMUP):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return stats(out)


def direct_stages(pil, n, tmp):
    #окремі етапи без HTTP: показують, де саме витрачається час
    #файл DICOM створюється поза заміром, бо load_dicom видаляє його після читання
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 4096, DICOM_SHAPE, dtype=np.uint16)
    src = write_dicom(os.path.join(tmp, 'src.dcm'), pixels)
    dst = os.path.join(tmp, 'x.dcm')

    #час читання DICOM; копіювання файлу перед кожним читанням у замір не входить
    def dicom_times():
        out = []
        for i in range(WARMUP + max(10, n // 4)):
            shutil.copy(src, dst)
            t0 = time.perf_counter()
            app.load_dicom(dst)
            if i >= WARMUP:
                out.append(time.perf_counter() - t0)
        return stats(out)

    #клас для Grad-CAM — той самий, що вибрала б diagnose
    top = int(np.argmax(app._probs(app.M_CA, pil)))
    return {'dicom_read': dicom_times(),
            'inference': timed(lambda: app._probs(app.M_CA, pil), n),
            'gradcam': timed(lambda: app._gradcam(app.M_CA, pil, top), max(10, n // 4)),
            'diagnose': timed(lambda: app.diagnose(pil, MODE), max(10, n // 4))}


#послідовність заміру: тестовий знімок -> запуск сервера -> простій -> одне натискання ->
#навантаження 1, 2, 4, 8 користувачів -> окремі етапи без HTTP -> results\benchmark.json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100, help='кількість вимірів у кожному сценарії')
    args = ap.parse_args()
    keep_awake()

    tmp = tempfile.mkdtemp(prefix='bench_')
    png = os.path.join(tmp, 'xray.png')
    rng = np.random.default_rng(0)
    #справжні знімки не використовуються: плавні переходи яскравості зі слабким шумом
    #стискаються близько до рентгена, а чистий шум дав би завеликий файл для передавання
    coarse = Image.fromarray(rng.integers(0, 255, (16, 16), dtype=np.uint8))
    smooth = np.asarray(coarse.resize((PNG_SIZE, PNG_SIZE), Image.BICUBIC), dtype=float)
    noise = rng.normal(0, 4, smooth.shape)
    pil = Image.fromarray(np.clip(smooth + noise, 0, 255).astype(np.uint8))
    pil.save(png)
    #3 файли для одного натискання і по одному на кожного користувача навантаження
    files = make_files(pil, max(CLIENTS) + 3, tmp)

    #відомості про комп'ютер, на якому зроблено замір
    res = {'system': {'cpu': platform.processor(), 'logical_cores': psutil.cpu_count(),
                      'ram_gb': psutil.virtual_memory().total / 2 ** 30,
                      'gpu': app.torch.cuda.is_available(), 'torch': app.torch.__version__,
                      'torch_threads': app.torch.get_num_threads(), 'python': platform.python_version()},
           'n': args.n, 'png_kb': os.path.getsize(png) / 2 ** 10}

    proc, res['startup_s'] = start_server()
    try:
        #пауза, щоб сервер завершив запуск, потім 10 секунд запису пам'яті без запитів
        time.sleep(5)
        with Sampler(proc.pid) as idle:
            time.sleep(10)
        res['idle'] = idle.result()
        print('запуск і простій готово', flush=True)

        with Sampler(proc.pid) as s:
            res['full_click'] = full_click(files[:3], args.n)
        res['full_click']['resources'] = s.result()
        print('натискання готово', flush=True)

        res['load'] = []
        for users in CLIENTS:
            with Sampler(proc.pid) as s:
                level = load_level(files[3:], users, args.n)
            level['resources'] = s.result()
            res['load'].append(level)
            print(f'{users} користувачів: {level["rps"]:.2f} запитів/с, '
                  f'помилок {level["errors"]}', flush=True)
    finally:
        #сервер зупиняється навіть тоді, коли замір перервано помилкою
        proc.terminate()
        proc.wait()

    res['stages'] = direct_stages(pil.convert('RGB'), args.n, tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f'Збережено: {OUT}')


if __name__ == '__main__':
    main()
