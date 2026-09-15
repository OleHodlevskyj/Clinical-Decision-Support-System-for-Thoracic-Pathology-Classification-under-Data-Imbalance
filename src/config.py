"""Конфігурація проєкту: класи патологій, коефіцієнти критичності, гіперпараметри."""

# Порядок 14 класів на ВИХОДІ моделі (за спаданням частоти в NIH ChestX-ray14).
# ВАЖЛИВО: індекс виходу моделі відповідає цьому порядку — не змінювати.
CLASS_NAMES = [
    'Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass',
    'Pneumothorax', 'Consolidation', 'Pleural_Thickening', 'Cardiomegaly',
    'Emphysema', 'Edema', 'Fibrosis', 'Pneumonia', 'Hernia',
]

# Українські підписи для інтерфейсу
UA_LABELS = {
    'Infiltration': 'Інфільтрація', 'Effusion': 'Випіт', 'Atelectasis': 'Ателектаз',
    'Nodule': 'Вузол', 'Mass': 'Новоутворення', 'Pneumothorax': 'Пневмоторакс',
    'Consolidation': 'Консолідація', 'Pleural_Thickening': 'Потовщення плеври',
    'Cardiomegaly': 'Кардіомегалія', 'Emphysema': 'Емфізема', 'Edema': 'Набряк легень',
    'Fibrosis': 'Фіброз', 'Pneumonia': 'Пневмонія', 'Hernia': 'Грижа',
}

# Коефіцієнти клінічної критичності k_c (розділ 2.4, таблиця 2.2)
K_CRITICALITY = {
    'Pneumothorax': 3.0, 'Effusion': 3.0,                              # невідкладні
    'Edema': 2.0, 'Consolidation': 2.0, 'Mass': 2.0, 'Pneumonia': 2.0,  # серйозні
    'Atelectasis': 1.5, 'Cardiomegaly': 1.5, 'Nodule': 1.5,           # помірні
    'Emphysema': 1.5, 'Fibrosis': 1.5, 'Hernia': 1.5,
    'Infiltration': 1.0, 'Pleural_Thickening': 1.0,                   # часті
}

# Гіперпараметри CA-CB-Focal (розділ 2.5)
BETA = 0.9999     # швидкість насичення інформативності ваг
GAMMA0 = 2.0      # базове фокусування
LAMBDA = 0.5      # зв'язок критичності з фокусуванням

# Препроцесинг
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
