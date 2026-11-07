#читання знімків NIH ChestX-ray14 і їх підготовка для моделі
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

from config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

#нормалізація тими самими числами, що й під час попереднього навчання на ImageNet
_norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

#аугментація: під час навчання кожен знімок щоразу трохи змінюється випадковим чином,
#щоб модель не запам'ятовувала конкретні картинки
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    #віддзеркалення зліва направо з імовірністю 0,5
    T.RandomHorizontalFlip(),
    #поворот до 10 градусів і масштаб від 0,9 до 1,1
    T.RandomAffine(degrees=10, scale=(0.9, 1.1)),
    #яскравість у межах ±10 %
    T.ColorJitter(brightness=0.1),
    #картинка -> тензор з числами від 0 до 1
    T.ToTensor(),
    _norm,
])
#для валідації, тесту й застосунку — без випадкових змін, щоб результат був однаковим
eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    _norm,
])


#обходить усі підпапки набору даних і складає словник «назва файлу -> повний шлях»;
#потрібен, бо в NIH знімки розкладено по 12 папках images_001 ... images_012
def build_image_index(base_dir):
    index = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.png'):
                index[f] = os.path.join(root, f)
    return index


#набір даних у форматі PyTorch: за номером повертає знімок і його 14 міток;
#DataLoader бере з нього знімки пакетами
class NIHDataset(Dataset):
    #dataframe — таблиця зі знімками й мітками, image_index — шляхи до файлів,
    #transform — train_transform або eval_transform
    def __init__(self, dataframe, image_index, transform):
        #reset_index робить номери рядків 0, 1, 2 ... без пропусків
        self.df = dataframe.reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform

    #кількість знімків у наборі
    def __len__(self):
        return len(self.df)

    #i-й знімок: відкрити файл, підготувати, взяти мітки з 14 стовпців таблиці
    def __getitem__(self, i):
        row = self.df.iloc[i]
        #знімки сірі, але модель чекає три канали, тому RGB
        img = Image.open(self.image_index[row['Image Index']]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(row[CLASS_NAMES].values.astype('float32'))
        return img, label
