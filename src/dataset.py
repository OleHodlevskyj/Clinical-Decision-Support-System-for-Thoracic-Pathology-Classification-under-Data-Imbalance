import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

from config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

_norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

#аугментація
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    _norm,
])
eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    _norm,
])


def build_image_index(base_dir):
    index = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.png'):
                index[f] = os.path.join(root, f)
    return index


class NIHDataset(Dataset):
    def __init__(self, dataframe, image_index, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.image_index[row['Image Index']]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(row[CLASS_NAMES].values.astype('float32'))
        return img, label
