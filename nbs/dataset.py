import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from utils import read_df

CWD           = Path.cwd()
DATA_PATH     = Path(CWD/'data'/'qol')
SENTINEL_MEAN = [0.3582, 0.3714, 0.3994]
SENTINEL_STD  = [0.1744, 0.1435, 0.1455]
IMAGE_SIZE = 224
BATCH_SIZE = 128

def match_file2image(imdir, metafile):
    ims = set(int(p.stem) for p in imdir.glob('*.png') if not p.stem.endswith('mask'))
    metainfo = read_df(metafile)
    missing = [row['cluster'] for (_,row) in metainfo.iterrows()
                              if row['cluster'] not in ims]

    metainfo[~metainfo.cluster.isin(missing)].to_csv(metafile, index=False)


class SentinelDataset(Dataset):
    def __init__(self, imdir, metafile, imid, target, train=True, transform=None, filter=None):
        super().__init__()
        self.imdir = imdir
        self.metafile = metafile
        self.imid = imid
        self.target = target
        self.train = train
        self.transform = transform
        self.filter = filter

        metainfo = read_df(metafile)[[imid, target]]
        if self.filter:
            metainfo = metainfo[metainfo[target] == filter]

        self.classes = sorted(metainfo[target].unique())
        self.o2i = {o:i for i,o in enumerate(self.classes)}
        self.data = self.split(metainfo, 0.2, 42)

    def split(self, data, test_size, random_state):
        train, valid = train_test_split(data, test_size=test_size, random_state=random_state)
        return train if self.train else valid

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cluster, label = row[self.imid], self.o2i[row[self.target]]
        im = Image.open(self.imdir/f'{cluster}.png').convert('RGB')

        return self.transform(im), label

transform = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=SENTINEL_MEAN, std=SENTINEL_STD)
    ]),
    'valid': transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=SENTINEL_MEAN, std=SENTINEL_STD)
    ])
}

def get_data():
    train_ds = SentinelDataset(DATA_PATH/'sentinel', DATA_PATH/'sentinel.csv', 'cluster', 'wealth', True,  transform['train'], None)
    valid_ds = SentinelDataset(DATA_PATH/'sentinel', DATA_PATH/'sentinel.csv', 'cluster', 'wealth', False, transform['valid'], None)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE*2, shuffle=False)

    return (train_dl, valid_dl)

if __name__ == '__main__':
    # match_file2image(DATA_PATH/'sentinel', DATA_PATH/'sentinel.csv')
    pass
