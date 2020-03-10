import torch
from torchvision import transforms

import pdb
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

from srm_single_model import get_model
from srm_single_dataset import get_data

from visualize import split, merge

device = torch.device('cuda')
metrics = ['$', 'H20', 'TOI', '^', 'COK', 'DRT', 'POP', 'LS', 'AG']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
CP = 'single_d121_ur_phase3_bal_4_final.pt'
IMG_DIR = Path('./test_images')

COLOR = {
    'GREEN': (52,168,82),
    'LIGHT_GREEN': (12,234,8),
    'YELLOW': (236,251,7),
    'ORANGE': (251,188,7),
    'RED': (234,67,53)
}

w2s = {
    'a:poorest': COLOR['RED'] + (128,),
    'b:poorer': COLOR['ORANGE'] + (128,),
    'c:middle': COLOR['YELLOW'] + (128,),
    'd:richer': COLOR['LIGHT_GREEN'] + (128,),
    'e:richest': COLOR['GREEN'] + (128,)
}

p2s = {
    'v.high': COLOR['RED'] + (128,),
    'high': COLOR['ORANGE'] + (128,),
    'avg': COLOR['YELLOW'] + (128,),
    'low': COLOR['LIGHT_GREEN'] + (128,),
    'v.low': COLOR['GREEN'] + (128,),
}

d2s = {
    'high': COLOR['RED'] + (128,),
    'mid': COLOR['YELLOW'] + (128,),
    'low': COLOR['GREEN'] + (128,),
}

t2s = {
    'flush to septic tank': COLOR['GREEN'] + (128,),
    'flush to pit latrine': COLOR['GREEN'] + (128,),
    'flush to piped sewer system': COLOR['GREEN'] + (128,),
    'flush to somewhere else': COLOR['GREEN'] + (128,),
    "flush, don't know where": COLOR['GREEN'] + (128,),
    "flush": COLOR['GREEN'] + (128,),
    'dry toilet': COLOR['YELLOW'] + (128,),
    'composting toilet': COLOR['YELLOW'] + (128,),
    "other": COLOR['YELLOW'] + (128,),
    "don't know where": COLOR['YELLOW'] + (128,),
    'pit latrine with slab': COLOR['ORANGE'] + (128,),
    'pit latrine without slab/open pit': COLOR['ORANGE'] + (128,),
    'ventilated improved pit latrine (vip)': COLOR['ORANGE'] + (128,),
    "no facility/bush/field": COLOR['RED'] + (128,),
}

def load_model():
    wrapper, db = get_model(), get_data()
    model = wrapper.model
    mapper = db.targets
    model.load_state_dict(torch.load(CP))
    model.eval()

    return model, mapper

model, mapper = load_model()
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE + 32),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def process_one(img):
    img_tfm = transform(img); img_tfm.unsqueeze_(0)
    out = model(img_tfm)

    t = dict()
    for i,k in enumerate(mapper):
        index = out[i].argmax()
        t[k] = mapper[k]['classes'][index]
    return t

def single_predict(filename, ntiles, metric, mapper, **kwargs):
    img = Image.open(filename)
    d = split(img, ntiles)
    for k,v in d.items():
        v['score'] = process_one(v['img'])

    new_img = merge(d, ntiles, metric, mapper)
    # new_img.save('./test_images/bangalore-64km_toilet_type.png')

    return (img, new_img)

def main():
    place = 'Mumbai'
    filename = IMG_DIR/f'{place}.png'
    _, wealth_img = single_predict(filename, 4, 'wealth', w2s)
    _, toilet_type_img = single_predict(filename, 4, 'toilet_type', t2s)
    _, pop_density_img = single_predict(filename, 4, 'pop_density', p2s)
    _, drought_img = single_predict(filename, 4, 'drought', d2s)

    wealth_img.save('./test_images/Mumbai_1-wealth.png')
    toilet_type_img.save('./test_images/Mumbai_1-toilet_type.png')
    pop_density_img.save('./test_images/Mumbai_1-pop_density.png')
    drought_img.save('./test_images/Mumbai_1-drought.png')


if __name__ == "__main__":
    main()
