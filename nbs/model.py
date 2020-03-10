import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models

device = torch.device('cuda')

def conv3x3(ni, no, ks=3, s=2, p=1):
    return nn.Conv2d(ni, no, kernel_size=ks, stride=s, padding=p)

def conv1x1(ni, no, ks=1, s=1, p=0):
    return nn.Conv2d(ni, no, kernel_size=ks, stride=s, padding=p)

class ConvBlock(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.conv = conv3x3(ni, no)
        self.bn = nn.BatchNorm2d(no)

    def forward(self, xb):
        return self.bn(F.relu_(self.conv(xb)))

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, xb):
        return self.func(xb)


class SentinelNet(nn.Module):
    def __init__(self, layers, nc):
        super().__init__()
        self.layers  = nn.ModuleList([ConvBlock(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.flatten = Lambda(lambda x: x.view(x.size(0), -1))
        self.linear  = nn.Linear(layers[-1], nc)

    def forward(self, xb):
        for layer in self.layers:
            xb = layer(xb)
        xb = F.adaptive_avg_pool2d(xb, 1)
        xb = self.flatten(xb)
        xb = self.linear(xb)

        return xb


class Flatten(nn.Module):
    def forward(self, xb):
        return xb.view(xb.size(0), -1)


class SentinelResNet(nn.Module):
    def __init__(self, M, c):
        super().__init__()
        self.features   = nn.Sequential(*list(M.children())[:-2])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(512, c)
        )

    def forward(self, xb):
        xb = self.features  (xb)
        xb = self.classifier(xb)

        return xb

class Model:
    def __init__(self, M):
        self.model = M

    def __call__(self, xb):
        return self.model(xb)

    @staticmethod
    def freeze(L):
        for p in L.parameters(): p.requires_grad_(False)

    @staticmethod
    def unfreeze(L):
        for p in L.parameters(): p.requires_grad_(True)

    def freeze_features(self, arg=True):
        Model.freeze(self.model.features) if arg else Model.unfreeze(self.model.features)

    def freeze_classifier(self, arg=True):
        Model.freeze(self.model.classifier) if arg else Model.unfreeze(self.model.classifier)

    def partial_freeze_features(self, pct=0.2):
        sz = len(list(self.model.features.children()))
        point = int(sz * pct)

        for idx, child in enumerate(self.model.features.children()):
            Model.freeze(child) if idx <= point else Model.unfreeze(child)

    def summary(self):
        print('\n\n')
        for idx, (name, child) in enumerate(self.model.features.named_children()):
            print(f'{idx}: {name}-{child}')
            for param in child.parameters():
                print(f'{param.requires_grad}')

        for idx, (name, child) in enumerate(self.model.classifier.named_children()):
            print(f'{idx}: {name}-{child}')
            for param in child.parameters():
                print(f'{param.requires_grad}')
        print('\n\n')


def get_model():
    model = SentinelNet([3, 16, 32, 64, 128], 5).to(device)
    opt = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    return model, opt


def get_resnet_model():
    resnet18 = models.resnet18(pretrained=True)
    model = Model(SentinelResNet(resnet18, c=5))

    return model
