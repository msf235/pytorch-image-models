import timm
from timm import data
from timm import models
from matplotlib import pyplot as plt

vgg11 = timm.create_model('vgg11', in_chans=1)
mnist = data.create_dataset('torch/mnist', 'mnist', download=True,
                            split='val')



