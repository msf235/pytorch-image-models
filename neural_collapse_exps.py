import itertools
import copy
import pandas as pd
from pathlib import Path

epochs = 350

def product_dict(dict_of_lists):
    keys = dict_of_lists.keys()
    vals_it = itertools.product(*dict_of_lists.values())
    d = [dict(zip(keys, val_tup)) for val_tup in vals_it]
    return d

if Path('/n').exists():
    out = '/n/holyscratch01/pehlevan_lab/Lab/matthew/output_neural_collapse'
else:
    out = 'output_neural_collapse'

core_params = dict(
    output=out,
    # output='/n/holyscratch01/pehlevan_lab/Lab/matthew/output_neural_collapse_backup',
    # output='output_neural_collapse',
    checkpoint_hist=1,
    dataset_download=True, cooldown_epochs=0,
    smoothing=0, sched='multistep', decay_rate=0.1,
    epochs=epochs, decay_epochs=[epochs//3, epochs*2//3],
    batch_size=128, weight_decay=0, momentum=0,
    interpolation='', train_interpolation='',
    checkpoint_every=10, checkpoint_first=10, resume=True,
    workers=1, no_prefetcher=True, device='cpu',
)

ps_mnist = dict(core_params, data_dir='data', dataset='torch/mnist',
                num_classes=10, input_size=(1,28,28), mean=(0.1307,),
                std=(0.3081,))

ps_cifar10 = dict(core_params, data_dir='data', dataset='torch/cifar10',
                  num_classes=10, img_size=32)

ps_imagenet = dict(core_params,
    data_dir='/n/pehlevan_lab/Everyone/imagenet/ILSVRC/Data/CLS-LOC',
    dataset='imagenet')

ps_sgd = dict(
    opt=('momentum',),
    lr=(0.0184,),
    mse_loss=(False, True),
    momentum=(0, .4, .9),
    weight_decay=(0, 5e-4, 1e-3, 1e-2),
)
ps_sgd2 = dict(
    opt=('momentum',),
    lr=(0.0184,),
    mse_loss=(False, True),
    momentum=(0, .4, .9),
    drop=(.2, .4),
)
ps_sgd_list = product_dict(ps_sgd) + product_dict(ps_sgd2)
# ps_sgd_list = product_dict(ps_sgd2)
ps_rmsprop = dict(
    opt=('rmsprop',),
    lr=(0.0184,),
    mse_loss=(False, True),
    weight_decay=(0, 5e-4, 1e-3, 1e-2),
)
ps_rmsprop2 = dict(
    opt=('rmsprop',),
    lr=(0.0184,),
    mse_loss=(False, True),
    drop=(.2, .4),
)
ps_rmsprop_list = product_dict(ps_rmsprop) + product_dict(ps_rmsprop2)
# ps_rmsprop_list = product_dict(ps_rmsprop2)

ps_noisy_sgd = dict(
    opt=('noisy_sgd',),
    lr=(0.0184,),
    mse_loss=(False, True),
    grad_noise=(.1, .4),
)
ps_noisy_sgd_list = product_dict(ps_noisy_sgd)

ps_sgd_rmsprop_comb = dict(
    opt=('sgd_rmsprop_comb',),
    lr=(0.0184,),
    mse_loss=(False, True),
    sgd_rmsprop_prop=(.25, .5, .75),
)
ps_sgd_rmsprop_comb_list = product_dict(ps_sgd_rmsprop_comb)

ps_resnet18_mnist = dict(ps_mnist, model='resnet18')
ps_resnet18_mnist_sgd = [
    dict(ps_resnet18_mnist, **d) for d in ps_sgd_list
]
ps_resnet18_mnist_rmsprop = [
    dict(ps_resnet18_mnist, **d) for d in ps_rmsprop_list
]
# %% 
ps_resnet18_cifar10 = dict(ps_cifar10, model='resnet18')

ps_resnet18_cifar10_sgd = [
    dict(ps_resnet18_cifar10, **d) for d in ps_sgd_list
]
ps_resnet18_cifar10_rmsprop = [
    dict(ps_resnet18_cifar10, **d) for d in ps_rmsprop_list
]
ps_resnet18_cifar10_noisy_sgd = [
    dict(ps_resnet18_cifar10, **d) for d in ps_noisy_sgd_list
]
ps_resnet18_cifar10_sgd_rmsprop_comb = [
    dict(ps_resnet18_cifar10, **d) for d in ps_sgd_rmsprop_comb_list
]

ps_resnet152_imagenet = dict(ps_imagenet, model='resnet152')
ps_resnet152_imagenet_sgd = [
    dict(ps_resnet152_imagenet, **d) for d in ps_sgd_list
]
ps_resnet152_imagenet_rmsprop = [
    dict(ps_resnet152_imagenet, **d) for d in ps_rmsprop_list
]

