import itertools
import copy
import pandas as pd

epochs = 350

def product_dict(dict_of_lists):
    keys = dict_of_lists.keys()
    vals_it = itertools.product(*ps_sgd.values())
    d = [dict(zip(keys, val_tup)) for val_tup in vals_it]
    return d


core_params = dict(
    output='/n/holyscratch01/pehlevan_lab/Lab/matthew/output_neural_collapse',
    # output='/n/holyscratch01/pehlevan_lab/Lab/matthew/output_neural_collapse_backup',
    checkpoint_hist=0,
    dataset_download=True, cooldown_epochs=0,
    smoothing=0, sched='multistep', decay_rate=0.1,
    epochs=epochs, decay_epochs=[epochs//3, epochs*2//3],
    batch_size=128, weight_decay=0, momentum=0,
    interpolation='', train_interpolation='',
    checkpoint_every=10, checkpoint_first=10, resume=False,
)

ps_mnist = dict(core_params, data_dir='data', dataset='torch/mnist', num_classes=10,
                input_size=(1,28,28), mean=(0.1307,),
                std=(0.3081,))

ps_cifar10 = dict(core_params, dataset='torch/cifar10', num_classes=10,
                  img_size=32)

ps_imagenet = dict(core_params,
    data_dir='/n/pehlevan_lab/Everyone/imagenet/ILSVRC/Data/CLS-LOC',
    dataset='imagenet', img_size=32)

ps_sgd = dict(
    opt=('momentum',),
    lr=(0.0184,),
    mse=(False, True),
    momentum=(0, .4, .9),
    decay=(0, 1e-4, 5e-4, 1e-3),
)
# ps_sgd_list = list(itertools.product(*ps_sgd.values()))
ps_sgd_list = product_dict(ps_sgd)

ps_resnet18_mnist = dict(ps_mnist, model='resnet18')
ps_resnet18_mnist_sgd = [
    dict(ps_resnet18_mnist, **d) for d in ps_sgd_list
]
ps_resnet18_mnist_rmsprop = dict(ps_resnet18_mnist, opt='rmsprop')

ps_resnet18_cifar10 = dict(ps_cifar10, model='resnet18')

ps_resnet18_cifar10_sgd = [
    dict(ps_resnet18_cifar10, **d) for d in ps_sgd_list
]
ps_resnet18_cifar10_rmsprop = dict(ps_resnet18_cifar10, opt='rmsprop')

ps_resnet18_imagenet = dict(ps_imagenet, model='resnet18')
ps_resnet18_imagenet_sgd = [
    dict(ps_resnet18_imagenet, **d) for d in ps_sgd_list
]
ps_resnet18_imagenet_rmsprop = dict(ps_resnet18_imagenet,
                                    opt='rmsprop')

