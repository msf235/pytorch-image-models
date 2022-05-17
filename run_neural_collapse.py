import time
import math
import sys
from pathlib import Path
import argparse
import itertools
import numpy as np
import torch
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt
import train
import neural_collapse_utils as utils
import model_loader_utils as load_utils
import neural_collapse_exps as exp
import torchvision.models.feature_extraction as fe
import model_output_manager_hash as mom

sbn.set_palette("colorblind")

# %% 

parser_outer = argparse.ArgumentParser(description='outer',
                                       add_help=False)
parser_outer.add_argument('--run-num', type=int, default=None, metavar='N',
                    help='Run number.')
args_outer, remaining_outer = parser_outer.parse_known_args()
run_num = args_outer.run_num

mem_cache = Path('.neural_collapse_small_filter_cache')
# mem_cache = Path('.neural_collapse_cache_test')
# mem_cache = Path('.neural_collapse_cache_old')
memory = mom.Memory(location=mem_cache)
# memory.clear()


# %% 

sbn.set_palette('colorblind')
plt.rcParams['font.size'] = 6
plt.rcParams['font.size'] = 6
plt.rcParams['lines.markersize'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.titlesize'] = 8
# figsize_default = (3,2.5)
height_default = 2


# n_batches = 2
n_batches = 1
# n_jobs = 12
n_jobs = 4
# run_num = 3
# parser = argparse.ArgumentParser()
# parser.add_argument('n_jobs', type=int)
# parser.add_argument('run_num', type=int)
# args = parser.parse_args()
# n_jobs = args.n_jobs
# run_num = args.run_num
outdir = exp.core_params['output']


# @memory.cache(ignore=['train_out'])
# def get_compressions_cached(param_dict, epoch, layer_ids, n_batches=n_batches,
                            # train_out=None):
    # return get_compressions(param_dict=param_dict, epoch=epoch,
                            # layer_ids=layer_ids, n_batches=n_batches,
                            # train_out=train_out)
@memory.cache(ignore=['train_out', 'device', 'param_dict.output', 'param_dict.workers',
                      'param_dict.resume', 'param_dict.dataset_download',
                      'param_dict.device', 'param_dict.no_prefetcher'])
def get_dists_projected(param_dict, epoch, layer_ids, n_batches=n_batches,
                        n_samples=None, lin_class_its=50, mode='val',
                        train_out=None, device='cpu'):
    if mode != 'val' or n_samples is not None:
        raise AttributeError('Not implemented yet')
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    nodes, __ = fe.get_graph_node_names(model)
    load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch,
                                             device='cpu')
    # model.eval().cpu()
    model.eval()
    feature_dict = {}
    node_range = list(range(len(nodes)))
    node_range_dict = {key: val for key, val in zip(nodes, node_range)}
    if isinstance(layer_ids, slice):
        layer_ids = node_range[layer_ids]
    layer_id_conv = []
    nodes_filt = []
    for layer_id in layer_ids:
        if isinstance(layer_id, str):
            if layer_id not in nodes:
                raise ValueError("layer_id not valid.")
            feature_dict[layer_id] = layer_id
            layer_id_conv.append(node_range_dict[layer_id])
            nodes_filt.append(layer_id)
        elif isinstance(layer_id, int):
            layer_key = nodes[layer_id]
            feature_dict[layer_key] = layer_key
            layer_id_conv.append(node_range[layer_id])
            nodes_filt.append(nodes[layer_id])
    feat_extractor = fe.create_feature_extractor(model,
                                                 return_nodes=feature_dict)
    # compression_train = utils.get_compressions_projected(
        # feat_extractor, loader_train, run_dir, n_batches)
    dists = utils.get_dists_projected(
        feat_extractor, loader_val, run_dir, n_batches,
        lin_class_its, device)
    # (d_within_avgs, d_across_avgs, d_within_aligned_avgs,
                 # d_across_aligned_avgs, d_within_aligned_ratio_avgs,
                 # d_across_aligned_ratio_avgs) = out
    return dists, layer_id_conv, nodes_filt


# @memory.cache(ignore=['train_out'])
# def get_compressions(param_dict, epoch, layer_ids, n_batches=n_batches,
                     # train_out=None):
        # model, loader_train, loader_val, run_dir, pd_mom = train_out
        # model.eval()
        # nodes, __ = fe.get_graph_node_names(model)
        # load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch)
        # model.eval()
        # feature_dict = {}
        # node_range = list(range(len(nodes)))
        # node_range_dict = {key: val for key, val in zip(nodes, node_range)}
        # if isinstance(layer_ids, slice):
            # layer_ids = node_range[layer_ids]
        # layer_id_conv = []
        # nodes_filt = []
        # for layer_id in layer_ids:
            # if isinstance(layer_id, str):
                # if layer_id not in nodes:
                    # raise ValueError("layer_id not valid.")
                # feature_dict[layer_id] = layer_id
                # layer_id_conv.append(node_range_dict[layer_id])
                # nodes_filt.append(layer_id)
            # elif isinstance(layer_id, int):
                # layer_key = nodes[layer_id]
                # feature_dict[layer_key] = layer_key
                # layer_id_conv.append(node_range[layer_id])
                # nodes_filt.append(nodes[layer_id])
        # feat_extractor = fe.create_feature_extractor(model, return_nodes=feature_dict)
        # compression_train = utils.get_compressions(feat_extractor, loader_train,
                                                  # run_dir, n_batches)
        # compression_val = utils.get_compressions(feat_extractor, loader_val,
                                                # run_dir, n_batches)
        # return compression_train, compression_val, layer_id_conv, nodes_filt


def get_compressions_over_training(param_dict, epochs_idx=None, layer_id=-1,
                                   n_batches=n_batches, projection=None,
                                   n_samples=None, mode='val',
                                   device='cpu'):
    train_out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    epochs = np.array(load_utils.get_epochs(run_dir))
    if epochs_idx is not None:
        epochs = epochs[epochs_idx]
    ds = []
    for k1, epoch in enumerate(epochs):
        if projection is not None:
            out = get_dists_projected(param_dict, epoch, [-1],
                                      n_batches, n_samples, 100, mode,
                                      train_out, device)
            dists, layer_id_k1, name_k1 = out
            compression = (dists[0] / dists[1]).item()
            d = {'epoch': epoch, 'compression': compression, 
                 'layer_idx': layer_id_k1[0], 'layer_name': name_k1[0],
                 'mode': mode}
            d = {**d, **pd_mom}
            ds.append(d)
        else:
            out = get_compressions(
                param_dict, epoch, [layer_id], n_batches, train_out)
            compression_train, compression_val, layer_id_k1, name_k1 = out
            layer_id_k1 = layer_id_k1[0]
            name_k1 = name_k1[0]
            compression_train = compression_train[0].tolist()
            compression_val = compression_val[0].tolist()
            d = {'epoch': epoch, 'compression': compression_train, 
                 'layer_idx': layer_id_k1, 'layer_name': name_k1,
                 'mode': 'train'}
            d = {**d, **pd_mom}
            ds.append(d)
    df = pd.DataFrame(ds)
    df['mode'] = df['mode'].astype("category")
    df['epoch'] = df['epoch'].astype(int)
    return df


# @memory.cache
def get_compressions_over_training_batch(param_dict_list, epochs_idx=None,
                                         layer_id=-1, n_batches=n_batches,
                                         projection=None, n_samples=None,
                                         mode='val', device='cpu'):
    dfs = []
    for param_dict in param_dict_list:
        dfs += [get_compressions_over_training(param_dict, epochs_idx,
                                               layer_id, n_batches, projection,
                                               n_samples, mode, device)]
        
    return pd.concat(dfs, ignore_index=True)


def get_compressions_over_layers(param_dict, epochs_idx,
                                 layer_ids=slice(None), n_batches=n_batches,
                                 projection=None, n_samples=None, mode='val',
                                 device='cpu'):
    train_out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    df = pd.DataFrame()
    epochs = np.array(load_utils.get_epochs(run_dir))
    if epochs_idx is not None:
        epochs = epochs[epochs_idx]
    ds = []
    for k1, epoch in enumerate(epochs):
        if projection is not None:
            out = get_dists_projected(param_dict, epoch, layer_ids,
                                      n_batches, n_samples, 100, mode,
                                      train_out, device)
        else:
            out = get_compressions(param_dict, epoch, layer_ids, n_batches,
                                          train_out)
        dists, layer_ids_k1, layer_names_k1 = out
        # compression_train, compression_val, layer_ids_k1, layer_names_k1 = out
        compression = dists[0] / dists[1]
        compression_aligned = dists[2] / dists[3]
        compression_aligned_ratio = compression_aligned / compression
        filt = ~torch.isnan(compression)
        layer_names_k1 = [n for n, f in zip(layer_names_k1, filt) if f]
        layer_ids_k1 = [n for n, f in zip(layer_ids_k1, filt) if f]
        compression = compression[filt].tolist()
        compression_aligned = compression_aligned[filt].tolist()
        # compression_train, compression_val, layer_names = get_compressions(
            # param_dict, epoch, layer_ids, n_batches)
        for k1, layer_id in enumerate(layer_ids_k1):
            layer_name = layer_names_k1[k1]
            ct = compression[k1]
            cta = compression_aligned[k1]
            ctar = compression_aligned_ratio[k1]
            d = dict(epoch=epoch, compression=ct,
                     compression_aligned=cta,
                     compression_aligned_ratio=ctar,
                     layer_idx=layer_id, layer_name=layer_name, mode=mode)
            d = {**d, **pd_mom}
            ds.append(d)
    df = pd.DataFrame(ds)
    df['mode'] = df['mode'].astype("category")
    df['epoch'] = df['epoch'].astype(int)
    df['layer_idx'] = df['layer_idx'].astype(int)
    return df

def get_compressions_over_layers_batch(param_dict_list, epochs_idx,
                                 layer_ids=slice(None), n_batches=n_batches,
                                 projection=None, n_samples=None, mode='val',
                                 device='cpu'):
    dfs = []
    for param_dict in param_dict_list:
        dfs.append(get_compressions_over_layers(param_dict, epochs_idx,
                                 layer_ids, n_batches,
                                 projection, n_samples, mode,
                                 device)
                  )
    return pd.concat(dfs, ignore_index=True)


def plots_df(data, x, y, hue=None, style=None, row=None, col=None,
             height=height_default, aspect=1, figname='fig.pdf'):
    fg = sbn.relplot(data=data, x=x, y=y, hue=hue, style=style, row=row,
                     col=col, kind='line', height=height, aspect=aspect)
    for i, ax in enumerate(fg.fig.axes):   ## getting all axes of the fig object
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90) 
    fg.savefig(figname)
    # new_fig_vals = [pd.unique(df[key]) for key in new_fig_keys]



    # sbn.lineplot(data=df_train, x='epoch', y='compression_train')
    # fig, ax = plt.subplots(figsize=figsize_default)
    # sbn.lineplot(ax=ax, data=df, x='epoch', y='compression', hue='mode')
    # fig.savefig('test.pdf')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def plot_over_epochs():
    ps_set1 = exp.ps_resnet18_mnist_sgd + exp.ps_resnet18_mnist_rmsprop
    ps_set2 = exp.ps_resnet18_cifar10_sgd + exp.ps_resnet18_cifar10_rmsprop

    ps_all = ps_set1
    df = get_compressions_over_training_batch(ps_all, [0, 5, 10, 20, -1],
                                            projection='s', device='cpu')
                                      # device='cuda')

    plot_keys = ['dataset', 'epoch', 'compression', 'mode', 'momentum',
                 'mse_loss', 'opt', 'weight_decay', 'drop', 'layer_idx',
                 'layer_name']
    dfn = df[plot_keys]
    filt = (
            (dfn['dataset']=='torch/mnist') &
            (dfn['opt']=='momentum') &
            # (dfn['layer_name']=='global_pool.flatten') &
            # (dfn['layer_name']=='fc') &
            (dfn['weight_decay']==0.0)
           )
    df1 = dfn[filt]
    plots_df(df1, x='epoch', y='compression',
             hue='drop',
             row='mse_loss',
             col='momentum',
             figname='plots/epochs/mnist_sgd_drop.png')

    filt = (
            (dfn['dataset']=='torch/mnist') &
            (dfn['opt']=='momentum') &
            # (dfn['layer_name']=='global_pool.flatten') &
            # (dfn['layer_name']=='fc') &
            (dfn['drop']==0.0)
           )
    df1 = dfn[filt]
    plots_df(df1, x='epoch', y='compression',
             hue='weight_decay',
             row='mse_loss',
             col='momentum',
             figname='plots/epochs/mnist_sgd_weight_decay.png')

    filt = (
            (dfn['dataset']=='torch/mnist') &
            (dfn['opt']=='rmsprop') &
            # (dfn['layer_name']=='global_pool.flatten') &
            # (dfn['layer_name']=='fc') &
            (dfn['weight_decay']==0.0)
           )
    df1 = dfn[filt]
    plots_df(df1, x='epoch', y='compression',
             hue='drop',
             col='mse_loss',
             figname='plots/epochs/mnist_rmsprop_drop.png')

    filt = (
            (dfn['dataset']=='torch/mnist') &
            (dfn['opt']=='rmsprop') &
            # (dfn['layer_name']=='global_pool.flatten') &
            # (dfn['layer_name']=='fc') &
            (dfn['drop']==0.0)
           )
    df1 = dfn[filt]
    plots_df(df1, x='epoch', y='compression',
             hue='weight_decay',
             col='mse_loss',
             figname='plots/epochs/mnist_rmsprop_weight_decay.png')
    breakpoint()

def plot_over_layers(y):
    ps_set1 = exp.ps_resnet18_mnist_sgd + exp.ps_resnet18_mnist_rmsprop
    ps_set2 = exp.ps_resnet18_cifar10_sgd + exp.ps_resnet18_cifar10_rmsprop

    # ps_all = exp.ps_resnet18_mnist_sgd
    ps_all = ps_set1
    # ps_all = exp.ps_resnet18_mnist_rmsprop
    df = get_compressions_over_layers_batch(ps_all, [0, -1], projection='s',
                                      device='cpu', layer_ids=slice(-1))
    dfe = get_compressions_over_layers_batch(ps_all, [0, -1], projection='s',
                                      device='cpu', layer_ids=[-1])
                                      # device='cuda')
    df = pd.concat((df, dfe), ignore_index=True)

    plot_keys = ['dataset', 'epoch', y,
                 'mode', 'momentum', 'mse_loss', 'opt', 'weight_decay', 'drop',
                 'layer_idx', 'layer_name']
    dfn = df[plot_keys]
    dfn = dfn.sort_values('layer_idx', ignore_index=True)
    figdir=Path(f'plots/layers/{y}/')
    figdir.mkdir(parents=True, exist_ok=True)
    # breakpoint()
    dfn['epoch'] = dfn['epoch'].astype('category')
    filt = (
            (dfn['dataset']=='torch/mnist') &
            (dfn['opt']=='momentum') &
            (dfn['weight_decay']==0.0) &
            (dfn['layer_name']!='dropout')
           )
    df1 = dfn[filt]
    plots_df(df1,
             # x='layer_idx',
             x='layer_name',
             y=y,
             hue='drop',
             style='epoch',
             row='mse_loss',
             col='momentum',
             height=2,
             aspect=4,
             figname=figdir/'mnist_sgd_drop.png')

    breakpoint()

    filt = (
            (dfn['dataset']=='torch/mnist') &
            (dfn['opt']=='momentum') &
            (dfn['layer_name']!='dropout') &
            (dfn['drop']==0.0)
           )
    df1 = dfn[filt]
    plots_df(df1,
             # x='layer_idx',
             x='layer_name',
             y=y,
             hue='weight_decay',
             style='epoch',
             row='mse_loss',
             col='momentum',
             height=2,
             aspect=4,
             figname=figdir/'mnist_sgd_weight_decay.png')

    filt = (
            (dfn['dataset']=='torch/mnist') &
            (dfn['opt']=='rmsprop') &
            (dfn['layer_name']!='dropout') &
            (dfn['weight_decay']==0.0)
           )
    df1 = dfn[filt]
    plots_df(df1, x='layer_name', y=y,
             hue='drop',
             style='epoch',
             col='mse_loss',
             height=2,
             aspect=4,
             figname=figdir/'mnist_rmsprop_drop.png')

    filt = (
            (dfn['dataset']=='torch/mnist') &
            (dfn['opt']=='rmsprop') &
            (dfn['layer_name']!='dropout') &
            (dfn['drop']==0.0)
           )
    df1 = dfn[filt]
    plots_df(df1, x='layer_name', y=y,
             hue='weight_decay',
             style='epoch',
             col='mse_loss',
             height=2,
             aspect=4,
             figname=figdir/'mnist_rmsprop_weight_decay.png')

if __name__ == '__main__':
    fn = train.train
    ps_set1 = exp.ps_resnet18_mnist_sgd + exp.ps_resnet18_mnist_rmsprop
    ps_set2 = exp.ps_resnet18_cifar10_sgd + exp.ps_resnet18_cifar10_rmsprop
    # ps_set1 = exp.ps_resnet18_mnist_rmsprop
    # ps_set2 = exp.ps_resnet18_cifar10_rmsprop
    # ps_set1 = exp.ps_resnet18_cifar10_rmsprop
    # ps_set2 = exp.ps_resnet18_cifar10_rmsprop
    # ps_set3 = exp.ps_resnet18_cifar100_sgd + exp.ps_resnet18_cifar100_rmsprop
    # ps_set2 = exp.ps_resnet18_cifar10_sgd
    # ps_set2 = exp.ps_resnet18_cifar10_rmsprop
    ps_all = ps_set1 + ps_set2
    # ps_all = exp.ps_resnet18_mnist_sgd + exp.ps_resnet18_cifar10_sgd
    # ps_all = ps_set1
    # ps_all = ps_set3
    # ps_all = exp.ps_resnet18_mnist_sgd
    # ps_all = ps_set1
    # ps_chunks = list(chunks(ps_all, len(ps_all)//n_jobs))
    # run_num=1
    # print(run_num)
    # ps = ps_all[run_num-1]
    # fn(ps)
    # df = get_compressions_over_layers(ps, [0, -1])
    # df2 = get_compressions_over_training(ps, epochs_idx=[0, 5, 10, 20 -1])
    # for ps in ps_set2:
        # fn(ps)
    # for ps in ps_all:
        # fn(ps)
        # df = get_compressions_over_layers(ps, [0, -1])
        # df2 = get_compressions_over_training(ps, epochs_idx=[0, 5, 10, 20 -1])
    # for ps in ps_set2:
    # for ps in ps_all:
        # df = get_compressions_over_layers(ps, [0, -1], projection='s')
        # get_compressions_over_training(ps, epochs_idx=[0, 5, 10, 20 -1],
                                       # projection='s')
    # breakpoint()
    # print(run_num)
    # run_num=1
    print('run_num:', run_num)
    ps = ps_all[run_num-1]
    fn(ps)
    print("done.")
    # ps = ps_all[0]
    # df = get_compressions_over_layers(ps, [0, -1])
    # df = get_compressions_over_layers(ps, [-1], projection='s',
                                      # device='cpu')
                                      # device='cuda')
    # df = get_compressions_over_layers(ps_epoch0[0], [0], projection='s',
                                      # device='cpu')
    # df = get_compressions_over_layers(ps_epoch0[1], [0], projection='s',
                                      # device='cpu')
    sys.exit()
    # df = get_compressions_over_layers_batch(ps_all, [0, -1], projection='s',
                                      # # device='cpu')
                                      # device='cuda')
    # df = get_compressions_over_layers(ps_all[43], [0, -1], projection='s',
                                      # device='cpu')
                                      # # device='cuda')
    # sys.exit()
    # breakpoint()
    # df = get_compressions_over_training(ps, epochs_idx=[0, 5, 10, 20 -1],
                                        # projection='s',
                                        # # device='cpu')
                                        # device='cuda')
    # df = get_compressions_over_layers(ps, [0, -1])
    # df2 = get_compressions_over_training(ps, epochs_idx=[0, 5, 10, 20 -1])
    # fn(ps_all[run_num-1])
    # df = get_compressions_over_training_batch(ps_all, epochs_idx=[0, 5, 10, 20 -1])
    # df = get_compressions_over_training(ps_set1, epochs_idx=[0, 5, 10, 20 -1])
    # df = get_compressions_over_training_batch(ps_all,
                                              # epochs_idx=[0, 5, 10, 20 -1],
                                              # projection='s')
    # plot_over_epochs()
    plot_over_layers('compression')
    plot_over_layers('compression_aligned')
    plot_over_layers('compression_aligned_ratio')
    sys.exit()
# %% 
# %% 
    breakpoint()
    plot_keys = ['dataset', 'epoch', 'compression', 'mode', 'momentum',
                 'mse_loss', 'opt', 'weight_decay', 'drop', 'layer_idx']
    dfn = df[plot_keys]
    # filt = (dfn['dataset']=='torch/mnist') & (dfn['opt']=='momentum')
    filt = (dfn['dataset']=='torch/mnist') & (dfn['opt']=='momentum') \
            & (dfn['mse_loss']==True) & ((dfn['epoch']==0) | (dfn['epoch']==350))
    df1 = dfn[filt]
    plots_df(df1, x='layer_idx', y='compression', style='weight_decay',
             hue='drop', row='mode', col='momentum',
             figname='plots/layers/mnist_sgd_mse.png')

    breakpoint()
    filt = (dfn['dataset']=='torch/mnist') & (dfn['opt']=='momentum') \
            & (dfn['mse_loss']==False) & (dfn['layer_name']=='global_pool.flatten')

    df1 = dfn[filt]
    plots_df(df1, x='epoch', y='compression', hue='weight_decay', style='drop', row='mode',
             col='momentum', figname='mnist_sgd_cce.png')
    filt = (dfn['dataset']=='torch/mnist') & (dfn['opt']=='rmsprop') \
            & (dfn['mse_loss']==True)
    df1 = dfn[filt]
    plots_df(df1, 'epoch', 'compression', 'weight_decay', style='drop', row='mode',
             figname='mnist_rmsprop_mse.png')
    filt = (dfn['dataset']=='torch/mnist') & (dfn['opt']=='rmsprop') \
            & (dfn['mse_loss']==False)
    df1 = dfn[filt]
    plots_df(df1, 'epoch', 'compression', 'weight_decay', style='drop', row='mode',
             figname='mnist_rmsprop_cce.png')

    
    # filt = (dfn['dataset']=='torch/cifar10') & (dfn['opt']=='momentum')
    # df1 = dfn[filt]
    # plots_df(df1, 'epoch', 'compression', 'weight_decay', 'mse_loss', row='mode',
             # col='momentum', figname='cifar10_sgd.png')
    # filt = (dfn['dataset']=='torch/cifar10') & (dfn['opt']=='rmsprop')
    # df1 = dfn[filt]
    # plots_df(df1, 'epoch', 'compression', 'weight_decay', style='mse_loss', row='mode',
             # figname='cifar10_rmsprop.png')

