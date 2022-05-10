import time
import math
import sys
import argparse
import itertools
import numpy as np
import torch
import joblib
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt
import train
import neural_collapse_utils as utils
import model_loader_utils as load_utils
import neural_collapse_exps as exp
import joblib
from torch.multiprocessing import Process, Lock

# %% 

parser_outer = argparse.ArgumentParser(description='outer',
                                       add_help=False)
parser_outer.add_argument('--run-num', type=int, default=None, metavar='N',
                    help='Run number.')
args_outer, remaining_outer = parser_outer.parse_known_args()
run_num = args_outer.run_num
# run_num=1
# breakpoint()

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

memory = joblib.Memory(location='.neural_collapse_cache',
                       verbose=40)
# memory.clear()

n_batches = 2
# n_jobs = 12
n_jobs = 4
run_num = 3
# parser = argparse.ArgumentParser()
# parser.add_argument('n_jobs', type=int)
# parser.add_argument('run_num', type=int)
# args = parser.parse_args()
# n_jobs = args.n_jobs
# run_num = args.run_num
outdir = exp.core_params['output']


@memory.cache
def get_compressions_cached(param_dict, epoch, n_batches=n_batches):
    return get_compressions(param_dict=param_dict, epoch=epoch,
                            n_batches=n_batches)

def get_compressions(param_dict, epoch, n_batches=n_batches):
        out = train.train(param_dict)
        model, loader_train, loader_val, run_dir, pd_mom = out
        load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch)
        compression_train = utils.get_compression(model, loader_train, run_dir,
                                                  n_batches)
        compression_val = utils.get_compression(model, loader_val, run_dir,
                                                n_batches)
        return compression_train, compression_val

@memory.cache
def get_compressions_over_training_cached(param_dict, epochs_idx=None,
                                          n_batches=n_batches):
    return get_compressions_over_training(param_dict, epochs_idx=None,
                                          n_batches=n_batches)

def get_compressions_over_training(param_dict,
                                   epochs_idx=None,
                                   n_batches=n_batches,
                                  ):
    out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = out
    df = pd.DataFrame()
    epochs = np.array(load_utils.get_epochs(run_dir))
    if epochs_idx is not None:
        epochs = epochs[epochs_idx]
    for k1, epoch in enumerate(epochs):
        compression_train, compression_val = get_compressions_cached(
            param_dict, epoch, n_batches)
        # compression_train, compression_val = get_compressions(
            # param_dict, epoch, n_batches)
        d = {'epoch': epoch, 'compression': compression_train, 'mode': 'train'}
        d = {**d, **pd_mom}
        df = pd.concat((df, pd.DataFrame(d, index=[0])), ignore_index=True)
        d = {'epoch': epoch, 'compression': compression_val, 'mode': 'val'}
        d = {**d, **pd_mom}
        df = pd.concat((df, pd.DataFrame(d, index=[0])), ignore_index=True)
    df['mode'] = df['mode'].astype("category")
    df['epoch'] = df['epoch'].astype(int)
    return df


@memory.cache
def get_compressions_over_training_batch(param_dict_list, epochs_idx=None,
                                         n_batches=n_batches,):
    dfs = []
    for param_dict in param_dict_list:
        dfs += [get_compressions_over_training(param_dict, epochs_idx,
                                               n_batches)]
        
    return pd.concat(dfs, ignore_index=True)

# def plots_df(df, x, y, hue_key, new_fig_keys):
    # new_fig_vals = [pd.unique(df[key]) for key in new_fig_keys]
    # new_fig_val_combss = itertools.product(*new_fig_vals)
    # for fig_val_comb in new_fig_val_combs:
        # df_comb = df
        # key_str = ''
        # for k1, key in enumerate(new_fig_keys):
            # val = fig_val_comb[k1]
            # df_comb = df_comb[df_comb[key] == val]
            # key_str += '_' + key + str(val)
        # df_plot = df_comb

def plots_df(data, x, y, hue, style, row=None, col=None,
             height=height_default, figname='fig.pdf'):
    fg = sbn.relplot(data=data, x=x, y=y, hue=hue, style=style, row=row,
                     col=col, kind='line', height=height)
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

if __name__ == '__main__':
    fn = train.train
    fn_par = joblib.delayed(fn)
    ps_set1 = exp.ps_resnet18_mnist_sgd + exp.ps_resnet18_mnist_rmsprop
    ps_set2 = exp.ps_resnet18_cifar10_rmsprop 
    ps_set3 = exp.ps_resnet18_cifar10_sgd
    ps_all = ps_set1 + ps_set2 + ps_set3
    print(len(ps_all))
    sys.exit()
    # ps_chunks = list(chunks(ps_all, len(ps_all)//n_jobs))
    # ps_chunk = ps_chunks[run_num-1]
    fn(ps_all[run_num-1])
    # df = get_compressions_over_training_batch(ps_all, epochs_idx=[0, 5, 10, 20 -1])
    # plot_keys = ['dataset', 'epoch', 'compression', 'mode', 'momentum', 'mse_loss', 'opt',
                 # 'weight_decay']
    # dfn = df[plot_keys]
    # filt = (dfn['dataset']=='torch/mnist') & (dfn['opt']=='momentum')
    # df1 = dfn[filt]
    # plots_df(df1, 'epoch', 'compression', 'weight_decay', 'mse_loss', row='mode',
             # col='momentum', figname='mnist_sgd.png')
    # filt = (dfn['dataset']=='torch/mnist') & (dfn['opt']=='rmsprop')
    # df1 = dfn[filt]
    # plots_df(df1, 'epoch', 'compression', 'weight_decay', style='mse_loss', row='mode',
             # figname='mnist_rmsprop.png')

    # filt = (dfn['dataset']=='torch/cifar10') & (dfn['opt']=='momentum')
    # df1 = dfn[filt]
    # plots_df(df1, 'epoch', 'compression', 'weight_decay', 'mse_loss', row='mode',
             # col='momentum', figname='cifar10_sgd.png')
    # filt = (dfn['dataset']=='torch/cifar10') & (dfn['opt']=='rmsprop')
    # df1 = dfn[filt]
    # plots_df(df1, 'epoch', 'compression', 'weight_decay', style='mse_loss', row='mode',
             # figname='cifar10_rmsprop.png')

