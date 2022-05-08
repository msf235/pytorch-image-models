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

sbn.set_palette('colorblind')
plt.rcParams['font.size'] = 6
plt.rcParams['font.size'] = 6
plt.rcParams['lines.markersize'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.titlesize'] = 8
figsize_default = (3,2.5)

memory = joblib.Memory(location='.neural_collapse_cache',
                       verbose=40)
memory.clear()

n_batches = 2
n_jobs = 12
run_num = 3
# parser = argparse.ArgumentParser()
# parser.add_argument('n_jobs', type=int)
# parser.add_argument('run_num', type=int)
# args = parser.parse_args()
# n_jobs = args.n_jobs
# run_num = args.run_num
outdir = exp.core_params['output']


@memory.cache
def get_compressions_over_training_cached(param_dict, n_batches=n_batches,
                                          epochs_idx=None):
    return get_compressions_over_training(param_dict=param_dict,
                                          n_batches=n_batches,
                                          epochs_idx=epochs_idx)

def get_compressions_over_training(param_dict, n_batches=n_batches,
                                   epochs_idx=None):
    df = pd.DataFrame()
    out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = out
    epochs = np.array(load_utils.get_epochs(run_dir))
    if epochs_idx is not None:
        epochs = epochs[epochs_idx]
    for k1, epoch in enumerate(epochs):
        tic = time.time()
        load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch)
        compression_train = utils.get_compression(model, loader_train, run_dir,
                                                  n_batches)
        compression_val = utils.get_compression(model, loader_val, run_dir,
                                                n_batches)
        d = {'epoch': epoch, 'compression': compression_train, 'mode': 'train'}
        d = {**d, **pd_mom}
        df = pd.concat((df, pd.DataFrame(d, index=[0])), ignore_index=True)
        d = {'epoch': epoch, 'compression': compression_val, 'mode': 'val'}
        d = {**d, **pd_mom}
        df = pd.concat((df, pd.DataFrame(d, index=[0])), ignore_index=True)
        toc = time.time()
        print(k1+1, '/', len(epochs), toc-tic)
    df['mode'] = df['mode'].astype("category")
    df['epoch'] = df['epoch'].astype(int)
    return df


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

def plots_df(data, x, y, hue_key, row=None, col=None):
    fg = sbn.relplot(data=data, x=x, y=y, hue_key=hue_key, row=row, col=col,
                kind='line')
    fg.savefig('test.pdf')
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
    # get_compressions_over_training(exp.ps_resnet18_mnist_sgd)
    # fn = get_compressions_over_training
    # sys.exit()
    fn = train.train
    fn_par = joblib.delayed(fn)
    # fn(exp.ps_resnet18_mnist_rmsprop[0])
    ps_set1 = exp.ps_resnet18_mnist_rmsprop + exp.ps_resnet18_mnist_sgd
    ps_set2 = exp.ps_resnet18_cifar10_rmsprop 
    ps_set3 = exp.ps_resnet18_cifar10_sgd
    # ps_set1 = exp.ps_resnet18_mnist_sgd
    # ps_set2 = exp.ps_resnet18_cifar10_sgd
    ps_all = ps_set1 + ps_set2 + ps_set3
    ps_chunks = list(chunks(ps_all, len(ps_all)//n_jobs))
    # for run_num in range(1, 13):
        # print('================================')
        # print('================================')
        # print(run_num)
        # print('================================')
        # print('================================')
    ps_chunk = ps_chunks[run_num-1]
    print(f"Running chunk {run_num} / {len(ps_chunks)}")
    print(f"This chunk has size {len(ps_chunk)}")
    [fn(ps_chunk[k1]) for k1 in range(len(ps_chunk))]
    # for ps in ps_all:
        # fn(ps)
    # fn(ps_all[run_num-1])
    # df = pd.DataFrame()
    # for ps in ps_all:
        # df_new = get_compressions_over_training_cached(ps)
        # df = pd.concat((df, df_new), ignore_index=True)
    # for ps in ps_all:
        # df_new = get_compressions_over_training_cached(ps,
                                                       # epochs_idx=[0, -1])
        # df = pd.concat((df, df_new), ignore_index=True)
    # df1 = df[df['data_set']=='mnist'][df['opt']=='sgd'][df['weight_decay']==0]
    # plot_keys = ['epoch', 'compression', 'mode', 'momentum', ]
    # df =
    # breakpoint()
    

    # processes = []
    # processes += [Process(target=fn, args=(ps,)) for ps in ps_set1]
    # + exp.ps_resnet18_cifar10_sgd + \
                # exp.ps_resnet18_cifar10_rmsprop
    # if n_jobs == 1:
        # for proc in processes:
            # proc.run()
    # else:
        # chunked_processes = list(chunks(processes, n_jobs))
        # for i0, process_chunk in enumerate(chunked_processes):
            # print(f"Starting batch {i0+1} of {len(chunked_processes)} batches")
            # for process in process_chunk:
                # # time.sleep(.5)
                # process.start()
            # [process.join() for process in process_chunk]
    # ps_all = ps_set2
    # if n_jobs > 1:
        # joblib.Parallel(n_jobs=n_jobs, backend='loky')(fn_par(ps) for ps in ps_all)
    # else:
        # dfs = []
        # for ps in ps_alll:
            # fn(ps)
    # dfs = [fn(ps) for ps in exp.ps_resnet18_mnist_sgd]
    # df = pd.concat(dfs)
    # df1 = df[df['mse']==True].drop('mse')
    # df2 = df[df['mse']==False].drop('mse')
    # plots_df(df1, x='epoch', y='compression', hue_key='mode', row='momentum',
             # col='decay')
    # plots_df(df2, x='epoch', y='compression', hue_key='mode')
    # dfs = joblib.Parallel(n_jobs=2)(fn_par(ps) for ps in exp.ps_resnet18_mnist_rmsprop)
    # dfs = joblib.Parallel(n_jobs=2)(fn_par(ps) for ps in
                                    # exp.ps_resnet18_cifar10_rmsprop)
    # [fn(ps) for ps in ps_all]
    # dfs = joblib.Parallel(n_jobs=4)(fn_par(ps) for ps in ps_all)
    # get_compressions_over_training(exp.ps_resnet18_cifar_sgd)

    # df = get_compressions_over_training(
       # exp.ps_resnet18_mnist_sgd)
    # breakpoint()




# batch_size = loader_train.loader.batch_size
# num_its = data_lim // batch_size
