import time
import argparse
import itertools
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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-r', default=0, help='Run number')
parser.add_argument('-j', default=1, help='Total jobs')
args = parser.parse_known_args()[0]
# torch.multiprocessing.set_start_method('spawn')
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

memory = joblib.Memory(location='.neural_collapse_cache')
memory.clear()
n_batches = 2
n_jobs = 4
# n_jobs = 1
outdir = exp.core_params['output']


# @memory.cache
def get_compressions_over_training(param_dict, n_batches=n_batches):
    df = pd.DataFrame()
    out = train.train(param_dict)
    model, loader_train, loader_val, run_dir = out
    epochs = load_utils.get_epochs(run_dir)
    for k1, epoch in enumerate(epochs):
        tic = time.time()
        load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch)
        compression_train = utils.get_compression(model, loader_train, run_dir,
                                                  n_batches)
        compression_val = utils.get_compression(model, loader_val, run_dir,
                                                n_batches)
        df.concat({'epoch': epoch, 'compression': compression_train,
                   'mode': 'train'}, axis=0, join='outer', ignore_index=True)
        df.condat({'epoch': epoch, 'compression': compression_val,
                   'mode': 'val'}, axis=0, join='outer', ignore_index=True)
        toc = time.time()
        print(k1, '/', len(epochs), toc-tic)
    df['mode'] = df['mode'].astype(pd.Categorical)
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
    fn = train.train
    fn_par = joblib.delayed(fn)
    # fn(exp.ps_resnet18_mnist_rmsprop[0])
    ps_set1 = exp.ps_resnet18_mnist_rmsprop + exp.ps_resnet18_mnist_sgd
    ps_set2 = exp.ps_resnet18_cifar10_rmsprop 
    ps_set3 = exp.ps_resnet18_cifar10_sgd
    ps_all = ps_set1 + ps_set2 + ps_set3
    ps_chunks = chunks(ps_all, len(ps_all)//args.j)
    # fn(ps_chunks[args.r])
    


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
