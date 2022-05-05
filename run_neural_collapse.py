import time
import torch
import joblib
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt
import train
import neural_collapse_utils as utils
import model_loader_utils as load_utils
import neural_collapse_exps as exp

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
outdir = exp.core_params['output']


@memory.cache
def get_compressions_over_training(param_dict_list):
    df = pd.DataFrame()
    for param_dict in param_dict_list:
        out = train.train(param_dict)
        model, loader_train, loader_val, run_dir = out
        epochs = load_utils.get_epochs(run_dir)
        for epoch in epochs:
            tic = time.time()
            load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch)
            compression_train = utils.get_compression(model, loader_train, run_dir,
                                                      n_batches)
            compression_val = utils.get_compression(model, loader_val, run_dir,
                                                    n_batches)
            df.append({'epoch': epoch, 'compression': compression_train,
                       'mode': 'train'}, ignore_index=True)
            df.append({'epoch': epoch, 'compression': compression_val,
                       'mode': 'val'}, ignore_index=True)
            toc = time.time()
            print(epoch, '/', len(epochs), toc-tic)
        df['mode'] = df['mode'].astype(pd.Categorical)
        df['epoch'] = df['epoch'].astype(int)
    return df

# def plots_df(df, x, y, hue_key, new_fig_keys):
    # df = get_compressions_over_training(param_dict)
    # df_train = df.copy().drop(columns='compression_val')
    # df_val = df.copy().drop(columns='compression_train')
    # sbn.lineplot(data=df_train, x='epoch', y='compression_train')
    # fig, ax = plt.subplots(figsize=figsize_default)
    # sbn.lineplot(ax=ax, data=df, x='epoch', y='compression', hue='mode')
    # fig.savefig('test.pdf')


if __name__ == '__main__':
    get_compressions_over_training(exp.ps_resnet18_mnist_sgd)

    # df = get_compressions_over_training(
       # exp.ps_resnet18_mnist_sgd)
    # breakpoint()




# batch_size = loader_train.loader.batch_size
# num_its = data_lim // batch_size
