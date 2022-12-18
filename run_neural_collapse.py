import time
import math
import sys
from pathlib import Path
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# memory = mom.Memory(location=mem_cache, rerun=True)
memory = mom.Memory(location=mem_cache)
# memory.clear()
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


n_batches_per = 2
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

def evenly_distributed_label_idx(dataset, dataset_indices, num_samples):
    # data_idx = torch.randperm(len(dataset))[:num_samples]
    n = len(set(dataset.targets))
    # num_labels = params['num_classes']

    import numpy as np
    # Y = torch.tensor([x[1] for x in dataset])
    # num_samples_per_label_in_data = torch.sum(Y==0)
    # num_samples_per_label_in_data = torch.sum(
    #     torch.tensor(dataset.targets) == dataset.targets[0]).item()
    # m = int(num_samples_per_label_in_data*num_samples/len(dataset))
    # r = int(len(dataset)/num_samples_per_label_in_data)
    # if m == 0:
    #     data_idx = num_samples_per_label_in_data*np.arange(num_samples)
    # else:
    #     data_idx = []
    #     for k in range(m+1):
    #         data_idx.extend(k+num_samples_per_label_in_data*np.arange(r))
    #     data_idx = data_idx[:num_samples]
    #     return data_idx

    num_samples_per_label_in_data = torch.sum(
        torch.tensor(dataset.targets) == dataset.targets[0]).item()
    m = int(num_samples_per_label_in_data*num_samples/len(dataset_indices))
    r = int(len(dataset_indices)/num_samples_per_label_in_data)
    if m == 0:
        data_idx = num_samples_per_label_in_data*torch.arange(num_samples)
    else:
        data_idx = []
        for k in range(m+1):
            data_idx.extend(k+num_samples_per_label_in_data*torch.arange(r))
        data_idx = data_idx[:num_samples]
    return data_idx

# @memory.cache(ignore=['train_out'])
# def get_compressions_cached(param_dict, epoch, layer_ids, n_batches=n_batches,
                            # train_out=None):
    # return get_compressions(param_dict=param_dict, epoch=epoch,
                            # layer_ids=layer_ids, n_batches=n_batches,
                            # train_out=train_out)
# @memory.cache(ignore=['train_out', 'device', 'param_dict.output', 'param_dict.workers',
                      # 'param_dict.resume', 'param_dict.dataset_download',
                      # 'param_dict.device', 'param_dict.no_prefetcher'])
def get_dists_projected(param_dict, epoch, layer_ids,
                        n_batches_per=n_batches_per, n_batches=None,
                        n_samples=None, lin_class_its=50, mode='val',
                        train_out=None, device='cpu'):
    if mode != 'val' or n_samples is not None:
        raise AttributeError('Not implemented yet')
    model, loader_train, loader_val, run_dir, pd_mom = train_out

    if params['num_classes'] != 'na':
        indices_for_classes_train = torch.where(
            torch.tensor(dataset.targets) < params['num_classes'])[0]
    else:
        indices_for_classes_train = torch.arange(len(dataset.targets))
    # subset_indices = indices_for_classes_train[
    #     torch.randperm(len(indices_for_classes_train))[:NUM_SAMPLES]]
    breakpoint()

    data_idx = evenly_distributed_label_idx(dataset, indices_for_classes_train, NUM_SAMPLES)
    indices_for_classes_train = indices_for_classes_train[data_idx]

    X = torch.stack([dataset[x][0] for x in indices_for_classes_train]).type(torch.float32).to(device)
    Y = torch.tensor([dataset[x][1] for x in indices_for_classes_train]).to(device)

    breakpoint()
    # fe = model
    nodes, __ = fe.get_graph_node_names(model)
    load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch,
                                             device='cpu')
    # model.eval().cpu()
    model.eval()
    feature_dict = {}
    node_range = list(range(len(nodes)))
    # if filt_maxpool:
        # node_range = [node for node in node_range if node != 'maxpool']
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
    model = model.to(device)
    feat_extractor = fe.create_feature_extractor(model,
                                                 return_nodes=feature_dict)
    # compression_train = utils.get_compressions_projected(
        # feat_extractor, loader_train, run_dir, n_batches)
    dists = utils.get_dists_projected(
        feat_extractor, loader_val, run_dir, n_batches_per, n_batches,
        lin_class_its, device)
    # (d_within_avgs, d_across_avgs, d_within_aligned_avgs,
                 # d_across_aligned_avgs, d_within_aligned_ratio_avgs,
                 # d_across_aligned_ratio_avgs) = out
    return dists, layer_id_conv, nodes_filt


def get_feat_ext(epoch, layer_ids, train_out, device):
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    nodes, __ = fe.get_graph_node_names(model)
    load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch,
                                             device=device)
    model.eval()
    model = model.to(device)
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
    return feat_extractor, layer_id_conv, nodes_filt

# @memory.cache(ignore=['train_out'])
def get_compressions(param_dict, epoch, layer_ids, n_batches, n_batches_per,
                     device, train_out=None):
        model, loader_train, loader_val, run_dir, pd_mom = train_out
        model.eval()
        nodes, __ = fe.get_graph_node_names(model)
        load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch)
        model.eval()
        model = model.to(device)
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
        feat_extractor = fe.create_feature_extractor(model, return_nodes=feature_dict)
        compression_train = utils.get_compressions(feat_extractor, loader_train,
                                                  run_dir, n_batches_per,
                                                  n_batches)
        compression_val = utils.get_compressions(feat_extractor, loader_val,
                                                run_dir, n_batches_per,
                                                n_batches)
        return compression_train, compression_val, layer_id_conv, nodes_filt


@memory.cache(ignore=['train_out', 'device', 'param_dict.output', 'param_dict.workers',
                      'param_dict.resume', 'param_dict.dataset_download',
                      'param_dict.device', 'param_dict.no_prefetcher'])
def get_compressions_over_training(param_dict, epochs=None, layer_id=-2,
                                   n_batches_per=n_batches_per, n_batches=None,
                                   projection=None, n_samples=None, mode='val',
                                   device='cpu'):
    train_out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    if n_batches is None:
        n_batches = len(loader_val)
    # if epochs is None:
        # epochs = np.array(load_utils.get_epochs(run_dir))
    epochs_temp = np.array(load_utils.get_epochs(run_dir))
    if epochs is not None:
        epochs = epochs_temp[epochs]
    ds = []
    for k1, epoch in enumerate(epochs):
        if projection is not None:
            if epoch == 0:
                param_dict0 = param_dict.copy()
                for key in exp.opt_params:
                    if key in param_dict0:
                        del param_dict0[key]
                param_dict0['epochs'] = 0
                train_out0 = train.train(param_dict0)
                # model, loader_train, loader_val, run_dir, pd_mom = train_out0
                out = get_dists_projected(param_dict0, epoch, [layer_id],
                                          n_batches_per, n_batches, n_samples, 100, mode,
                                          train_out0, device)
            else:
                out = get_dists_projected(param_dict, epoch, [layer_id],
                                          n_batches_per, n_batches, n_samples, 100, mode,
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
                param_dict, epoch, [layer_id], n_batches, n_batches_per,
                device, train_out)
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



@memory.cache(ignore=['train_out', 'device', 'param_dict.output', 'param_dict.workers',
                      'param_dict.resume', 'param_dict.dataset_download',
                      'param_dict.device', 'param_dict.no_prefetcher'])
def get_compressions_over_layers(param_dict, epochs_idx, layer_ids=slice(None),
                                 n_batches_per=n_batches_per, n_batches=None,
                                 projection=None, n_samples=None, mode='val',
                                 device='cpu'):
    train_out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    if n_batches is None:
        n_batches = len(loader_val)
    df = pd.DataFrame()
    epochs = np.array(load_utils.get_epochs(run_dir))
    if epochs_idx is not None:
        epochs = epochs[epochs_idx]
    ds = []
    for k1, epoch in enumerate(epochs):
        if epoch == 0:
            param_dict0 = param_dict.copy()
            for key in exp.opt_params:
                if key in param_dict0:
                    del param_dict0[key]
            param_dict0['epochs'] = 0
            train_out0 = train.train(param_dict0)
            # model, loader_train, loader_val, run_dir, pd_mom = train_out0
            out = get_dists_projected(param_dict0, epoch, layer_ids,
                                      n_batches_per, n_batches, n_samples, 100,
                                      mode, train_out0, device)
        else:
            out = get_dists_projected(param_dict, epoch, layer_ids,
                                      n_batches_per, n_batches, n_samples, 100,
                                      mode, train_out, device)
        # if projection is not None:
        # else:
            # out = get_compressions(param_dict, epoch, layer_ids, n_batches,
                                          # train_out)
        dists, layer_ids_k1, layer_names_k1 = out
        # compression_train, compression_val, layer_ids_k1, layer_names_k1 = out
        compression = dists[0] / dists[1]
        compression_aligned = dists[2] / dists[3]
        compression_aligned_ratio = compression_aligned / compression
        compression_orth = (dists[0]-dists[2])/(dists[1]-dists[3])
        filt = ~torch.isnan(compression)
        layer_names_k1 = [n for n, f in zip(layer_names_k1, filt) if f]
        layer_ids_k1 = [n for n, f in zip(layer_ids_k1, filt) if f]
        compression = compression[filt].tolist()
        compression_aligned = compression_aligned[filt].tolist()
        # compression_train, compression_val, layer_names = get_compressions(
            # param_dict, epoch, layer_ids, n_batches)
        for k2, layer_id in enumerate(layer_ids_k1):
            layer_name = layer_names_k1[k2]
            ct = compression[k2]
            cta = compression_aligned[k2]
            cto = compression_orth[k2]
            ctar = compression_aligned_ratio[k2]
            d = dict(epoch=epoch, compression=ct,
                     compression_aligned=cta,
                     compression_orth=cto,
                     compression_aligned_ratio=ctar,
                     layer_idx=layer_id, layer_name=layer_name, mode=mode)
            d = {**d, **pd_mom}
            ds.append(d)
    df = pd.DataFrame(ds)
    df['mode'] = df['mode'].astype("category")
    df['epoch'] = df['epoch'].astype(int)
    df['layer_idx'] = df['layer_idx'].astype(int)
    return df


# @memory.cache(ignore=['train_out', 'device', 'param_dict.output', 'param_dict.workers',
                      # 'param_dict.resume', 'param_dict.dataset_download',
                      # 'param_dict.device', 'param_dict.no_prefetcher'])
@memory.cache(ignore=['train_out', 'device', 'param_dict.output', 'param_dict.workers',
                      'param_dict.resume', 'param_dict.dataset_download',
                      'param_dict.device', 'param_dict.no_prefetcher'])
def get_pcs(param_dict, epochs_idx, layer_ids=[-2],
            n_batches=n_batches_per, n_samples=None, mode='val',
            device='cpu'):

    train_out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    df = pd.DataFrame()
    epochs = np.array(load_utils.get_epochs(run_dir))
    if epochs_idx is not None:
        epochs = epochs[epochs_idx]
    ds = []
    pcs_col = []
    labels_col = []
    for k1, epoch in enumerate(epochs):
        if epoch == 0:
            param_dict0 = param_dict.copy()
            for key in exp.opt_params:
                if key in param_dict0:
                    del param_dict0[key]
            param_dict0['epochs'] = 0
            train_out0 = train.train(param_dict0)
            # model, loader_train, loader_val, run_dir, pd_mom = train_out0
            fe, layer_ids, layer_names = get_feat_ext(epoch, layer_ids,
                                                      train_out0, device)
            pcs, labels = utils.get_pcs(fe, loader_val, n_batches, device)
        else:
            fe, layer_ids, layer_names = get_feat_ext(epoch, layer_ids,
                                                      train_out, device)
            pcs, labels = utils.get_pcs(fe, loader_val, n_batches, device)
        pcs_col.append(pcs)
        labels_col.append(labels)
    return pcs_col, labels_col


@memory.cache(ignore=['train_out', 'device', 'param_dict.output', 'param_dict.workers',
                      'param_dict.resume', 'param_dict.dataset_download',
                      'param_dict.device', 'param_dict.no_prefetcher'])
def get_acc_and_loss(param_dict, epoch, mode='val', device='cpu'):
    train_out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    model.eval()
    epochs = np.array(load_utils.get_epochs(run_dir))
    validate_loss_fn = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    def validate_loss_fn(out, target):
        return criterion(out, F.one_hot(target,
                                 num_classes=pd_mom['num_classes']).float())
    if epoch == 0:
        param_dict0 = param_dict.copy()
        for key in exp.opt_params:
            if key in param_dict0:
                del param_dict0[key]
        param_dict0['epochs'] = 0
        train_out0 = train.train(param_dict0)
        model0, loader_train0, loader_val0, run_dir0, pd_mom0 = train_out0
        model.eval()
        acc, loss = utils.get_acc_and_loss(model0, validate_loss_fn,
                                           loader_val, device)
        # sd = load_utils.load_model_from_epoch_and_dir(model0, run_dir0,
                                                      # epoch)
        # model, loader_train, loader_val, run_dir, pd_mom = train_out0
    else:
        # model, loader_train, loader_val, run_dir, pd_mom = train_out
        load_utils.load_model_from_epoch_and_dir(model, run_dir, epoch,
                                                 device=device)
        model.eval()
        acc, loss = utils.get_acc_and_loss(model, validate_loss_fn,
                                           loader_val, device)
    d = {'epoch': epoch, 'accuracy': acc, 'loss': loss,
     'mode': mode}
    d = {**d, **pd_mom}
    df = pd.DataFrame(d, index=[0])
    df['mode'] = df['mode'].astype("category")
    df['epoch'] = df['epoch'].astype(int)
    return df


# @memory.cache(ignore=['device', 'param_dict.output', 'param_dict.workers',
                      # 'param_dict.resume', 'param_dict.dataset_download',
                      # 'param_dict.device', 'param_dict.no_prefetcher'])
def get_acc_and_loss_over_training(param_dict, epochs_idx=slice(None),
                          mode='val', device='cpu'):
    train_out = train.train(param_dict)
    model, loader_train, loader_val, run_dir, pd_mom = train_out
    epochs = np.array(load_utils.get_epochs(run_dir))
    validate_loss_fn = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    def validate_loss_fn(out, target):
        return criterion(out, F.one_hot(target,
                                 num_classes=pd_mom['num_classes']).float())
    if epochs_idx is not None:
        epochs = epochs[epochs_idx]
    dfs = []
    for k1, epoch in enumerate(epochs):
        print(epoch)
        df = get_acc_and_loss(param_dict, epoch, mode, device)
        dfs.append(df)
    df = pd.concat(dfs)
    df['mode'] = df['mode'].astype("category")
    df['epoch'] = df['epoch'].astype(int)
    return df


def batch_fn(fn, param_dicts, *args, **kwargs):
    dfs = []
    for k1, param_dict in enumerate(param_dicts):
        print(f"Running {k1} / {len(param_dicts)}")
        dfs.append(fn(param_dict, *args, **kwargs))
    df = pd.concat(dfs, ignore_index=True)
    return df


def plots_df(data, x, y, height=height_default, aspect=1, figname='fig.pdf',
             **fig_kwargs):
    with plt.rc_context({'axes.titlesize':'xx-small'}):
        fg = sbn.relplot(data=data, x=x, y=y, height=height, aspect=aspect,
                         **fig_kwargs, kind='line')
        for i, ax in enumerate(fg.fig.axes):   ## getting all axes of the fig object
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90) 
        fg.tight_layout()
        fg.savefig(figname, bbox_inches='tight')
    # new_fig_vals = [pd.unique(df[key]) for key in new_fig_keys]



    # sbn.lineplot(data=df_train, x='epoch', y='compression_train')
    # fig, ax = plt.subplots(figsize=figsize_default)
    # sbn.lineplot(ax=ax, data=df, x='epoch', y='compression', hue='mode')
    # fig.savefig('test.pdf')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def plot_over_epochs(y, df, figdir):
    df = df.copy()
    # breakpoint()
    # df['mse_loss']=df['mse_loss'].astype(str)
    # breakpoint()
    # df[df['mse_loss']==True] = 'mse'
    # df[df['mse_loss']==False] = 'cce'
    # breakpoint()
    df.loc[df['mse_loss']==True, 'mse_loss'] = 'MSE'
    df.loc[df['mse_loss']==False, 'mse_loss'] = 'CCE'
    loss_fn_str = 'loss fun'
    mom_str = 'mom'
    df = df.rename(columns={'mse_loss': loss_fn_str})
    df = df.rename(columns={'momentum': mom_str})

    plot_keys = {'dataset', 'epoch', y, 'mode', mom_str,
                 loss_fn_str, 'opt', 'weight_decay', 'drop', 'layer_idx',
                 'layer_name'}
    plot_keys = list(plot_keys.intersection(df.columns))
    dfn = df[plot_keys].copy()
    dfn.loc[dfn['opt']=='rmsprop', mom_str] = 'rmsprop'
    dfn = dfn.drop(columns='opt')
    # dfn = dfn[dfn['loss function']<5]
    def set_cat(df):
        df[loss_fn_str]=df[loss_fn_str].astype('category')
        df['weight_decay']=df['weight_decay'].astype('category')
        df[mom_str]=df[mom_str].astype('category')
        return df

    figdir = Path(f'plots/epochs/{figdir}')
    figdir.mkdir(parents=True, exist_ok=True)
    def plot_dset(dset='torch/mnist'):
        dset_stripped = dset.split('/')[-1]
        filt = (
                (dfn['dataset']==dset) &
                # (dfn['opt']== mom_str) &
                # (dfn['layer_name']=='global_pool.flatten') &
                # (dfn['layer_name']=='fc') &
                (dfn['weight_decay']==0.0)
               )
        df1 = set_cat(dfn[filt])
        plots_df(df1, x='epoch', y=y, hue='drop', row=loss_fn_str,
                 col=mom_str,
                 figname=figdir/f'{dset_stripped}_sgd_drop.png')

        filt = (
                (dfn['dataset']==dset) &
                # (dfn['opt']== mom_str) &
                # (dfn['layer_name']=='global_pool.flatten') &
                # (dfn['layer_name']=='fc') &
                (dfn['drop']==0.0)
               )
        df1 = set_cat(dfn[filt])
        plots_df(df1, x='epoch', y=y,
                 hue='weight_decay',
                 row=loss_fn_str,
                 col=mom_str,
                 # figname=figdir/f'{dset_stripped}_sgd_weight_decay.png')
                 figname=figdir/f'{dset_stripped}_sgd_weight_decay.pdf')

        filt = (
                (dfn['dataset']==dset) &
                # (dfn['opt']=='rmsprop') &
                # (dfn['layer_name']=='global_pool.flatten') &
                # (dfn['layer_name']=='fc') &
                (dfn['weight_decay']==0.0)
               )
        df1 = set_cat(dfn[filt])
        plots_df(df1, x='epoch', y=y,
                 hue='drop',
                 row=loss_fn_str,
                 col=mom_str,
                 # figname=figdir/f'{dset_stripped}_rmsprop_drop.png')
                 figname=figdir/f'{dset_stripped}_rmsprop_drop.pdf')

        filt = (
                (dfn['dataset']==dset) &
                # (dfn['opt']=='rmsprop') &
                # (dfn['layer_name']=='global_pool.flatten') &
                # (dfn['layer_name']=='fc') &
                (dfn['drop']==0.0)
               )
        df1 = set_cat(dfn[filt])
        plots_df(df1, x='epoch', y=y,
                 hue='weight_decay',
                 row=loss_fn_str,
                 col=mom_str,
                 # figname=figdir/f'{dset_stripped}_rmsprop_weight_decay.png')
                 figname=figdir/f'{dset_stripped}_rmsprop_weight_decay.pdf')
        filt = (
                (dfn['dataset']==dset) &
                # (dfn['opt']=='rmsprop') &
                # (dfn['layer_name']=='global_pool.flatten') &
                # (dfn['layer_name']=='fc') &
                (dfn['drop']==0.0) &
                (dfn['weight_decay']==0.0)
               )
        df1 = set_cat(dfn[filt])
        plots_df(df1, x='epoch', y=y,
                 # hue='opt',
                 hue=mom_str,
                 col=loss_fn_str,
                 # figname=figdir/f'{dset_stripped}_opt_comp.png')
                 figname=figdir/f'{dset_stripped}_opt_comp.pdf')
        
    plot_dset('torch/mnist')
    plot_dset('torch/cifar10')

def plot_over_layers(y, df):

    plot_keys = ['dataset', 'epoch', y,
                 'mode', 'momentum', 'mse_loss', 'opt', 'weight_decay', 'drop',
                 'layer_idx', 'layer_name']
    dfn = df[plot_keys]
    dfn = dfn.sort_values('layer_idx', ignore_index=True)
    figdir=Path(f'plots/layers/{y}/')
    figdir.mkdir(parents=True, exist_ok=True)
    # breakpoint()
    dfn['epoch'] = dfn['epoch'].astype('category')
    def plot_dset(dset='torch/mnist'):
        dset_stripped = dset.split('/')[-1]
        filt = (
                (dfn['dataset']==dset) &
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
                 style_order=[350, 0],
                 row='mse_loss',
                 col='momentum',
                 height=2,
                 aspect=4,
                 figname=figdir/f'{dset_stripped}_sgd_drop.png')


        filt = (
                (dfn['dataset']==dset) &
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
                 style_order=[350, 0],
                 row='mse_loss',
                 col='momentum',
                 height=2,
                 aspect=4,
                 figname=figdir/f'{dset_stripped}_sgd_weight_decay.png')

        filt = (
                (dfn['dataset']==dset) &
                (dfn['opt']=='rmsprop') &
                (dfn['layer_name']!='dropout') &
                (dfn['weight_decay']==0.0)
               )
        df1 = dfn[filt]
        plots_df(df1, x='layer_name', y=y,
                 hue='drop',
                 style='epoch',
                 style_order=[350, 0],
                 col='mse_loss',
                 height=2,
                 aspect=4,
                 figname=figdir/f'{dset_stripped}_rmsprop_drop.png')

        filt = (
                (dfn['dataset']==dset) &
                (dfn['opt']=='rmsprop') &
                (dfn['layer_name']!='dropout') &
                (dfn['drop']==0.0)
               )
        df1 = dfn[filt]
        plots_df(df1, x='layer_name', y=y,
                 hue='weight_decay',
                 style='epoch',
                 style_order=[350, 0],
                 col='mse_loss',
                 height=2,
                 aspect=4,
                 figname=figdir/f'{dset_stripped}_rmsprop_weight_decay.png')
        # filt = (
                # (dfn['dataset']==dset) &
                # # (dfn['opt']=='rmsprop') &
                # # (dfn['layer_name']=='global_pool.flatten') &
                # # (dfn['layer_name']=='fc') &
                # (dfn['drop']==0.0)
                # (dfn['weight_decay']==0.0)
               # )
        # df1 = dfn[filt]
        # plots_df(df1, x='epoch', y=y,
                 # hue='opt',
                 # col='momentum',
                 # row='mse_loss',
                 # figname=figdir/f'{dset_stripped}_opt_comp.png')
    
    plot_dset('torch/mnist')
    plot_dset('torch/cifar10')

if __name__ == '__main__':
    fn = train.train
    ps_set1 = exp.ps_resnet18_mnist_sgd + exp.ps_resnet18_mnist_rmsprop
    ps_set2 = exp.ps_resnet18_cifar10_sgd + exp.ps_resnet18_cifar10_rmsprop
    ps_all = ps_set1 + ps_set2
    for ps in ps_all:
        ps['device'] = 'cpu'
    # ps_all = [exp.ps_resnet152_imagenet_pretrain]
    # ps_all = ps_set1
    print(run_num)
    if run_num is not None and run_num > len(ps_all):
        sys.exit()

    # df = get_acc_and_loss_over_training(ps_all[run_num-1], device='cuda')
    # # df = get_compressions_over_training(ps_all[run_num-1],
                                        # # epochs=[0, 5, 10, 20, -1],
                                        # # n_batches_per=4,
                                        # # # projection='s',
                                        # # # device='cpu')
                                        # # device='cuda')
    df = get_compressions_over_layers(ps_all[run_num], [0, -1],
                  device='cpu')
    # sys.exit()

    # df = batch_fn(get_acc_and_loss_over_training, ps_all, device='cuda')
    # plot_over_epochs('accuracy', df, 'accuracy')
    # df = batch_fn(get_compressions_over_training, ps_all, 
                  # epochs=[0, 5, 10, 20, -1],
                  # n_batches_per=4,
                  # device='cuda')
    # plot_over_epochs('compression', df, 'compression')
    # df = batch_fn(get_compressions_over_training, ps_all, 
                  # epochs=[0, 5, 10, 20, -1],
                  # n_batches_per=4,
                  # projection='s',
                  # device='cuda')
    # plot_over_epochs('compression', df, 'compression_proj')
    # sys.exit()

    # df = batch_fn(get_compressions_over_layers, ps_all, [0, -1],
                  # device='cpu')
                  # device='cuda')

