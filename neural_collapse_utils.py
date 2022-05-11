import timm
import torch
from pathlib import Path
from timm import data
from timm import models
from matplotlib import pyplot as plt
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint

def get_pcs_covariance(X, pcs, original_shape=True, return_extra=False):
    """
        Return principal components of X (using the covariance matrix).
        Args:
            X ([num_samples, ambient space dimension]): Data matrix of samples where each sample corresponds to a row of
                X.
            pcs ([num_pcs,]): List of principal components to return.

        Returns:
            pca_proj: ([num_pcs, ambient space dimension]): Projection of X onto principal components given by pcs.

        """
    N = X.shape[0]
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])
        print("Warning: concatenated last however many dimensions to get square data array")
    mu = torch.mean(X, dim=0)
    X = X - mu
    if X.shape[0] < X.shape[1]:
        X = X.T
    cov = X.T @ X / (N - 1)
    # eig, ev = torch.symeig(cov, eigenvectors=True)
    # ind = torch.argsort(torch.abs(eig), descending=True)
    eig, ev = torch.linalg.eigh(cov)
    ind = torch.argsort(eig.abs(), descending=True)
    ev = ev[:, ind]
    # pca_proj = np.dot(ev[:, pcs].T, X.T)
    pca_proj = X @ ev[:, pcs]
    if original_shape:
        pca_proj = pca_proj.reshape(*X.shape[:-1], pca_proj.shape[-1])
    if return_extra:
        return {'pca_projection': pca_proj, 'pca_projectors': ev, 'mean': mu}
    else:
        return pca_proj

def within_over_across_class_mean_dist(X, y, breakv=False):
    cs = set(y.tolist())
    m = len(cs)
    ds = torch.zeros(m,m)
    for k1 in range(m):
        for k2 in range(k1, m):
            xk1 = X[y == k1]
            xk1 = xk1.reshape(1, xk1.shape[0], -1) 
            xk2 = X[y == k2]
            xk2 = xk2.reshape(1, xk2.shape[0], -1) 
            d = torch.cdist(xk1, xk2)[0]
            r1 = d.shape[0]
            r2 = d.shape[1]
            if k1 == k2:
                ds[k1, k2] = torch.sum(torch.triu(d,1)) / ((r1-1)*(r2-1)/2)
            else:
                ds[k1, k2] = torch.sum(torch.triu(d,0)) / (r1*r2/2)
            # if ds[k1, k2] == torch.nan:
                # breakpoint()
            
    d_within = torch.nanmean(torch.diag(ds))
    d_across = torch.sum(torch.triu(ds, 1)) / ((m-1)**2/2)

    return d_within.item(), d_across.item()

def get_compressions(feat_extractor, loader, run_dir, n_batches):
    data_size = len(loader)
    feat_col = []
    labels_col = []
    d_within = []
    d_across = []
    inpdata = []
    labels = []
    out_inputs = []
    within_inputs = []
    across_inputs = []
    print("Reminder to check layer orderings.")
    for k1, (inpdata_batch, labels_batch) in enumerate(loader):
        inpdata += [inpdata_batch]
        labels += [labels_batch]
        if (k1 + 1) % n_batches == 0:
            inpdata = torch.cat(inpdata, dim=0)
            labels = torch.cat(labels, dim=0)
            # features = feat_extractor(inpdata)
            featt = feat_extractor(inpdata)
            features = featt.values()
            features = [feat.data.squeeze() for feat in features]
            breakv = k1 >= len(loader)-1
            out_layers = [within_over_across_class_mean_dist(feat, labels,
                                                             breakv=breakv) for
                          feat in features]
            within_layers = [o[0] for o in out_layers]
            across_layers = [o[1] for o in out_layers]
            # lr = range(len(within_layers))
            # zt = [h for h in within_layers if h != 0]
            # within_layers = [within_layers[k] for k in lr if zt[k]]
            # across_layers = [across_layers[k] for k in lr if zt[k]]
            # within_layers = [h for h in within_layers if h != 0]
            # zt = [w!=0 for w in within_layers]
            # within_layers = []
            
            within_inputs += [within_layers]
            across_inputs += [across_layers]
            inpdata = []
            labels = []
    d_within = zip(*within_inputs)
    d_across = zip(*across_inputs)

    d_within_avgs = torch.tensor([sum(d) / len(d) for d in d_within])
    d_across_avgs = torch.tensor([sum(d) / len(d) for d in d_across])
    compression = d_within_avgs / d_across_avgs
    return compression

# %% 
# if __name__ == '__main__':
    
    # vgg13 = timm.create_model('vgg13', num_classes=10)
    # outdir = Path('output/train/run_3/')
    # load_file = outdir/'last.pth.tar'
# # load_file = 'output/train/run_1'
    # final_epoch = resume_checkpoint(vgg13, load_file)
# # vgg13.cuda()
# # vgg13.cpu()

# # vgg11.load_state_dict(load_file)
# # %% 
    # cifar = data.create_dataset('torch/cifar10', 'cifar10', download=True,
                                # split='train')
                                # # split='val')

    # loader_eval = data.create_loader(
        # cifar,
        # input_size=(3, 32, 32),
        # batch_size=1000,
        # is_training=False,
    # )
    # vald = next(iter(loader_eval))
    # valdx = vald[0].cpu()
    # valdy = vald[1].cpu()
    # del vald
    # model_out = vgg13(valdx)
    # features = vgg13.forward_features(valdx).data.squeeze()
    # U, S, V = torch.svd(features)
    # feature_pcs = U[:, :3] * S[:3]
# # feature_pcs = get_pcs_covariance(features, [0,1,2])   
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(feature_pcs[:,0], feature_pcs[:,1], feature_pcs[:,2],
              # c=valdy)
    # fig.savefig(
    # fig.show()
# # b1 = cifar[:20]

    # # collapse = within_over_across_class_mean_dist(valdx, valdy)
