import timm
import time
from sklearn.svm import LinearSVC as Classifier
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# from sklearn.linear_model import LogisticRegression as Classifier
import torch
from pathlib import Path
from timm import data
from timm import models
from matplotlib import pyplot as plt
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint

def vmean(v):
    return sum(v) / len(v)

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

def pairwise_class_dists(X, y, breakv=False):
    yl = y.tolist()
    cs = set(yl)
    m = len(cs)
    ds = torch.zeros(m, m)
    for k1, y1 in enumerate(cs):
        for k2, y2 in enumerate(cs):
            xk1 = X[y == y1]
            xk1 = xk1.reshape(1, xk1.shape[0], -1) 
            xk2 = X[y == y2]
            xk2 = xk2.reshape(1, xk2.shape[0], -1) 
            d = torch.cdist(xk1, xk2)[0]
            r1 = d.shape[0]
            r2 = d.shape[1]
            if k1 == k2:
                nu = r1*(r1-1)/2
                ds[k1, k2] = torch.sum(torch.triu(d,1)) / nu
            else:
                ds[k1, k2] = torch.mean(d)
    return ds 

# def within_over_across_class_mean_dist(X, y, breakv=False):
    # cs = set(y.tolist())
    # m = len(cs)
    # ds = torch.zeros(m,m)
    # for k1 in range(m):
        # for k2 in range(k1, m):
            # xk1 = X[y == k1]
            # xk1 = xk1.reshape(1, xk1.shape[0], -1) 
            # xk2 = X[y == k2]
            # xk2 = xk2.reshape(1, xk2.shape[0], -1) 
            # d = torch.cdist(xk1, xk2)[0]
            # r1 = d.shape[0]
            # r2 = d.shape[1]
            # if k1 == k2:
                # ds[k1, k2] = torch.sum(torch.triu(d,1)) / ((r1-1)*(r2-1)/2)
            # else:
                # ds[k1, k2] = torch.sum(torch.triu(d,0)) / (r1*r2/2)
            # # if ds[k1, k2] == torch.nan:
                # # breakpoint()
            
    # d_within = torch.nanmean(torch.diag(ds))
    # d_across = torch.sum(torch.triu(ds, 1)) / ((m-1)**2/2)

    # return d_within.item(), d_across.item()


# def get_compressions(feat_extractor, loader, run_dir, n_batches):
    # data_size = len(loader)
    # feat_col = []
    # labels_col = []
    # d_within = []
    # d_across = []
    # inpdata = []
    # labels = []
    # out_inputs = []
    # within_inputs = []
    # across_inputs = []
    # print("Reminder to check layer orderings.")
    # for k1, (inpdata_batch, labels_batch) in enumerate(loader):
        # inpdata += [inpdata_batch]
        # labels += [labels_batch]
        # if (k1 + 1) % n_batches == 0:
            # inpdata = torch.cat(inpdata, dim=0)
            # labels = torch.cat(labels, dim=0)
            # # features = feat_extractor(inpdata)
            # featt = feat_extractor(inpdata)
            # features = featt.values()
            # features = [feat.data.squeeze() for feat in features]
            # breakv = k1 >= len(loader)-1
            # out_layers = [within_over_across_class_mean_dist(feat, labels,
                                                             # breakv=breakv) for
                          # feat in features]
            # within_layers = [o[0] for o in out_layers]
            # across_layers = [o[1] for o in out_layers]
            
            # within_inputs += [within_layers]
            # across_inputs += [across_layers]
            # inpdata = []
            # labels = []
    # d_within = zip(*within_inputs)
    # d_across = zip(*across_inputs)

    # d_within_avgs = torch.tensor([vmean(d) for d in d_within])
    # d_across_avgs = torch.tensor([vmean(d) for d in d_across])
    # compression = d_within_avgs / d_across_avgs
    # return compression

def get_dists_projected(feat_extractor, loader, run_dir, n_batches,
                        lin_class_its, device):
    data_size = len(loader)
    feat_col = []
    labels_col = []
    ds_within_tot = []
    ds_across_tot = []
    ds_within_aligned_tot = []
    ds_across_aligned_tot = []
    ds_within_aligned_ratio_tot = []
    ds_across_aligned_ratio_tot = []
    inpdata = []
    labels = []
    out_inputs = []
    within_inputs = []
    across_inputs = []
    print("Reminder to check layer orderings.")
    print("Loading data.")
    tica = time.time()
    tic1 = time.time()
    tdiff_feat = []
    tdiff_d = []
    tdiff_class = []
    tdiff_proj_class = []
    for k1, (inpdata_batch, labels_batch) in enumerate(loader):
        toc1 = time.time()
        print(f"Data loaded in {toc1-tic1}s", flush=True)
        tic = time.time()
        print(k1, '/', len(loader), ' inputs')
        inpdata += [inpdata_batch.to(device)]
        labels += [labels_batch.to(device)]
        if (k1 + 1) % n_batches == 0:
            inpdata = torch.cat(inpdata, dim=0)
            labels = torch.cat(labels, dim=0)
            # labels_pm1 = (2*labels - 1).cpu()
            # features = feat_extractor(inpdata)
            tic1 = time.time()
            featt = feat_extractor(inpdata)
            toc1 = time.time()
            tdiff_feat.append(toc1-tic1)
            # print(f"Features generated in {toc1-tic1}s", flush=True)
            features = featt.values()
            features = [feat.data.squeeze() for feat in features]
            features_mc = [feat - torch.mean(feat, dim=0) for feat in features]
            features_mc = [feat.reshape(feat.shape[0], -1).cpu() for feat in
                           features_mc]
            ds_within_layers = []
            ds_across_layers = []
            ds_within_aligned_layers = []
            ds_across_aligned_layers = []
            ds_within_aligned_ratio_layers = []
            ds_across_aligned_ratio_layers = []
            for k2, feat in enumerate(features_mc):
                # ticf = time.time()
                tic1 = time.time()
                ds = pairwise_class_dists(feat, labels)
                toc1 = time.time()
                tdiff_d.append(toc1-tic1)
                m = ds.shape[0]
                ds_within_layers.append(
                    (torch.nanmean(torch.diag(ds))).item())
                ds_across_layers.append(
                    (torch.sum(torch.triu(ds, 1))/((m-1)**2/2)).item())
                tic1 = time.time()
                classf = Classifier(max_iter=lin_class_its, C=10)
                classf.fit(feat.cpu().numpy(), labels.cpu().numpy())
                toc1 = time.time()
                tdiff_class.append(toc1-tic1)
                w = torch.tensor(classf.coef_, dtype=torch.float)
                b = torch.tensor(classf.intercept_, dtype=torch.float)
                wn = w / torch.norm(w, dim=1, keepdim=True)
                D = w.shape[1]
                # P_orths = torch.stack(
                    # [torch.eye(D) - torch.outer(w, w) for w in wn])
                # feat_align = 
                # out_layers = within_over_across_class_mean_dist(feat, labels)
                feat_aligned = wn @ feat.cpu().T + b.unsqueeze(dim=1)
                ds_within_aligned = []
                ds_across_aligned = []
                ds_within_aligned_ratio = []
                ds_across_aligned_ratio = []
                # print(f"Computing projected classifications.", flush=True)
                tic1 = time.time()
                for k3 in range(m):
                    ds_aligned = pairwise_class_dists(
                        feat_aligned[k3], labels)
                    ds_aligned_ratio = ds_aligned / ds
                    ds_orth_ratio = 1 - ds_aligned_ratio
                    ds_within_aligned.append(torch.nanmean(
                        torch.diag(ds_aligned)).item())
                    ds_across_aligned.append(
                        (torch.sum(torch.triu(ds_aligned, 1))/(m*(m-1)/2)).item())
                    ds_within_aligned_ratio.append(torch.nanmean(
                        torch.diag(ds_aligned_ratio)).item())
                    ds_across_aligned_ratio.append(
                        (torch.sum(torch.triu(ds_aligned_ratio, 1))/(m*(m-1)/2)).item())
                toc1 = time.time()
                tdiff_proj_class.append(toc1-tic1)
                # print(f"Finished computing projected classifications in",
                      # f"{toc1-tic1}s", flush=True)
                ds_within_aligned_layers.append(vmean(ds_within_aligned))
                ds_across_aligned_layers.append(vmean(ds_across_aligned))
                ds_within_aligned_ratio_layers.append(vmean(ds_within_aligned_ratio))
                ds_across_aligned_ratio_layers.append(vmean(ds_across_aligned_ratio))
                # tocf = time.time()
                # print(f"Time this layer: {tocf-ticf}")

            ds_within_tot.append(ds_within_layers)
            ds_across_tot.append(ds_across_layers)
            ds_within_aligned_tot.append(ds_within_aligned_layers)
            ds_across_aligned_tot.append(ds_across_aligned_layers)
            ds_within_aligned_ratio_tot.append(ds_within_aligned_ratio_layers)
            ds_across_aligned_ratio_tot.append(ds_across_aligned_ratio_layers)

            inpdata = []
            labels = []
            toc = time.time()
            print('time elapsed:', toc-tic)
            print('avg feature compute time:', vmean(tdiff_feat))
            print('avg distance compute time:', vmean(tdiff_d))
            print('avg classification compute time:', vmean(tdiff_class))
            print('avg projected classification compute time:', vmean(tdiff_proj_class))
            tdiff_feat = []
            tdiff_d = []
            tdiff_class = []
            tdiff_proj_class = []
        tic1 = time.time()

    ds_within_tot = zip(*ds_within_tot)
    ds_across_tot = zip(*ds_across_tot)
    ds_within_aligned_tot = zip(*ds_within_aligned_tot)
    ds_across_aligned_tot = zip(*ds_across_aligned_tot)
    # d_within_orth_ratio_tot = 1 - ds_within_aligned_ratio_tot
    # ds_across_orth_ratio_tot = 1 - ds_across_aligned_ratio_tot
    ds_within_aligned_ratio_tot = zip(*ds_within_aligned_ratio_tot)
    ds_across_aligned_ratio_tot = zip(*ds_across_aligned_ratio_tot)
    # ds_within_orth_ratio_tot = zip(*ds_within_orth_ratio_tot)
    # ds_across_orth_ratio_tot = zip(*ds_across_orth_ratio_tot)

    ds_within_avgs = torch.tensor([vmean(d) for d in ds_within_tot])
    ds_across_avgs = torch.tensor([vmean(d) for d in ds_across_tot])
    ds_within_aligned_avgs = torch.tensor([vmean(d) for d in
                                          ds_within_aligned_tot])
    ds_across_aligned_avgs = torch.tensor([vmean(d) for d in
                                          ds_across_aligned_tot])
    ds_within_aligned_ratio_avgs = torch.tensor([vmean(d) for d in
                                          ds_within_aligned_ratio_tot])
    ds_across_aligned_ratio_avgs = torch.tensor([vmean(d) for d in
                                          ds_across_aligned_ratio_tot])
    # ds_within_orth_ratio_avgs = torch.tensor([vmean(d) for d in
                                          # ds_orth_aligned_ratio_tot])
    # ds_across_orth_ratio_avgs = torch.tensor([vmean(d) for d in
                                          # ds_orth_aligned_ratio_tot])
    # compression_avgs = ds_within_avgs / ds_across_avgs
    # compression_aligned_ratio_avgs = d_within_aligned_ratio_tot / d_across_aligned_ratio_tot
    # compression_orth_ratio_avgs = d_within_orth_ratio_tot / d_across_orth_ratio_tot 
    return (ds_within_avgs, ds_across_avgs, ds_within_aligned_avgs,
            ds_across_aligned_avgs, ds_within_aligned_ratio_avgs,
            ds_across_aligned_ratio_avgs)
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
