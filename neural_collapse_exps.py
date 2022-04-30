import timm
from timm import data
from timm import models
from matplotlib import pyplot as plt

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


vgg11 = timm.create_model('vgg11', in_chans=1)
cifar = data.create_dataset('torch/cifar10', 'cifar10', download=True,
                            split='val')

args = {}
args['input_size'] = (3, 32, 32)
args['mean'] = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
args['std'] = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
args['crop_pct'] = 1
data_config = data.config.resolve_data_config(args, model=vgg11)
loader_eval = data.create_loader(
    cifar,
    input_size=(3, 32, 32),
    batch_size=20,
    is_training=False,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    crop_pct=data_config['crop_pct'],
)
vald = next(iter(loader_eval))
model_out = vgg11(vald[0])
features = vgg11.forward_features(vald[0])
feature_pcs = get_pcs_covariance(features, [0,1,2], original_shape=False)   
# b1 = cifar[:20]

