from timm import train
import neural_collapse_exps as exp

outdir = exp.core_params['output']

out = train.train(exp.resnet_mnist_mse_sgd_vary_momentum_and_decay)
model, loader_train, loader_eval, eval_metrics = out
