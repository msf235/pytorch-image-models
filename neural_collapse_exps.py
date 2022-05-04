epochs = 350
core_params = dict(data_dir='data',
   output='/n/holyscratch01/pehlevan_lab/Lab/matthew/output_neural_collapse',
   dataset_download=True, cooldown_epochs=0,
   smoothing=0, sched='multistep', decay_rate=0.1,
   epochs=epochs, decay_epochs=[epochs//3, epochs*2//3],
   batch_size=128, weight_decay=0, momentum=0,
   interpolation='', train_interpolation='',
   check_point_every=1, resume=True,
)

resnet_mnist_mse_sgd_vary_momentum_and_decay = core_params.copy()
resnet_mnist_mse_sgd_vary_momentum_and_decay.update(
    dict(dataset='torch/mnist', num_classes=10,
         opt='momentum', # SGD without nesterov
         lr=0.0184, input_size=(1,28,28), mean=(0.1307,), std=(0.3081,))
)
