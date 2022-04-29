#!/bin/bash
#
#SBATCH --mail-user=msfarrell@seas.harvard.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH -p pehlevan_gpu # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 8
#SBATCH --mem=30G
#SBATCH --gres=gpu:2
#SBATCH -t 1-6:00 # time (D-HH:MM)
#SBATCH -o %j.out # STDOUT
#SBATCH -e %j.err # STDERR
#python $1
./distributed_train.sh 2 /n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC \
--model efficientnet_b2 -b 128 --sched step --epochs 450 --decay-epochs 2.4 \
--decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 \
--weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema \
--model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 \
--native-amp --lr .016 --checkpoint-hist 10 --save-every 10 --disable-bn \
--experiment nobn_nopretrain
