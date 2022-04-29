#!/bin/bash
#
#SBATCH --mail-user=msfarrell@g.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH -p pehlevan_gpu # partition (queue)
#SBATCH --gres=gpu:2
#SBATCH -N 1 # number of nodes
#SBATCH -c 16 # number of nodes
#SBATCH --mem=20GB
#SBATCH -t 2-0:00 # time (D-HH:MM)
#SBATCH -o sbatch/%j.out 
#SBATCH -e sbatch/%j.err 
./distributed_train.sh 2 /n/pehlevan_lab/Everyone/imagenet/ILSVRC/Data/CLS-LOC \
--model efficientnet_b2 -b 128 --sched step --epochs 450 --decay-epochs 2.4 \
--decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 \
--weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema \
--model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 \
--lr .016 --smoothing 0

