#!/usr/bin/env bash
# This code works fine on my RTX2080TI with 11GB memory.
# You may consider reducing batchsize/learning rate if you are using a GPU with smaller memory
CUDA_VISIBLE_DEVICES=2 python3 -u search.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--gen_model shared_gan \
--dis_model shared_gan \
--controller controller \
--latent_dim 128 \
--gf_dim 128 \
--df_dim 64 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--ctrl_sample_batch 1 \
--shared_epoch 15 \
--grow_step1 15 \
--grow_step2 35 \
--max_search_iter 65 \
--ctrl_step 30 \
--random_seed 12345 \
--exp_name autogan_search  | tee search.log
