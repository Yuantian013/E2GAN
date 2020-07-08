# E2GAN
Code used for [Off-Policy Reinforcement Learning for Efficient and Effective GAN Architecture Search](https://), ECCV 2020. 


### Dependencies
```bash
conda create --name e2ganrl python=3.6
conda activate e2ganrl

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch

python3 -m pip install imageio
python3 -m pip install scipy
python3 -m pip install six
python3 -m pip install numpy==1.18.1
python3 -m pip install python-dateutil==2.7.3
python3 -m pip install tensorboardX==1.6
# For the reward calculation, external tf code
python3 -m pip install tensorflow-gpu==1.13.1
python3 -m pip install tqdm==4.29.1
```

### Prepare fid statistic file
Download the pre-calculated statistics from AutoGAN
([Google Drive](https://drive.google.com/drive/folders/1UUQVT2Zj-kW1c2FJOFIdGdlDHA3gFJJd?usp=sharing)) to `./search/fid_stat`and `./eval/fid_stat` . 


## Run NAS for GAN search on CIFAR-10
```bash
cd search
bash exps/search.sh
```
You can find the architectures in the log file search.log . 

## Train from scratch the discovered architecture
To train from scratch and get the performance of your discovered architecture, run the following command (you should replace the architecture vector following "--arch" in the script with candidate architectures that have highest rewards in the exploitation stage in search.log):

```bash
cd eval
# Train the discovered GAN on CIFAR-10
bash exps/train_derived.sh
# Train the discovered GAN on STL
bash exps/train_derived_stl.sh
```

## Test the discovered architecture reported in the paper

### Test

Run the following script:
```bash
cd eval
# Testing the pretrained CIFAR-10 Model
bash exps/test.sh
# Testing the pretrained STL Model
bash exps/test_stl.sh
```
Pre-trained models (both CIFAR and STL) are provided ([Google Drive](https://drive.google.com/drive/folders/1MGJjqsvJBxqfDLlelUarYZUfWTUOwEVt?usp=sharing)). You should put them in `eval/checkpoints/` .

## Citation
Please cite our work if you find it useful.
```bibtex
@InProceedings{Yuan_2020_ECCV,
author = {Yuan Tian, Qin Wang, Zhiwu Huang, Wen Li, Dengxin Dai, Minghao Yang, Jun Wang, Olga Fink},
title = {Off-Policy Reinforcement Learning for Efficient and Effective GAN Architecture Search},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2020}
}
```

## Acknowledgement
1. Inception Score code from [OpenAI's Improved GAN](https://github.com/openai/improved-gan/tree/master/inception_score) (official).
2. FID code and CIFAR-10 statistics file from [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) (official).
3. SAC code from [https://github.com/pranz24/pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic).
4. The code is heavily borrowed from AutoGAN [https://github.com/TAMU-VITA/AutoGAN](https://github.com/TAMU-VITA/AutoGAN)

