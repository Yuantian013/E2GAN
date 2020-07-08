from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models_search
import datasets
from utils.utils import set_log_dir, save_checkpoint, create_logger, RunningStats
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from functions import *
from sac import SAC
from replay_memory import ReplayMemory
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def create_shared_gan(args, weights_init):
    gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
    dis_net = eval('models_search.'+args.dis_model+'.Discriminator')(args=args).cuda()
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    return gen_net, dis_net, gen_optimizer, dis_optimizer


def main():
    args = cfg.parse_args()
    
    torch.cuda.manual_seed(args.random_seed)
    print(args)    
    

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init)

    # initial
    start_search_iter = 0

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        cur_stage = checkpoint['cur_stage']

        start_search_iter = checkpoint['search_iter']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        prev_archs = checkpoint['prev_archs']
        prev_hiddens = checkpoint['prev_hiddens']

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (search iteration {start_search_iter})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        prev_archs = None
        prev_hiddens = None

        # set controller && its optimizer
        cur_stage = 0

    # set up data_loader
    dataset = datasets.ImageDataset(args, 2**(cur_stage+3))
    train_loader = dataset.train
    print(args.rl_num_eval_img,"##############################")
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'controller_steps': start_search_iter * args.ctrl_step
    }

    g_loss_history = RunningStats(args.dynamic_reset_window)
    d_loss_history = RunningStats(args.dynamic_reset_window)

    # train loop
    Agent=SAC(131)
    print(Agent.alpha)
    
    memory = ReplayMemory(2560000)
    updates=0
    outinfo = {'rewards': [],
                'a_loss': [],
                'critic_error': [],
                }
    Best=False
    Z_NUMPY=None
    WARMUP=True
    update_time=1
    for search_iter in tqdm(range(int(start_search_iter), 100), desc='search progress'):
        logger.info(f"<start search iteration {search_iter}>")
        if search_iter>=1:
            WARMUP=False

        ### Define number of layers, currently only support 1->3
        total_layer_num = 3
        ### Different image size for different layers
        ds = [datasets.ImageDataset(args, 2 ** (k + 3)) for k in range(total_layer_num)]
        train_loaders = [d.train for d in ds]
        last_R = 0. # Initial reward
        last_fid=10000  # Inital reward
        last_arch = [] 

        # Set exploration 
        if search_iter > 69: 
            update_time=10
            Best=True
        else:
            Best=False


        gen_net.set_stage(-1)
        last_R,last_fid,last_state = get_is(args, gen_net, args.rl_num_eval_img, get_is_score=True)
        for layer in range(total_layer_num):

            cur_stage = layer # This defines which layer to use as output, for example, if cur_stage==0, then the output will be the first layer output. Set it to 2 if you want the output of the last layer.
            action = Agent.select_action([layer, last_R,0.01*last_fid] + last_state,Best)
            arch = [action[0][0],action[0][1],action[1][0],action[1][1],action[1][2],action[2][0],action[2][1],action[2][2],action[3][0],action[3][1],action[4][0],action[4][1],action[5][0],action[5][1]]
            print(arch)
            # argmax to get int description of arch    
            cur_arch = [np.argmax(k) for k in action]
            # Pad the skip option 0=False (for only layer 1 and layer2, not layer0, see builing_blocks.py for why)
            if layer == 0:
                cur_arch =cur_arch[0:4]
            elif layer == 1:
                cur_arch =cur_arch[0:5] 
            elif layer == 2:
                if cur_arch[4]+cur_arch[5]==2:
                    cur_arch = cur_arch[0:4]+ [3]
                elif cur_arch[4]+cur_arch[5]==0:
                    cur_arch = cur_arch[0:4]+ [0]
                elif cur_arch[4]==1 and cur_arch[5]==0:
                    cur_arch = cur_arch[0:4]+ [1]
                else :
                    cur_arch = cur_arch[0:4] +[2]

            # Get the network arch with the new architecture attached.
            last_arch += cur_arch 
            gen_net.set_arch(last_arch, layer) # Set the network, given cur_stage
            # Train network
            dynamic_reset = train_qin(args, gen_net, dis_net, g_loss_history, d_loss_history, gen_optimizer,
                                     dis_optimizer, train_loaders[layer], cur_stage, smooth=False, WARMUP=WARMUP) 
       
            # Get reward, use the jth layer output for generation. (layer 0:j)
            R, fid,state = get_is(args, gen_net, args.rl_num_eval_img,z_numpy=Z_NUMPY)
            print("arch:",cur_arch,Best)
            print("update times:",updates,"step:",layer+1,"IS:",R,"FID:",fid)
            mask = 0 if layer == total_layer_num-1 else 1
            if search_iter >=0: # warm up
                memory.push([layer,last_R,0.01*last_fid]+last_state, arch, R-last_R+0.01*(last_fid-fid), [layer+1,R,0.01*fid] + state, mask)  # Append transition to memory
                     
                       
            if len(memory) >= 64:
                # Number of updates per step in environment
                for i in range(update_time):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = Agent.update_parameters(memory, min(len(memory),256), updates)
       
                    updates += 1 
                    outinfo['critic_error']=min(critic_1_loss, critic_2_loss)
                    outinfo['entropy']=ent_loss
                    outinfo['a_loss']=policy_loss
                print("full batch",outinfo,alpha)          
            last_R = R # next step
            last_fid = fid
            last_state = state
        outinfo['rewards']=R
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = Agent.update_parameters(memory, len(memory), updates)
        updates += 1 
        outinfo['critic_error']=min(critic_1_loss, critic_2_loss)
        outinfo['entropy']=ent_loss
        outinfo['a_loss']=policy_loss
        print("full batch",outinfo,alpha)       
        del gen_net, dis_net, gen_optimizer, dis_optimizer
        gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init)
        print(outinfo,len(memory))
        Agent.save_model("test")
        WARMUP=False



if __name__ == '__main__':
    main()
