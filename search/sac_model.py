import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models_search.building_blocks_search import CONV_TYPE, NORM_TYPE, UP_TYPE, SHORT_CUT_TYPE, SKIP_TYPE

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim).cuda()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.linear3 = nn.Linear(hidden_dim, 1).cuda()

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim).cuda()
        self.linear5 = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.linear6 = nn.Linear(hidden_dim, 1).cuda()

        self.apply(weights_init_)

    def forward(self, state, action):
        ps = action.cuda()
        a1, a2, a3, a4, a5, a6 = ps[:, :2], ps[:, 2:2+3], ps[:, 2+3:2+3+3], ps[:, 2+3+3: 2+3+3+2], ps[:, 2+3+3+2:2+3+3+2+2], ps[:, 2+3+3+2+2:2+3+3+2+2+2]
        a1_ = a1 
        a2_ = a2 
        a3_ = a3 
        a4_ = a4
        a5_ = a5 
        a6_ = a6 
        a = torch.cat([a1_, a2_, a3_, a4_, a5_, a6_], dim=1)
        xu = torch.cat([state.cuda(), a], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim).cuda()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).cuda()

        self.mean_linear_1 = nn.Linear(hidden_dim, len(CONV_TYPE)).cuda()
        self.mean_linear_2 = nn.Linear(hidden_dim, len(NORM_TYPE)).cuda()
        self.mean_linear_3 = nn.Linear(hidden_dim, len(UP_TYPE)).cuda()
        self.mean_linear_4 = nn.Linear(hidden_dim, len(SHORT_CUT_TYPE)).cuda()
        self.mean_linear_5 = nn.Linear(hidden_dim, len(SKIP_TYPE)).cuda()
        self.mean_linear_6 = nn.Linear(hidden_dim, len(SKIP_TYPE)).cuda()

        self.log_std_linear_1 = nn.Linear(hidden_dim, len(CONV_TYPE)).cuda()
        self.log_std_linear_2 = nn.Linear(hidden_dim, len(NORM_TYPE)).cuda()
        self.log_std_linear_3 = nn.Linear(hidden_dim, len(UP_TYPE)).cuda()
        self.log_std_linear_4 = nn.Linear(hidden_dim, len(SHORT_CUT_TYPE)).cuda()
        self.log_std_linear_5 = nn.Linear(hidden_dim, len(SKIP_TYPE)).cuda()
        self.log_std_linear_6 = nn.Linear(hidden_dim, len(SKIP_TYPE)).cuda()


        self.apply(weights_init_)
 

    def forward(self, state):
        x = F.relu(self.linear1(state.cuda()))
        x = F.relu(self.linear2(x))

        mean_1 = self.mean_linear_1(x)
        log_std_1 = self.log_std_linear_1(x)
        log_std_1 = torch.clamp(log_std_1, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        mean_2 = self.mean_linear_2(x)
        log_std_2 = self.log_std_linear_2(x)
        log_std_2 = torch.clamp(log_std_2, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        mean_3 = self.mean_linear_3(x)
        log_std_3 = self.log_std_linear_3(x)
        log_std_3 = torch.clamp(log_std_3, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        mean_4 = self.mean_linear_4(x)
        log_std_4 = self.log_std_linear_4(x)
        log_std_4 = torch.clamp(log_std_4, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        mean_5= self.mean_linear_5(x)  
        log_std_5 = self.log_std_linear_5(x)
        log_std_5 = torch.clamp(log_std_5, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        mean_6= self.mean_linear_6(x)  
        log_std_6 = self.log_std_linear_6(x)
        log_std_6 = torch.clamp(log_std_6, min=LOG_SIG_MIN, max=LOG_SIG_MAX)


        
        return mean_1, log_std_1,mean_2, log_std_2,mean_3, log_std_3,mean_4, log_std_4,mean_5,log_std_5,mean_6,log_std_6
        

    def sample(self, state):
            mean_1, log_std_1,mean_2, log_std_2,mean_3, log_std_3,mean_4, log_std_4,mean_5, log_std_5,mean_6, log_std_6= self.forward(state)

            std_1 = log_std_1.exp()
            std_2 = log_std_2.exp()
            std_3 = log_std_3.exp()
            std_4 = log_std_4.exp()
            std_5 = log_std_5.exp()
            std_6 = log_std_6.exp()

            normal_1 = Normal(mean_1, std_1)
            normal_2 = Normal(mean_2, std_2)
            normal_3 = Normal(mean_3, std_3)
            normal_4 = Normal(mean_4, std_4)
            normal_5 = Normal(mean_5, std_5)
            normal_6 = Normal(mean_6, std_6)

            x_t_1 = normal_1.rsample()  
            x_t_2 = normal_2.rsample()  
            x_t_3 = normal_3.rsample()  
            x_t_4 = normal_4.rsample()  
            x_t_5 = normal_5.rsample()  
            x_t_6 = normal_6.rsample()  

            a1 = torch.tanh(x_t_1)
            a2 = torch.tanh(x_t_2)
            a3 = torch.tanh(x_t_3)
            a4 = torch.tanh(x_t_4)
            a5 = torch.tanh(x_t_5)
            a6 = torch.tanh(x_t_6)


            log_prob_1 = normal_1.log_prob(x_t_1)
            log_prob_2 = normal_2.log_prob(x_t_2)
            log_prob_3 = normal_3.log_prob(x_t_3)
            log_prob_4 = normal_4.log_prob(x_t_4)
            log_prob_5 = normal_5.log_prob(x_t_5)
            log_prob_6 = normal_6.log_prob(x_t_6)

            # Enforcing Action Bound
            log_prob_1 -= torch.log((1 - a1.pow(2)) + epsilon)
            log_prob_1 = log_prob_1.sum(1, keepdim=True)

            log_prob_2 -= torch.log((1 - a2.pow(2)) + epsilon)
            log_prob_2 = log_prob_2.sum(1, keepdim=True)

            log_prob_3 -= torch.log((1 - a3.pow(2)) + epsilon)
            log_prob_3 = log_prob_3.sum(1, keepdim=True)

            log_prob_4 -= torch.log((1 - a4.pow(2)) + epsilon)
            log_prob_4 = log_prob_4.sum(1, keepdim=True)

            log_prob_5 -= torch.log((1 - a5.pow(2)) + epsilon)
            log_prob_5 = log_prob_5.sum(1, keepdim=True)

            log_prob_6 -= torch.log((1 - a6.pow(2)) + epsilon)
            log_prob_6 = log_prob_6.sum(1, keepdim=True)

            mean_1 = torch.tanh(mean_1)
            mean_2 = torch.tanh(mean_2)
            mean_3 = torch.tanh(mean_3)
            mean_4 = torch.tanh(mean_4)
            mean_5 = torch.tanh(mean_5)
            mean_6 = torch.tanh(mean_6)

           
            return a1,a2,a3,a4,a5,a6,log_prob_1,log_prob_2,log_prob_3,log_prob_4,log_prob_5,log_prob_6,mean_1,mean_2,mean_3,mean_4,mean_5,mean_6
            
