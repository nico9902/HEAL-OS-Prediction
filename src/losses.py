import torch
import torch.nn as nn

# user-defined functions
_EPSILON = 1e-08
def log(x):
    return torch.log(x + _EPSILON)

def div(x, y):
    return torch.div(x, (y + _EPSILON))

# create loss function
# LOSS-FUNCTION 1 -- Log-likelihood loss
def loss_Log_Likelihood(out, mask1, k):
    
    I_1 = torch.sign(k)

    # for uncenosred: log P(T=t,K=k|x)
    tmp1 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
    tmp1 = I_1 * log(tmp1)

    # for censored: log \sum P(T>t|x)
    tmp2 = torch.sum(torch.sum(mask1 * out, dim=2), dim=1, keepdim=True)
    tmp2 = (1. - I_1) * log(tmp2)

    loss1 = - torch.mean(tmp1 + 1.0*tmp2)

    return loss1

# LOSS-FUNCTION 2 -- Ranking loss
def loss_Ranking(out, num_Event, num_Category, mask2, t, k):
    
    sigma1 = 0.1

    eta = []
    for e in range(num_Event):
        one_vector = torch.ones_like(t, dtype=torch.float32)
        I_2 = torch.eq(k, e+1)
        I_2 = I_2.type(torch.float32) #indicator for event
        I_2 = torch.diag(torch.squeeze(I_2))
        tmp_e = torch.reshape(out[:,e,:], (-1, num_Category)) #event specific joint prob.
        #tmp_e = torch.reshape(tf.slice(out, [0, e, 0], [-1, 1, -1]), (-1, num_Category)) #event specific joint prob.

        R = torch.matmul(tmp_e, torch.transpose(mask2, 0, 1)) #no need to divide by each individual dominator
        # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

        diag_R = torch.reshape(torch.diagonal(R,0), (-1, 1))
        R = torch.matmul(one_vector, torch.transpose(diag_R, 0, 1)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = torch.transpose(R, 0, 1)                                    # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        criterion = nn.ReLU()
        T = criterion(torch.sign(torch.matmul(one_vector, torch.transpose(t,0,1)) - torch.matmul(t, torch.transpose(one_vector,0,1))))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        T = torch.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i

        tmp_eta = torch.mean(T * torch.exp(-R/sigma1), dim=1, keepdim=True)

        eta.append(tmp_eta)
    eta = torch.stack(eta, dim=1) #stack referenced on subjects
    eta = torch.mean(torch.reshape(eta, (-1, num_Event)), dim=1, keepdim=True)

    loss2 = torch.sum(eta) #sum over num_Events
    return loss2

# LOSS-FUNCTION 3 -- Calibration Loss
def loss_Calibration(out, num_Event, num_Category, mask2, t, k):

    eta = []
    for e in range(num_Event):
        one_vector = torch.ones_like(t, dtype=torch.float32)
        I_2 = torch.eq(k, e+1)
        I_2 = I_2.type(torch.float32) #indicator for event
        tmp_e = torch.reshape(out[:,e,:], (-1, num_Category)) #event specific joint prob.

        r = torch.sum(tmp_e * mask2, dim=0) #no need to divide by each individual dominator
        tmp_eta = torch.mean((r - I_2)**2, dim=1, keepdim=True)

        eta.append(tmp_eta)
    eta = torch.stack(eta, dim=1) #stack referenced on subjects
    eta = torch.mean(torch.reshape(eta, (-1, num_Event)), dim=1, keepdim=True)

    loss3 = torch.sum(eta) #sum over num_Events
    return loss3

# DeepHit loss function
def DeepHit_loss(out, alpha, beta, gamma, num_Event, num_Category, mask1, mask2, t, k):
    """
    DeepHit loss function
        out: predicted output
        alpha: weight for Log-likelihood loss
        beta: weight for Ranking loss
        gamma: weight for Calibration loss
        num_Event: number of events
        num_Category: number of categories
        mask1: mask for Log-likelihood loss
        mask2: mask for Ranking and Calibration loss
        t: time
        k: event
    return: DeepHit loss
    """
    loss = alpha*loss_Log_Likelihood(out, mask1, k) + beta*loss_Ranking(out, num_Event, num_Category, mask2, t, k) + gamma*loss_Calibration(out, num_Event, num_Category, mask2, t, k)
    return loss
