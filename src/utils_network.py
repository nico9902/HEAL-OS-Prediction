import torch
import torch.nn as nn
import torch.nn.functional as F

# safe softmax function to handle -inf values in the input tensor
def safe_softmax(x, dim=-1):
    is_inf_mask = torch.all(x == float('-inf'), dim=dim)
    num_inf_samples = torch.sum(is_inf_mask).item()

    if num_inf_samples == x.size(0):
        # special case when all samples have all elements as -inf, return a tensor of zeros
        return torch.zeros_like(x)
    elif num_inf_samples > 0:
        # handle samples with all elements as -inf separately
        x_non_inf = x[~is_inf_mask]
        softmax_output = torch.zeros_like(x)
        softmax_output[~is_inf_mask] = F.softmax(x_non_inf, dim=dim)
        return softmax_output
    else:
        # use standard F.softmax for valid inputs
        return F.softmax(x, dim=dim)

# feedforward network
def create_FCNet(input_dim, num_layers, h_dim, h_fn, o_dim, o_fn, limit=None, dropout=0.0, w_reg=None):
        '''
        GOAL             : Create FC network with different specifications 
        inputs (tensor)  : input tensor
        num_layers       : number of layers in FCNet
        h_dim  (int)     : number of hidden units
        h_fn             : activation function for hidden layers (default: tf.nn.relu)
        o_dim  (int)     : number of output units
        o_fn             : activation function for output layers (defalut: None)
        w_init           : initialization for weight matrix (defalut: Xavier)
        keep_prob        : keep probabilty [0, 1]  (if None, dropout is not employed)
        '''
        # default active functions (hidden: relu, out: None)
        if h_fn is None:
            h_fn = nn.ReLU()
        if o_fn is None:
            o_fn = None
        
        layers = []
        if num_layers == 1:
            fc = nn.Linear(input_dim, o_dim)
            if not limit is None:
                nn.init.uniform_(fc.weight, a=-limit, b=limit)
            else:
                nn.init.xavier_normal_(fc.weight)
            layers.append(fc)
            if not o_fn is None:
                layers.append(o_fn)
            return nn.Sequential(*layers)
        else:
            fc1 = nn.Linear(input_dim, h_dim)
            if not limit is None:
                nn.init.uniform_(fc1.weight, a=-limit, b=limit)
            else:
                nn.init.xavier_normal_(fc1.weight)
            layers.append(fc1)
            layers.append(h_fn)
            if not dropout is None:
                layers.append(dropout)

            for _ in range(1, num_layers-1):
                fc2 = nn.Linear(h_dim, h_dim)
                if not limit is None:
                    nn.init.uniform_(fc2.weight, a=-limit, b=limit)
                else:
                    nn.init.xavier_normal_(fc2.weight)
                layers.append(fc2)
                layers.append(h_fn)
                if not dropout is None:
                    layers.append(dropout)

            fc3 = nn.Linear(h_dim, o_dim)
            if not limit is None:
                nn.init.uniform_(fc3.weight, a=-limit, b=limit)
            else:
                nn.init.xavier_normal_(fc3.weight)
            layers.append(fc3)
            if not o_fn is None:
                layers.append(o_fn)
            return nn.Sequential(*layers)
