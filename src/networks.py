'''
This declare DeepHit architecture:

INPUTS:
    - input_dims: dictionary of dimension information
        > x_dim: dimension of features
        > num_Event: number of competing events (this does not include censoring label)
        > num_Category: dimension of time horizon of interest, i.e., |T| where T = {0, 1, ..., T_max-1}
                      : this is equivalent to the output dimension
    - network_settings:
        > h_dim_shared & num_layers_shared: number of nodes and number of fully-connected layers for the shared subnetwork
        > h_dim_CS & num_layers_CS: number of nodes and number of fully-connected layers for the cause-specific subnetworks
        > active_fn: 'relu', 'elu', 'tanh'
        > initial_W: Xavier initialization is used as a baseline

LOSS FUNCTIONS:
    - 1. loglikelihood (this includes log-likelihood of subjects who are censored)
    - 2. rankding loss (this is calculated only for acceptable pairs; see the paper for the definition)
    - 3. calibration loss (this is to reduce the calibration loss; this is not included in the paper version)
'''
import torch
import torch.nn as nn
import utils_network as utils

class Attention(nn.Module):
    """
    Attention mechanism
        Calculate attention weights and apply them to the input tensor
        Input:
            - inputs: input tensor
        Output:
            - weighted_sum: weighted sum of input tensor
            - attention_weights: attention weights
    """
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, 1)
        self.entity_linear = nn.Linear(input_size+25, 1)

    def forward(self, inputs, entities=None): 
        # inputs: (batch_size, seq_len, input_size)
        # handle 4D input tensor (batch_size, num_Event, seq_len, input_size) for hierarchical attention
        num_dims = len(inputs.size())
        if num_dims == 4:
            dim0 = inputs.size(dim=0)
            dim1 = inputs.size(dim=1)
            dim2 = inputs.size(dim=2)
            dim3 = inputs.size(dim=3)
            inputs = inputs.view(dim0*dim1, dim2, dim3)
            if entities != None:
                entities = entities.view(dim0*dim1, dim2, 25)
        elif entities != None and len(entities.size()) == 4:
            entities = entities.sum(dim=2)

        # create a mask to ignore padded elements
        mask = (inputs.sum(dim=2) != 0).float()

        # get batch size and sequence length from inputs
        batch_size, seq_len, _ = inputs.size()
    
        # calculate attention scores
        if entities != None:
            att_inputs = torch.cat([inputs, entities], dim=2)
            scores = self.entity_linear(att_inputs).view(batch_size, seq_len)
        else:
            scores = self.linear(inputs).view(batch_size, seq_len)

        # apply softmax to get attention weights
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = utils.safe_softmax(scores, dim=1)
        #attention_weights = F.softmax(scores, dim=1)

        # compute the weighted mean
        attention_weights = attention_weights.unsqueeze(2)
        weighted_sum = torch.bmm(inputs.transpose(1, 2), attention_weights).squeeze(2)

        # in case of hierarchical attention
        if num_dims == 4:
            weighted_sum = weighted_sum.view(dim0, dim1, dim3)

        return weighted_sum, attention_weights

# DeepHit architecture
class Model_DeepHit(nn.Module):
    """
    DeepHit architecture
        Input:
            - input_dims: dictionary of dimension information
            - network_settings: dictionary of network hyper-parameters
            - attention_mode: 'single', 'hierarchical', 'hierarchical_sharing'
        Output:
            - out: predicted probability of event occurrence
    """
    def __init__(self, input_dims, network_settings, attention_mode='single'):
        super(Model_DeepHit, self).__init__()

        # input dimensions
        self.x_dim              = input_dims['x_dim']
        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']

        # network hyper-parameters
        self.h_dim_shared       = network_settings['h_dim_shared']
        self.h_dim_CS           = network_settings['h_dim_CS']
        self.num_layers_shared  = network_settings['num_layers_shared']
        self.num_layers_CS      = network_settings['num_layers_CS']
        self.active_fn          = network_settings['active_fn']
        self.p_dropout          = network_settings['dropout']

        # xavier initialization
        self.limit = None
        # dropout
        self.dropout = nn.Dropout(p=self.p_dropout)

        # attention mechanism
        self.attention_mode = attention_mode   # single, hierarchical, hierarchical_sharing
        if self.attention_mode == 'single':
            self.seq_attention_layer = Attention(input_size=self.x_dim)
        elif self.attention_mode == 'hierarchical':
            self.seq_attention_layer = Attention(input_size=self.x_dim)
            self.tok_attention_layer = Attention(input_size=self.x_dim)
        elif self.attention_mode == 'hierarchical_sharing':
            self.seq_attention_layer = Attention(input_size=self.x_dim)
        else:
            raise ValueError('Invalid attention mode')

        # shared network
        self.shared_net = utils.create_FCNet(self.x_dim, self.num_layers_shared, self.h_dim_shared, self.active_fn, self.h_dim_shared, self.active_fn, self.limit, self.dropout)
        
        # cause-specific network
        self.cs_net = utils.create_FCNet(self.x_dim + self.h_dim_shared, (self.num_layers_CS), self.h_dim_CS, self.active_fn, self.h_dim_CS, self.active_fn, self.limit, self.dropout)

        # output layer (fc + softmax)
        self.out_fc = nn.Linear(self.num_Event*self.h_dim_CS, self.num_Event*self.num_Category)
        nn.init.xavier_normal_(self.out_fc.weight)
        self.m = nn.Softmax(dim=1)

    def forward(self, x):
        # attention mechanism (single or hierarchical)
        # x: (batch_size, num_Event, seq_len, x_dim) for hierarchical attention
        # x: (batch_size, seq_len, x_dim) for single attention
        if self.attention_mode == 'single':
            x, _ = self.seq_attention_layer(x)
        elif (self.attention_mode == 'hierarchical' or self.seq_attention_layer == 'hierarchical_sharing'):
            if self.attention_mode == 'hierarchical_sharing':   # sharing attention weights for two layers
                x, _ = self.seq_attention_layer(x)
                x, _ = self.seq_attention_layer(x)
            else:
                x, _ = self.tok_attention_layer(x)
                x, _ = self.seq_attention_layer(x)
        
        # shared network
        shared_out = self.shared_net(x)

        # residual connection
        shared_out = torch.cat([x, shared_out], dim=1)

        # (num_layers_CS) layers for cause-specific (num_Event subNets)
        out = []
        for _ in range(self.num_Event):
            cs_out = self.cs_net(shared_out)
            out.append(cs_out)
        out = torch.stack(out, dim=1) # stack referenced on subject
        out = torch.reshape(out, (-1, self.num_Event*self.h_dim_CS))
        out = self.dropout(out)

        # output layer
        out = self.out_fc(out)
        out = self.m(out)
        out = torch.reshape(out, (-1, self.num_Event, self.num_Category))

        return out

    # get attention weights for visualization
    def get_attention_weights(self, x):
        # attention mechanism (single or hierarchical)
        # x: (batch_size, num_Event, seq_len, x_dim) for hierarchical attention
        # x: (batch_size, seq_len, x_dim) for single attention
        if self.attention_mode == 'single':
            x, att_weight1 = self.seq_attention_layer(x)
            return att_weight1, None
        elif (self.attention_mode == 'hierarchical' or self.seq_attention_layer == 'hierarchical_sharing'):
            if self.attention_mode == 'hierarchical_sharing':
                x, att_weight2 = self.seq_attention_layer(x)
                x, att_weight1 = self.seq_attention_layer(x)
            else:
                x, att_weight2 = self.tok_attention_layer(x)
                x, att_weight1 = self.seq_attention_layer(x)
            return att_weight1, att_weight2
