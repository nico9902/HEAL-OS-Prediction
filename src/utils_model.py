import sys;
sys.path.extend(["./"])

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from datasets import CLARO_att, CLARO_clinical
from scripts.networks import Model_DeepHit
from losses import DeepHit_loss
from utils_eval import c_index
from utils_data import collate_batch

# get activation function
def get_activation(fn):
    if fn == "relu":
        return nn.ReLU()
    elif fn == "tanh":
        return nn.Tanh()
    elif fn == "elu":
        return nn.ELU()
    elif fn == "sigmoid":
        return nn.Sigmoid()
    elif fn == "selu":
        return nn.SELU()
    else:
        raise ValueError("Invalid activation function")

# define train function (1 epoch)
# returns average loss and accuracy
def train(dataset, dataloader, net, device, optimizer, scheduler, alpha, beta, gamma, num_Event, num_Category):

    # switch to train mode
    net.train()

    # reset performance measures
    loss_sum = 0.0

    # 1 epoch = 1 complete loop over the dataset
    for batch in dataloader:
        
        # get data from dataloader
        inputs, times, targets, mask1s, mask2s = batch

        # move data to device
        inputs, times, targets, mask1s, mask2s = inputs.to(device, non_blocking=True), times.to(device, non_blocking=True), targets.to(device, non_blocking=True), mask1s.to(device, non_blocking=True), mask2s.to(device, non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = net(inputs)

        # lambda1 = 1e-4
        # lambda2 = 1e-4
        # all_shared_net_parameters = torch.cat([x.view(-1) for x in net.shared_net.parameters()])
        # all_cs_net_parameters = torch.cat([x.view(-1) for x in net.cs_net.parameters()])
        # all_out_fc_parameters = torch.cat([x.view(-1) for x in net.out_fc.parameters()])
        # l1_regularization = lambda1 * torch.norm(all_shared_net_parameters, 1) + lambda1 * torch.norm(all_cs_net_parameters, 1)
        # l2_regularization = lambda2 * torch.norm(all_out_fc_parameters, 2).pow(2)

        # calculate loss
        loss = DeepHit_loss(outputs, alpha, beta, gamma, num_Event, num_Category, mask1s, mask2s, times, targets) #+ l1_regularization + l2_regularization

        # loss gradient backpropagation
        loss.backward()

        # net parameters update
        optimizer.step()

        # accumulate loss
        loss_sum += loss.item()

    # step learning rate scheduler
    scheduler.step()

    # return average loss and accuracy
    return loss_sum / len(dataloader) 

# define test function
# returns predictions
def test(dataset, dataloader, net, device):

    # switch to test mode
    net.eval()  

    # initialize predictions
    predictions = []

    # do not accumulate gradients (faster)
    with torch.no_grad():

        # test all batches
        for batch in dataloader:
            # get data from dataloader [ignore labels/targets as they are not used in test mode]
            inputs = batch[0]

            # move data to device
            inputs = inputs.to(device, non_blocking=True)

            # forward pass
            outputs = net(inputs)

            # store predictions
            outputs = outputs.cpu().detach().numpy()
            for output in outputs:
                predictions.append(output)

    return np.asarray(predictions)

# train model from scratch
def train_model(train_dataset, dataloader_train, test_dataset, dataloader_test, net, device, optimizer, scheduler, alpha, beta, gamma, num_Event, num_Category, epochs, experiment_ID, out_dir, random_search=False):
    losses = []
    ticks = []
    for epoch in range(1, epochs+1):

        # measure time elapsed
        t0 = time.time()

        # train
        avg_loss = train(train_dataset, dataloader_train, net, device, optimizer, scheduler, alpha, beta, gamma, num_Event, num_Category)

        # test
        predictions = test(test_dataset, dataloader_test, net, device)
        te_result1 = np.zeros([num_Event])
        for k in range(num_Event):
            te_result1[k] = c_index(predictions, test_dataset.times, (test_dataset.events[:,0] == k+1).astype(int), k)
        tmp_test = np.mean(te_result1)

        # update performance history
        losses.append(avg_loss)
        ticks.append(epoch)

        fig, ax1 = plt.subplots(figsize=(12, 8), num=1)
        ax1.set_xticks(np.arange(0, epochs+1, step=epochs/10.0))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel("DeepHit_Loss", color='blue')
        ax1.set_ylim(1, 200)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_yscale('log')
        ax1.plot(ticks, losses, 'b-', linewidth=1.0, aa=True, 
                label='Training (best at ep. %d)' % ticks[np.argmin(losses)])
        ax1.legend(loc="lower left")
        plt.xlim(0, epochs+1)
        plt.draw()
        plt.savefig(str(out_dir) + "/" + experiment_ID + ".png", dpi=300)
        fig.clear()

        print ("\nEpoch %d\n"
            "...TIME: %.1f seconds\n"
            "...loss: %g (best %g at epoch %d)\n" % (
            epoch,
            time.time()-t0,
            avg_loss, min(losses), ticks[np.argmin(losses)]))
        
        print("test c-td index ", tmp_test)
    
    if random_search:
        # save model checkpoint
        torch.save({
            'net': net,
            'epoch': epoch
        }, str(out_dir) + "/" + experiment_ID + ".tar")

    return net

def random_search(train_embeddings, train_label, train_time, train_mask1, train_mask2, input_dims, num_Event, num_Category, index, data_mode, aggregation, rs_iterations, learning_rate, lr_step_size, lr_gamma, epochs, num_workers, device, attention_mode):
    # network hyper-parameters
    if aggregation:
        SET_BATCH_SIZE    = [16, 32, 64]            #batch_size 
    else:
        SET_BATCH_SIZE    = [8, 16]                 #batch_size 
    SET_LAYERS            = [1, 2, 3, 5]            #number of layers
    SET_NODES             = [20, 50, 100, 200]      #number of nodes
    SET_DROPOUT           = [0.2, 0.3, 0.4]         #dropout
    #SET_LEARNING_RATE    = [1e-4, 1e-3]            #learning rate
    SET_ACTIVATION_FN     = ['selu', 'relu']        #non-linear activation functions
    SET_ALPHA             = [0.1, 0.5, 1.0, 3.0]    #alpha values -> log-likelihood loss 
    SET_BETA              = [0.1, 0.5, 1.0, 3.0]    #beta values  -> ranking loss
    SET_GAMMA             = [0.1, 0.5, 1.0, 3.0]    #gamma values -> calibration loss

    max_experiment_ID = ""
    best_hyperparameters = {}
    max_valid = 0.0
    previous_configs = []
    for rs_index in range(rs_iterations):
        print("rs_iteration: ", rs_index+1)

        while True:
            # hyper-parameters
            batch_size = SET_BATCH_SIZE[np.random.randint(len(SET_BATCH_SIZE))]
            #learning_rate = SET_LEARNING_RATE[np.random.randint(len(SET_LEARNING_RATE))]
            dropout = SET_DROPOUT[np.random.randint(len(SET_DROPOUT))]
            alpha = SET_ALPHA[np.random.randint(len(SET_ALPHA))]
            beta = SET_BETA[np.random.randint(len(SET_BETA))]
            gamma = SET_GAMMA[np.random.randint(len(SET_GAMMA))]

            # Network hyper-parameters
            h_dim_shared = SET_NODES[np.random.randint(len(SET_NODES))]
            num_layers_shared = SET_LAYERS[np.random.randint(len(SET_LAYERS))]
            h_dim_CS = SET_NODES[np.random.randint(len(SET_NODES))]
            num_layers_CS = SET_LAYERS[np.random.randint(len(SET_LAYERS))]
            activation = get_activation(SET_ACTIVATION_FN[np.random.randint(len(SET_ACTIVATION_FN))])
            
            config_str = "bs(%d)lr(%.4f_%d_%.1f)e(%d)act(%s)xavier(yes)do(%.1f)a(%.1f)b(%.1f)g(%.1f)version(%s)num_lay_sh(%d)num_lay_CS(%d)hid_sh(%d)hid_CS(%d)fold(%d)BN" % (batch_size,
                            learning_rate, lr_step_size, lr_gamma, epochs, type(activation).__name__, dropout, alpha, beta, gamma, data_mode, num_layers_shared, num_layers_CS, h_dim_shared, h_dim_CS, index+1)

            if config_str not in previous_configs:
                break
        
        previous_configs.append(config_str)
        print("config_str: ", config_str)
        network_settings = { 'h_dim_shared'       : h_dim_shared,
                            'num_layers_shared'  : num_layers_shared,
                            'h_dim_CS'           : h_dim_CS,
                            'num_layers_CS'      : num_layers_CS,
                            'active_fn'          : activation, 
                            'dropout'            : dropout           }
        # hyerparameters
        hyperparameters = { 'alpha'              : alpha,
                            'beta'               : beta,
                            'gamma'              : gamma,
                            'batch_size'         : batch_size,
                            'learning_rate'      : learning_rate,
                            'h_dim_shared'       : h_dim_shared,
                            'num_layers_shared'  : num_layers_shared,
                            'h_dim_CS'           : h_dim_CS,
                            'num_layers_CS'      : num_layers_CS,
                            'active_fn'          : activation, 
                            'dropout'            : dropout           }

        k = 10
        skf = StratifiedKFold(n_splits=k)
        val_preds = []
        val_all_times = []
        val_all_labels = []
        for training, valid in skf.split(train_embeddings, train_label):

            # training dataset
            tr_data = train_embeddings[training]
            tr_label = train_label[training]
            tr_time = train_time[training]
            tr_mask1 = train_mask1[training]
            tr_mask2 = train_mask2[training]
            train_dataset = CLARO_att(tr_data, tr_time, tr_label, tr_mask1, tr_mask2)

            # validation dataset
            valid_data = train_embeddings[valid]
            valid_label = train_label[valid]
            valid_time = train_time[valid]
            valid_mask1 = train_mask1[valid]
            valid_mask2 = train_mask2[valid]
            val_dataset = CLARO_att(valid_data, valid_time, valid_label, valid_mask1, valid_mask2)

            # create network
            net = Model_DeepHit(input_dims=input_dims, network_settings=network_settings, attention_mode=attention_mode)
            # create dataparallel
            # if torch.cuda.device_count() > 1:
            #     net = nn.DataParallel(net, [0,1])
                #print("number of GPUs: ", torch.cuda.device_count())

            # create Adam optimizer
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

            # create learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

            # create data loaders
            # NOTE 1: shuffle helps training
            # NOTE 2: in test mode, batch size can be as high as the GPU can handle (faster, but requires more GPU RAM)
            dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch(aggregation), drop_last=True) 
            dataloader_valid = torch.utils.data.DataLoader(val_dataset, batch_size=512, num_workers=num_workers, collate_fn=collate_batch(aggregation), pin_memory=True) 
    
            # move net to device
            net.to(device)

            # experiment ID
            experiment_ID = "%s_%s_%s_bs(%d)lr(%.4f_%d_%.1f)e(%d)act(%s)xavier(yes)do(%.1f)a(%.1f)b(%.1f)g(%.1f)version(%s)num_lay_sh(%d)num_lay_CS(%d)hid_sh(%d)hid_CS(%d)fold(%d)BN" % (type(net).__name__, type("DeepHit_Loss").__name__, "Adam",
                    batch_size, learning_rate, lr_step_size, lr_gamma, epochs, type(activation).__name__, dropout, alpha, beta, gamma, data_mode, num_layers_shared, num_layers_CS, h_dim_shared, h_dim_CS, index+1)
            
            # start training
            for epoch in range(1, epochs+1):
                # # measure time elapsed
                # t0 = time.time()

                # train
                avg_loss = train(train_dataset, dataloader_train, net, device, optimizer, scheduler, alpha, beta, gamma, num_Event, num_Category)
                # print("time for 1 epoch: ", float(time.time()-t0))

            # test on validation set
            predictions = test(val_dataset, dataloader_valid, net, device)

            val_preds.append(predictions)
            val_all_times.append(val_dataset.times)
            val_all_labels.append(val_dataset.events)

        # test on validation set
        val_preds = np.concatenate(val_preds, axis=0)
        val_all_times = np.concatenate(val_all_times, axis=0)
        val_all_labels = np.concatenate(val_all_labels, axis=0)

        va_result1 = np.zeros(num_Event)
        for k in range(num_Event):
            va_result1[k] = c_index(val_preds, val_all_times, (val_all_labels[:,k] == k+1).astype(int), k)
        tmp_valid = np.mean(va_result1)

        # maximum search
        if tmp_valid > max_valid:
            max_valid = tmp_valid
            max_experiment_ID = experiment_ID
            best_hyperparameters = hyperparameters

    print("experiment ", max_experiment_ID)
    print("valid c-td index ", max_valid) 

    return best_hyperparameters, max_valid, max_experiment_ID
    

def random_search_clinical(train_data, train_label, train_time, train_mask1, train_mask2, input_dims, num_Event, num_Category, rs_iteration, learning_rate, lr_step_size, lr_gamma, epochs, num_workers, device, attention_mode):
    # Network hyper-parameters
    SET_BATCH_SIZE    = [16, 32, 64] #batch_size
    
    SET_LAYERS        = [1, 2, 3, 5] #number of layers
    SET_NODES         = [50, 100, 200, 300] #number of nodes

    SET_DROPOUT       = [0.2, 0.3, 0.4] #dropout

    SET_ACTIVATION_FN = ['relu', 'tanh'] #non-linear activation functions

    SET_ALPHA         = [0.1, 0.5, 1.0, 3.0] #alpha values -> log-likelihood loss 
    SET_BETA          = [0.1, 0.5, 1.0, 3.0] #beta values -> ranking loss
    SET_GAMMA         = [0.0, 0.5, 1.0, 3.0] #gamma values -> calibration loss

    # rs 
    max_valid = 0.0
    max_experiment_ID = ""
    best_hyperparameters = {}
    previous_configs = []
    for rs_index in range(rs_iteration):
        print("rs_iteration ", rs_index+1)

        while True:
            # hyper-parameters
            batch_size = SET_BATCH_SIZE[np.random.randint(len(SET_BATCH_SIZE))]
            #learning_rate = SET_LEARNING_RATE[np.random.randint(len(SET_LEARNING_RATE))]
            dropout = SET_DROPOUT[np.random.randint(len(SET_DROPOUT))]
            alpha = SET_ALPHA[np.random.randint(len(SET_ALPHA))]
            beta = SET_BETA[np.random.randint(len(SET_BETA))]
            gamma = SET_GAMMA[np.random.randint(len(SET_GAMMA))]

            # Network hyper-parameters
            h_dim_shared = SET_NODES[np.random.randint(len(SET_NODES))]
            num_layers_shared = SET_LAYERS[np.random.randint(len(SET_LAYERS))]
            h_dim_CS = SET_NODES[np.random.randint(len(SET_NODES))]
            num_layers_CS = SET_LAYERS[np.random.randint(len(SET_LAYERS))]
            activation = get_activation(SET_ACTIVATION_FN[np.random.randint(len(SET_ACTIVATION_FN))])
            
            config_str = "bs(%d)lr(%.4f_%d_%.1f)e(%d)act(%s)xavier(yes)do(%.1f)a(%.1f)b(%.1f)g(%.1f)num_lay_sh(%d)num_lay_CS(%d)hid_sh(%d)hid_CS(%d)BN" % (batch_size,
                                learning_rate, lr_step_size, lr_gamma, epochs, type(activation).__name__, dropout, alpha, beta, gamma, num_layers_shared, num_layers_CS, h_dim_shared, h_dim_CS)

            if config_str not in previous_configs:
                break
        
        previous_configs.append(config_str)

        network_settings = { 'h_dim_shared'       : h_dim_shared,
                                'num_layers_shared'  : num_layers_shared,
                                'h_dim_CS'           : h_dim_CS,
                                'num_layers_CS'      : num_layers_CS,
                                'active_fn'          : activation, 
                                'dropout'            : dropout           }
        
        # hyerparameters
        hyperparameters = { 'alpha'              : alpha,
                            'beta'               : beta,
                            'gamma'              : gamma,
                            'batch_size'         : batch_size,
                            'learning_rate'      : learning_rate,
                            'h_dim_shared'       : h_dim_shared,
                            'num_layers_shared'  : num_layers_shared,
                            'h_dim_CS'           : h_dim_CS,
                            'num_layers_CS'      : num_layers_CS,
                            'active_fn'          : activation, 
                            'dropout'            : dropout           }
        
        k = 10
        skf = StratifiedKFold(n_splits=k)
        val_preds = []
        val_all_times = []
        val_all_labels = []
        for training, valid in skf.split(train_data, train_label):
            
            # training dataset
            tr_data = train_data[training]
            tr_label = train_label[training]
            tr_time = train_time[training]
            tr_mask1 = train_mask1[training]
            tr_mask2 = train_mask2[training]
            train_dataset = CLARO_clinical(tr_data, tr_time, tr_label, tr_mask1, tr_mask2)
            # validation dataset
            valid_data = train_data[valid]
            valid_label = train_label[valid]
            valid_time = train_time[valid]
            valid_mask1 = train_mask1[valid]
            valid_mask2 = train_mask2[valid]
            val_dataset = CLARO_clinical(valid_data, valid_time, valid_label, valid_mask1, valid_mask2)

            # create network
            net = Model_DeepHit(input_dims, network_settings, attention_mode)

            # create Adam optimizer
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

            # create learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

            # create data loaders
            # NOTE 1: shuffle helps training
            # NOTE 2: in test mode, batch size can be as high as the GPU can handle (faster, but requires more GPU RAM)
            dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True) 
            dataloader_valid = torch.utils.data.DataLoader(val_dataset, batch_size=512, num_workers=num_workers, pin_memory=True) 
        
            # move net to device
            net.to(device)

            # experiment ID
            experiment_ID = "%s_%s_%s_bs(%d)lr(%.4f_%d_%.1f)e(%d)act(%s)xavier(yes)do(%.1f)a(%.1f)b(%.1f)g(%.1f)num_lay_sh(%d)num_lay_CS(%d)hid_sh(%d)hid_CS(%d)BN" % (type(net).__name__, type("DeepHit_Loss").__name__, "Adam",
                    batch_size, learning_rate, lr_step_size, lr_gamma, epochs, type(activation).__name__, dropout, alpha, beta, gamma, num_layers_shared, num_layers_CS, h_dim_shared, h_dim_CS)


            for epoch in range(1, epochs+1):
                # # measure time elapsed
                # t0 = time.time()

                # train
                avg_loss = train(train_dataset, dataloader_train, net, device, optimizer, scheduler, alpha, beta, gamma, num_Event, num_Category)
            
                predictions = test(val_dataset, dataloader_valid, net, device)
                va_result1 = np.zeros(num_Event)
                for k in range(num_Event):
                    va_result1[k] = c_index(predictions, val_dataset.times, (val_dataset.events[:,0] == k+1).astype(int), k)
                tmp_valid = np.mean(va_result1)
                
    
            # test on validation set
            predictions = test(val_dataset, dataloader_valid, net, device)
            # va_result1 = np.zeros(num_Event)
            # for k in range(num_Event):
            #     va_result1[k] = c_index(predictions, val_dataset.times, (val_dataset.events[:,0] == k+1).astype(int), k)
            # tmp_valid = np.mean(va_result1)

            val_preds.append(predictions)
            val_all_times.append(val_dataset.times)
            val_all_labels.append(val_dataset.events)

        # test on validation set
        val_preds = np.concatenate(val_preds, axis=0)
        val_all_times = np.concatenate(val_all_times, axis=0)
        val_all_labels = np.concatenate(val_all_labels, axis=0)

        va_result1 = np.zeros(num_Event)
        for k in range(num_Event):
            va_result1[k] = c_index(val_preds, val_all_times, (val_all_labels[:,k] == k+1).astype(int), k)
        tmp_valid = np.mean(va_result1)

        # maximum search
        if tmp_valid > max_valid:
            max_valid = tmp_valid
            max_experiment_ID = experiment_ID
            best_hyperparameters = hyperparameters

    print("experiment ", max_experiment_ID)
    print("valid c-td index ", max_valid)  
    print(best_hyperparameters)

    return best_hyperparameters, max_valid, max_experiment_ID   
