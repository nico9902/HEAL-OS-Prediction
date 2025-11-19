import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

# import the required packages
import argparse
import numpy as np
import torch
import torch.optim as optim

from scripts.networks import Model_DeepHit
from utils_eval import c_index
from datasets import CLARO_clinical
import import_data as impt
import utils_model

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str)
parser.add_argument('--out-dir', type=str)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lr_step_size', type=int, default=1000)
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--rs_iteration', type=int, default=20)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--cv', type=int, default=10)
parser.add_argument('--random_search', action='store_true')
parser.add_argument('--attention_mode', type=str, default='hierarchical')

parser.add_argument('--no_rs_alpha', type=float, default=0.001)
parser.add_argument('--no_rs_beta', type=float, default=0.001)
parser.add_argument('--no_rs_gamma', type=float, default=0.001)
parser.add_argument('--no_rs_batch_size', type=int, default=32)
parser.add_argument('--no_rs_h_dim_shared', type=int, default=1)
parser.add_argument('--no_rs_h_dim_CS', type=int, default=1)
parser.add_argument('--no_rs_num_layers_shared', type=int, default=1)
parser.add_argument('--no_rs_num_layers_CS', type=int, default=1)
parser.add_argument('--no_rs_dropout', type=float, default=0.5)
parser.add_argument('--no_rs_active_fn', type=str, default="relu")
parser.add_argument('--no_rs_lr_step_size', type=int, default=1000)
parser.add_argument('--no_rs_lr_gamma', type=float, default=0.1)
parser.add_argument('--no_rs_alpha', type=float, default=0.1)
parser.add_argument('--no_rs_beta', type=float, default=0.1)
parser.add_argument('--no_rs_gamma', type=float, default=0.1)

args = parser.parse_args()

# # make visible only one GPU at the time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # <-- should be the ID of the GPU you want to use

# options
root_dir = args.root_dir              # root directory
out_dir = args.out_dir                # output directory
device = args.device                  # put here "cuda:0" if you want to run on GPU
num_workers = args.num_workers        # how many workers (=threads) for fetching data
pretrained = args.pretrained          # True if the model is pretrained (for testing)
rs_iteration = args.rs_iteration      # random search iteration
random_search = args.random_search    # True if you want to perform random search
attention_mode = args.attention_mode  # which attention mode to use

# trainer options
epochs = args.epochs
learning_rate = args.learning_rate
lr_step_size = args.lr_step_size   # if < epochs, we are using decaying learning rate
lr_gamma = args.lr_gamma
cv = args.cv

# hyperparameters without random search
no_rs_alpha = args.no_rs_alpha
no_rs_beta = args.no_rs_beta
no_rs_gamma = args.no_rs_gamma
no_rs_batch_size = args.no_rs_batch_size
no_rs_h_dim_shared = args.no_rs_h_dim_shared
no_rs_h_dim_CS = args.no_rs_h_dim_CS
no_rs_num_layers_shared = args.no_rs_num_layers_shared
no_rs_num_layers_CS = args.no_rs_num_layers_CS
no_rs_dropout = args.no_rs_dropout
no_rs_active_fn = args.no_rs_active_fn
no_rs_alpha = args.no_rs_alpha
no_rs_beta = args.no_rs_beta
no_rs_gamma = args.no_rs_gamma

if __name__ == "__main__":
    # pretrained model not available --> TRAIN a new one and save it
    if not pretrained:

        preds = []
        all_times = []
        all_labels = []
        # perform 10-fold cross-validation
        for index in range(cv):

            # import datasets
            x_dim, (tr_data, tr_times, tr_labels), (tr_masks1, tr_masks2) = impt.import_dataset_CLINICAL(root_dir, index, step="train")
            _, (te_data, te_times, te_labels), (te_masks1, te_masks2) = impt.import_dataset_CLINICAL(root_dir, index, step="test")

            # get input dimensions
            num_Event = np.shape(tr_masks1)[1]
            num_Category = np.shape(tr_masks1)[2]
            input_dims = { 'x_dim'         : x_dim,
                            'num_Event'     : num_Event,
                            'num_Category'  : num_Category}

            # random search
            if random_search:
                hyperparameters, max_valid, experiment_ID = utils_model.random_search_clinical(tr_data, tr_labels, tr_times, tr_masks1, tr_masks2, input_dims, num_Event, num_Category, rs_iteration, learning_rate, lr_step_size, lr_gamma, epochs, num_workers, device)
                print(f"Best model: {experiment_ID}")
            else:
                no_rs_activation = utils_model.get_activation(no_rs_active_fn)
                hyperparameters = {    'alpha'              : no_rs_alpha,
                                            'beta'               : no_rs_beta,
                                            'gamma'              : no_rs_gamma,
                                            'batch_size'         : no_rs_batch_size,
                                            'learning_rate'      : learning_rate,
                                            'h_dim_shared'       : no_rs_h_dim_shared,
                                            'num_layers_shared'  : no_rs_num_layers_shared,
                                            'h_dim_CS'           : no_rs_h_dim_CS,
                                            'num_layers_CS'      : no_rs_num_layers_CS,
                                            'active_fn'          : no_rs_activation, 
                                            'dropout'            : no_rs_dropout           }   

            # best hyperparameters on validation set
            network_settings = { 'h_dim_shared'       : hyperparameters['h_dim_shared'],
                                 'num_layers_shared'  : hyperparameters['num_layers_shared'],
                                 'h_dim_CS'           : hyperparameters['h_dim_CS'],
                                 'num_layers_CS'      : hyperparameters['num_layers_CS'],
                                 'active_fn'          : hyperparameters['active_fn'], 
                                 'dropout'            : hyperparameters['dropout']}
            alpha = hyperparameters['alpha']
            beta = hyperparameters['beta']
            gamma = hyperparameters['gamma']
            batch_size = hyperparameters['batch_size']

            # create network
            net = Model_DeepHit(input_dims, network_settings, attention_mode=attention_mode)

            if not random_search:
                experiment_ID = "%s_%s_%s_bs(%d)lr(%.4f_%d_%.1f)e(%d)act(%s)xavier(yes)do(%.1f)a(%.1f)b(%.1f)g(%.1f)num_lay_sh(%d)num_lay_CS(%d)hid_sh(%d)hid_CS(%d)fold(%d)BN" % (type(net).__name__, type("DeepHit_Loss").__name__, "Adam",
                    no_rs_batch_size, learning_rate, lr_step_size, lr_gamma, epochs, type(no_rs_activation).__name__, no_rs_dropout, no_rs_alpha, no_rs_beta, no_rs_gamma, no_rs_num_layers_shared, no_rs_num_layers_CS, no_rs_h_dim_shared, no_rs_h_dim_CS, index+1)

            # training dataset
            train_dataset = CLARO_clinical(tr_data, tr_times, tr_labels, tr_masks1, tr_masks2)
            # testing dataset
            test_dataset = CLARO_clinical(te_data, te_times, te_labels, te_masks1, te_masks2)

            # create Adam optimizer
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

            # create learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

            # create dataloaders
            dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True) 
            dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=num_workers, pin_memory=True)
            
            # move net to device
            net.to(device)
            
            # train model
            print("Training model...")
            net = utils_model.train_model(train_dataset, dataloader_train, test_dataset, dataloader_test, net, device, optimizer, scheduler, alpha, beta, gamma, num_Event, num_Category, epochs, experiment_ID, out_dir, random_search)

            # test
            predictions = utils_model.test(test_dataset, dataloader_test, net, device)
            te_result1 = np.zeros([num_Event])
            for k in range(num_Event):
                te_result1[k] = c_index(predictions, test_dataset.times, (test_dataset.events[:,0] == k+1).astype(int), k)
            tmp_test = np.mean(te_result1)
            print("c-index on test: ", tmp_test)

            # accomulate predictions
            preds.append(predictions)
            all_times.append(test_dataset.times)
            all_labels.append(test_dataset.events)
        
        # concatenate predictions over folds
        preds = np.concatenate(preds, axis=0)
        all_times = np.concatenate(all_times, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        te_result1 = np.zeros([num_Event])
        for k in range(num_Event):
            te_result1[k] = c_index(preds, all_times, (all_labels[:,k] == k+1).astype(int), k)
        tmp_test = np.mean(te_result1)
        print("test c-td index on 10 folds ", tmp_test)

    else:
        
        # pretrained model available --> TEST the model
        # parsing the model pathss
        parser.add_argument('--model_fold_0', type=str)
        parser.add_argument('--model_fold_1', type=str)
        parser.add_argument('--model_fold_2', type=str)
        parser.add_argument('--model_fold_3', type=str)
        parser.add_argument('--model_fold_4', type=str)
        parser.add_argument('--model_fold_5', type=str)
        parser.add_argument('--model_fold_6', type=str)
        parser.add_argument('--model_fold_7', type=str)
        parser.add_argument('--model_fold_8', type=str)
        parser.add_argument('--model_fold_9', type=str)

        model_fold_0 = args.model_fold_0
        model_fold_1 = args.model_fold_1
        model_fold_2 = args.model_fold_2
        model_fold_3 = args.model_fold_3
        model_fold_4 = args.model_fold_4
        model_fold_5 = args.model_fold_5
        model_fold_6 = args.model_fold_6
        model_fold_7 = args.model_fold_7
        model_fold_8 = args.model_fold_8
        model_fold_9 = args.model_fold_9

        # models configuration
        experiments = [model_fold_0, model_fold_1, model_fold_2, model_fold_3, model_fold_4, model_fold_5, model_fold_6, model_fold_7, model_fold_8, model_fold_9]
        
        # initialize lists
        preds = []
        all_times = []
        all_labels = []

        # perform 10-fold cross-validation
        for index in range(10):
            print("fold number: ", index+1)
            experiment_ID = experiments[index]

            # import datasets
            x_dim, (te_data, te_times, te_labels), (te_masks1, te_masks2) = impt.import_dataset_CLINICAL(root_dir, index, step="test")

            # get input dimensions
            num_Event = np.shape(te_masks1)[1]
            num_Category = np.shape(te_masks1)[2]
            input_dims = { 'x_dim'         : x_dim,
                            'num_Event'     : num_Event,
                            'num_Category'  : num_Category}
            print(f"input_dims: {input_dims}")

            # testing dataset
            test_dataset = CLARO_clinical(te_data, te_times, te_labels, te_masks1, te_masks2)
            dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=num_workers, pin_memory=True)

            # load pretrained model
            checkpoint = torch.load(experiment_ID + ".tar", map_location=lambda storage, loc: storage)
            net = checkpoint['net']
            print ("Loaded pretrained model\n...trained for %d epochs\n...reached c-index %.2f" % (checkpoint['epoch'], checkpoint['c-index']))
        
            # move net to device
            net.to(device)

            # test
            predictions = utils_model.test(test_dataset, dataloader_test, net, device)
        
            # accomulate predictions
            preds.append(predictions)
            all_times.append(test_dataset.times)
            all_labels.append(test_dataset.events)
        
        # concatenate predictions over folds
        preds = np.concatenate(preds, axis=0)
        all_times = np.concatenate(all_times, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        te_result1 = np.zeros([num_Event])

        for k in range(num_Event):
            te_result1[k] = c_index(preds, all_times, (all_labels[:,k] == k+1).astype(int), k)

        tmp_test = np.mean(te_result1)

        print("test c-td index on 10 folds ", tmp_test)
