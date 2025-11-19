import sys; 
sys.path.extend(["./"])

# import the required packages
import os
import argparse
import numpy as np

import torch
import torch.optim as optim

from scripts.networks import Model_DeepHit
from datasets import CLARO_att
from utils_eval import c_index
from utils_data import collate_batch
import import_data as impt
import utils_model as utils_model

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu:0')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--data_mode', type=str, default='CLAROv0')
parser.add_argument('--embedding_size', type=int, default=768)
parser.add_argument('--embedding_path', type=str, default='./raw_data/embeddings')
parser.add_argument('--prediction_path', type=str, default='./raw_data/predictions')
parser.add_argument('--label_path', type=str, default='./')
parser.add_argument('--kfold_path', type=str, default='./raw_data/k_fold_doccano')
parser.add_argument('--survival_file_path', type=str, default='survival_analysis.xslx')
parser.add_argument('--out_dir', type=str, default='run1')
parser.add_argument('--with_regex', action='store_true')
parser.add_argument('--rs_iteration', type=int, default=20)
parser.add_argument('--aggregation', action='store_true')
parser.add_argument('--end_fold', type=int, default=10)
parser.add_argument('--random_search', action='store_true')
parser.add_argument('--reg_features', action='store_true')
parser.add_argument('--attention_mode', type=str, default='hierarchical')
parser.add_argument('--pretrained', action='store_true')

parser.add_argument('--no_rs_learning_rate', type=float, default=0.001)
parser.add_argument('--no_rs_alpha', type=float, default=0.1)
parser.add_argument('--no_rs_beta', type=float, default=0.1)
parser.add_argument('--no_rs_gamma', type=float, default=0.1)
parser.add_argument('--no_rs_batch_size', type=int, default=32)
parser.add_argument('--no_rs_h_dim_shared', type=int, default=1)
parser.add_argument('--no_rs_h_dim_CS', type=int, default=1)
parser.add_argument('--no_rs_num_layers_shared', type=int, default=1)
parser.add_argument('--no_rs_num_layers_CS', type=int, default=1)
parser.add_argument('--no_rs_dropout', type=float, default=0.5)
parser.add_argument('--no_rs_active_fn', type=str, default="relu")
parser.add_argument('--no_rs_lr_step_size', type=int, default=1000)
parser.add_argument('--no_rs_lr_gamma', type=float, default=0.1)

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lr_step_size', type=int, default=1000)
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--cv', type=int, default=10)

args = parser.parse_args()

# make visible only one GPU at the time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # <-- should be the ID of the GPU you want to use

# options
device = args.device                                    # put here "cuda:0" if you want to run on GPU, otherwise "cpu"
num_workers = args.num_workers                          # how many workers (=threads) for fetching data
data_mode = args.data_mode                              # which features are used
embedding_size = args.embedding_size                    # size of the embeddings
embedding_path = args.embedding_path                    # path to the embeddings
prediction_path = args.prediction_path                  # path to the prediction file
label_path = args.label_path                            # path to the label file
kfold_path = args.kfold_path                            # path to the kfold file
survival_file_path = args.survival_file_path            # path to the survival file
out_dir = args.out_dir                                  # where to save the model
with_regex = args.with_regex                            # True if we consider also features extracted with regular expressions
rs_iteration = args.rs_iteration                        # random search iteration
aggregation = args.aggregation                          # if we want to aggregate tokens in a sentence
end_fold = args.end_fold                                # ending fold
random_search = args.random_search                      # whether apply a random grid search before training 
reg_features = args.reg_features                        # whether apply concatenation with regular expression features
attention_mode = args.attention_mode                    # which attention mode to use
pretrained = args.pretrained                            # True if the model is pretrained (for testing)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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

# network trainer options
epochs = args.epochs
learning_rate = args.learning_rate
lr_step_size = args.lr_step_size   # if < epochs, we are using decaying learning rate
lr_gamma = args.lr_gamma
cv = args.cv

if __name__ == "__main__":
    # pretrained model not available --> TRAIN a new one and save it
    if not pretrained:
        
        # initialize lists
        preds = []
        all_times = []
        all_labels = []

        # perform 10-fold cross-validation
        for index in range(cv):
            train_embeddings, train_time, train_label, train_mask1, train_mask2, test_embeddings, test_time, test_label, test_mask1, test_mask2, num_Category, num_Event = impt.import_CLARO_sequence_embeddings(survival_file=survival_file_path, embedding_path=embedding_path, prediction_path=prediction_path, label_path=label_path, fold=index+1, kfold_path=kfold_path, data_mode=data_mode, aggregation=aggregation)         

            # input dimension
            x_dim = embedding_size       
            input_dims = { 'x_dim'      : x_dim,
                        'num_Event'     : num_Event,
                        'num_Category'  : num_Category}

            # random search (until the end fold)
            if random_search and index < end_fold:
                hyperparameters, max_valid, experiment_ID = utils_model.random_search(train_embeddings, train_label, train_time, train_mask1, train_mask2, input_dims, num_Event, num_Category, index, data_mode, aggregation, rs_iteration, learning_rate, lr_step_size, lr_gamma, epochs, num_workers, device, attention_mode)
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

            # training dataset
            train_dataset = CLARO_att(train_embeddings, train_time, train_label, train_mask1, train_mask2)
            # testing dataset
            test_dataset = CLARO_att(test_embeddings, test_time, test_label, test_mask1, test_mask2) 

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
            net = Model_DeepHit(input_dims=input_dims, network_settings=network_settings, attention_mode=attention_mode)
            # # create dataparallel
            # if torch.cuda.device_count() > 1:
            #     net = nn.DataParallel(net, [0,1])

            if not random_search:
                experiment_ID = "%s_%s_%s_bs(%d)lr(%.4f_%d_%.1f)e(%d)act(%s)xavier(yes)do(%.1f)a(%.1f)b(%.1f)g(%.1f)version(%s)num_lay_sh(%d)num_lay_CS(%d)hid_sh(%d)hid_CS(%d)fold(%d)BN" % (type(net).__name__, type("DeepHit_Loss").__name__, "Adam",
                    no_rs_batch_size, learning_rate, lr_step_size, lr_gamma, epochs, type(no_rs_activation).__name__, no_rs_dropout, no_rs_alpha, no_rs_beta, no_rs_gamma, data_mode, no_rs_num_layers_shared, no_rs_num_layers_CS, no_rs_h_dim_shared, no_rs_h_dim_CS, index+1)

            # create Adam optimizer
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

            # create learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

            # create dataloaders
            dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch(aggregation), drop_last=True) 
            dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=num_workers, collate_fn=collate_batch(aggregation), pin_memory=True)
            
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

            # import dataset
            train_embeddings, train_time, train_label, train_mask1, train_mask2, test_embeddings, test_time, test_label, test_mask1, test_mask2, num_Category, num_Event = impt.import_CLARO_sequence_embeddings(survival_file="survival_analysis.xlsx", embedding_path="./raw_data/embeddings", label_path="./raw_data/predictions", fold=index+1, data_mode=data_mode, aggregation=aggregation)                

            # testing dataset
            test_dataset = CLARO_att(test_embeddings, test_time, test_label, test_mask1, test_mask2)
            dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=num_workers, collate_fn=collate_batch(aggregation), pin_memory=True)

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
