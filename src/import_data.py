'''
This provide the dimension/data/mask to train/test the network.

Once must construct a function similar to "import_dataset_CLARO":
    - DATA FORMAT:
        > data: covariates with x_dim dimension.
        > label: 0: censoring, 1 ~ K: K competing(single) risk(s)
        > time: time-to-event or time-to-censoring
    - Based on the data, creat mask1 and mask2 that are required to calculate loss functions.
'''
import numpy as np
import pandas as pd
import json

##### DEFINE USER-FUNCTIONS #####
def f_get_Normalization(X, norm_mode):
    _, num_Feature = np.shape(X)

    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))/np.std(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j]) - np.min(X[:,j]))
    else:
        print("INPUT MODE ERROR!")

    return X

### MASK FUNCTIONS
'''
    fc_mask2      : To calculate LOSS_1 (log-likelihood loss)
    fc_mask3      : To calculate LOSS_2 (ranking loss)
'''
def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask

# import patient embeddings for a given fold
# if aggregation embeddings size is batch_size x number of sequences x embedding_size, otherwise batch_size x number of sequences x number of tokens x embedding_size 
def get_embeddings(survival_file, embedding_path, label_path, fold, data_mode, k_fold_path, train=True, aggregation=False):
    """
    Get patient embeddings, labels, overall survival (OS), and censoring information for a given fold.

    Args:
        survival_file (str): Path to the file containing survival status and time.
        embedding_path (str): Path to the embeddings file.
        label_path (str): Path to the labels file.
        fold (int): Fold number.
        train (bool, optional): Whether to get embeddings for the training fold. Defaults to True.

    Returns:
        Tuple: Tuple containing patient embeddings
    """

    # patients studied for overall survival analysis
    if survival_file.endswith('.csv'):
        df = pd.read_csv(survival_file)
    else:
        df = pd.read_excel(survival_file)
    patients = df['ID paziente'].to_list()

    # selecting train/test patients 
    filename = f"{k_fold_path}/exp_{fold}th/train_{fold}.json" if train else f"{k_fold_path}/exp_{fold}th/test_{fold}.json"
    fold_patients = []
    with open(filename, 'r', encoding="utf-8") as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            ehr = json.loads(json_str)
            ehr_id = ehr['id']
            fold_patients.append(ehr_id)

    # computing embeddings
    current_patient = None
    counter_sentence = 0
    patient_embeddings = []
    # patient_entities = []
    with open(label_path, 'r') as label_file:
        
        embeddings = np.load(embedding_path, allow_pickle=True)

        tmp_embeddings = []
        tmp_sequences = []
        for embedding_sentence, label_sentence in zip(embeddings, label_file):
            # next patient
            patient = int(fold_patients[counter_sentence])
            counter_sentence += 1
            # patient analysis
            if patient in patients:
                # handling first patient
                if current_patient is None:
                    current_patient = patient
                # current patient id and next patient are different -> collect current patient tokens in a list
                if current_patient != patient:
                    if aggregation:
                        tmp_sequences = np.array(tmp_sequences)
                    else:
                        tmp_sequences = np.array(tmp_sequences, dtype=object)
                    patient_embeddings.append(tmp_sequences)
                    tmp_sequences = []
                # loading next patient tokens in a list (consider only entity tokens whether data_mode is not CLAROv0bis)
                for embedding_token, label_token in zip(embedding_sentence, label_sentence.split()):
                    if data_mode=="CLAROv0bis":
                        tmp_embeddings.append(embedding_token)
                    else:
                        if label_token != "O":
                            tmp_embeddings.append(embedding_token)
                # if aggregration consider the mean of tokens in the sentence, otherwise all tokens
                if len(tmp_embeddings) != 0:
                    # embeddings
                    tmp_embeddings = np.array(tmp_embeddings)
                    if aggregation:
                        mean_embedding = np.mean(tmp_embeddings, axis=0)
                        tmp_sequences.append(mean_embedding)                
                    else:
                        tmp_sequences.append(tmp_embeddings)
                tmp_embeddings = []
                # update current patient id
                current_patient = patient

        if aggregation:
            tmp_sequences = np.array(tmp_sequences)
        else:
            tmp_sequences = np.array(tmp_sequences, dtype=object)
        patient_embeddings.append(tmp_sequences)

    return patient_embeddings 

# import patient embeddings, labels, overall survival (OS), and censoring information for a given fold
def import_CLARO_sequence_embeddings(survival_file, embedding_path, prediction_path, label_path, fold, kfold_path, data_mode="CLAROv0", aggregation=False):
    """
    Import CLARO embeddings, labels, and overall survival (OS) data for a given fold.
    sequence embeddings are used for the input of the network.
        survival_file (str): Path to the survival file.
        embedding_path (str): Path to the embeddings file.
        prediction_path (str): Path to the predictions file.
        label_path (str): Path to the labels file.
        fold (int): Fold number.
        data_mode (str, optional): Data mode. Can be "CLAROv0", "CLAROv0bis", "CLAROclinical", or "regex_only". Defaults to "CLAROv0".
        aggregation (bool, optional): Whether to aggregate embeddings. Defaults to False.
    Returns:
        Tuple: Tuple containing training and testing patient embeddings, labels, overall survival (OS), censoring information, masks, number of categories, and number of events.
    """
    # paths of NER embeddings and predictions
    tr_embedding_path = embedding_path  + "/embedding_train_" + str(fold-1) + ".npy"
    te_embedding_path = embedding_path  + "/embedding_test_" + str(fold-1) + ".npy"
    tr_pred_path = prediction_path  + "/pred_train_" + str(fold-1) + ".txt"
    te_pred_path = prediction_path  + "/pred_test_" + str(fold-1) + ".txt"

    # get training and test embeddings
    tr_patient_embeddings = get_embeddings(survival_file, tr_embedding_path, tr_pred_path, fold, data_mode, kfold_path, True, aggregation)
    te_patient_embeddings = get_embeddings(survival_file, te_embedding_path, te_pred_path, fold, data_mode, kfold_path, False, aggregation)
    
    # read csv files containing survival times and events
    filename_train_labels = f"{label_path}/months/label__train_{fold-1}.csv"
    filename_test_labels = f"{label_path}/months/label__test_{fold-1}.csv"
    df_train_labels = pd.read_csv(filename_train_labels, sep =',')
    df_test_labels = pd.read_csv(filename_test_labels, sep =',')

    # concatenation in order to compute the number of categories over the whole dataset
    df_all_labels = pd.concat([df_train_labels, df_test_labels], axis=0, ignore_index=False)
        
    # reset the index of the concatenated DataFrame
    df_all_labels = df_all_labels.reset_index(drop=True)
    all_times  = np.asarray(df_all_labels[['event_time']])
    all_labels = np.asarray(df_all_labels[['label']])
    num_Category    = int(np.max(all_times) * 1.2)        #to have enough time-horizon
    num_Event       = int(len(np.unique(all_labels)) - 1) #only count the number of events (do not count censoring as an event)

    # numpy array conversions
    tr_patient_embeddings = np.asarray(tr_patient_embeddings, dtype=object)
    te_patient_embeddings = np.asarray(te_patient_embeddings, dtype=object)
    tr_patient_OS  = np.asarray(df_train_labels[['event_time']])
    te_patient_OS  = np.asarray(df_test_labels[['event_time']])
    tr_patient_cens_OS = np.asarray(df_train_labels[['label']])
    te_patient_cens_OS = np.asarray(df_test_labels[['label']])

    # get mask1s and mask2s
    tr_mask1 = f_get_fc_mask2(tr_patient_OS, tr_patient_cens_OS, num_Event, num_Category)
    tr_mask2 = f_get_fc_mask3(tr_patient_OS, -1, num_Category)
    te_mask1 = f_get_fc_mask2(te_patient_OS, tr_patient_cens_OS, num_Event, num_Category)
    te_mask2 = f_get_fc_mask3(te_patient_OS, -1, num_Category)

    return tr_patient_embeddings, tr_patient_OS, tr_patient_cens_OS, tr_mask1, tr_mask2, te_patient_embeddings, te_patient_OS, te_patient_cens_OS, te_mask1, te_mask2, num_Category, num_Event

# import regular expression features for a given fold 
def import_reg_features(reg_path, fold):
    tr_df = pd.read_csv(reg_path + "/reg_features_train_" + str(fold) + ".csv", sep =',')
    tr_reg_arr = np.asarray(tr_df)
    te_df = pd.read_csv(reg_path + "/reg_features_test_" + str(fold) + ".csv", sep =',')
    te_reg_arr = np.asarray(te_df)

    return tr_reg_arr, te_reg_arr


# import clinical data for a given fold
def import_dataset_CLINICAL(root_dir, fold, step):
    if not (step == 'train' or step == 'test'):
        raise ValueError("Invalid step")
    
    df_features = pd.read_csv(f'{root_dir}/clean_features_final__{step}_{fold}.csv', sep =',')
    df_labels = pd.read_csv(f'{root_dir}/label__{step}_{fold}.csv', sep =',')

    data  = np.asarray(df_features)
    data  = f_get_Normalization(data, 'standard')

    time  = np.asarray(df_labels[['event_time']])
    event = np.asarray(df_labels[['label']])

    num_Category    = int(np.max(time) * 1.2)        #to have enough time-horizon
    num_Event       = int(len(np.unique(event)) - 1) #only count the number of events (do not count censoring as an event)

    x_dim = np.shape(data)[1]

    mask1 = f_get_fc_mask2(time, event, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    DIM             = (x_dim)
    DATA            = (data, time, event)
    MASK            = (mask1, mask2)

    return DIM, DATA, MASK
