import numpy as np
import pandas as pd
import os

from rdkit import Chem

from keras.models import load_model
from keras import backend as K
from keras import metrics
import tensorflow as tf
from sklearn.model_selection import KFold

from SMILESX import utils, model, token, augm

##
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

## Inference on the SMILESX predictions
# smiles_list: targeted SMILES list for property inference (Default: ['CC','CCC','C=O'])
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for inference
# augmentation: SMILES's augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Inference/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# returns:
#         Array of SMILES with their inferred property (mean, standard deviation) from models ensembling
def Inference(data,
              data_name, 
              smiles_list = ['CC','CCC','C=O'], 
              data_units = '',
              k_fold_number = 10,
              folds_of_interest = False,
              n_runs = 1,
              augmentation = False, 
              outdir = "./data/"):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    input_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    save_dir = outdir+'Inference/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
    
    print("***SMILES_X for inference starts...***\n\n")
    print("***Checking the SMILES list for inference***\n")
    smiles_checked = list()
    smiles_rejected = list()
    for ismiles in smiles_list:
        mol_tmp = Chem.MolFromSmiles(ismiles)
        if mol_tmp != None:
            smiles_can = Chem.MolToSmiles(mol_tmp)
            smiles_checked.append(smiles_can)
        else:
            smiles_rejected.append(ismiles)
            
    if len(smiles_rejected) > 0:
        with open(save_dir+'rejected_smiles.txt','w') as f:
            for ismiles in smiles_rejected:
                f.write("%s\n" % ismiles)
                
    if len(smiles_checked) == 0:
        print("***Process of inference automatically aborted!***")
        print("The provided SMILES are all incorrect and could not be verified via RDKit.")
        return
    
    smiles_x = np.array(smiles_checked)
    smiles_y = np.array([[np.nan]*len(smiles_checked)]).flatten()
     
    # data augmentation or not
    if augmentation == True:
        print("***Data augmentation.***\n")
        canonical = False
        rotation = True
    else:
        print("***No data augmentation has been required.***\n")
        canonical = True
        rotation = False

    smiles_x_enum, smiles_x_enum_card, smiles_y_enum = \
    augm.Augmentation(smiles_x, smiles_y, canon=canonical, rotate=rotation)

    print("Enumerated SMILES: {}\n".format(smiles_x_enum.shape[0]))
    
    print("***Tokenization of SMILES.***\n")
    # Tokenize SMILES 
    smiles_x_enum_tokens = token.get_tokens(smiles_x_enum)

    # Tokens as a list
    tokens = token.get_vocab(input_dir+data_name+'_Vocabulary.txt')

    # Add 'pad', 'unk' tokens to the existing list
    vocab_size = len(tokens)
    tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)

    # Transformation of tokenized SMILES to vector of intergers and vice-versa
    token_to_int = token.get_tokentoint(tokens)
    int_to_token = token.get_inttotoken(tokens)

    # models ensembling
    smiles_y_pred_mean_array = np.empty(shape=(0,len(smiles_checked)), dtype='float')

    # If there is a list of folds of interest defined by the user
    if folds_of_interest:
        folds = folds_of_interest
    # If there is no defined list of the folds of interest, run for all folds
    else:
        folds = [n for n in range(0, k_fold_number)]
    ifold = 0
    np.random.seed(seed=123)
    kfold = KFold(k_fold_number, shuffle = True)
    for train_val_idx, test_idx in kfold.split(data.smiles):
        if ifold in folds:
            for run in range(n_runs):
                input_dir_run = input_dir +'fold_{}/run_{}/'.format(ifold, run)
                _, _, _, _, _, _, _, scaler = utils.random_split(smiles_input=data.smiles,
                                                                 prop_input=np.array(data.iloc[:,1]),
                                                                 err_input=np.array(data.iloc[:,2]),
                                                                 train_val_idx=train_val_idx,
                                                                 test_idx=test_idx,                                                         
                                                                 scaling = True)
                # Best architecture to visualize from
                model_train = load_model(input_dir_run+data_name+'_model.best_fold_'+str(ifold)+'_run_'+str(run)+'.hdf5', 
                                         custom_objects={'AttentionM': model.AttentionM(seed=5)})

                if ifold == folds[0]:
                    if run == 0:
                        # Maximum of length of SMILES to process
                        max_length = model_train.layers[0].output_shape[-1]
                        print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))
                        print("Maximum length of tokenized SMILES: {} tokens\n".format(max_length))

                # predict and compare for the training, validation and test sets
                smiles_x_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = smiles_x_enum_tokens, 
                                                                     max_length = max_length, 
                                                                     vocab = tokens)

                smiles_y_pred = model_train.predict(smiles_x_enum_tokens_tointvec)
                smiles_y_pred_unscaled = scaler.inverse_transform(smiles_y_pred)

                # compute a mean per set of augmented SMILES
                smiles_y_pred_mean, _ = utils.mean_median_result(smiles_x_enum_card, smiles_y_pred_unscaled)                
                smiles_y_pred_mean_array = np.append(smiles_y_pred_mean_array, smiles_y_pred_mean.reshape(1,-1), axis = 0)
            print("Finished the " + str(ifold) + " fold")
            print("----------------------------------------------------")
            ifold += 1
        else:
            print("Skipped the " + str(ifold) + " fold")
            print("----------------------------------------------------")  
            ifold += 1

    smiles_y_pred_mean_ensemble = np.mean(smiles_y_pred_mean_array, axis = 0)
    smiles_y_pred_sd_ensemble = np.std(smiles_y_pred_mean_array, axis = 0)

    pred_from_ens = pd.DataFrame(data=[smiles_x,
                                       smiles_y_pred_mean_ensemble,
                                       smiles_y_pred_sd_ensemble]).T
    pred_from_ens.columns = ['SMILES', 'ens_pred_mean', 'ens_pred_sd']
    
    print("***Inference of SMILES property done.***")
    
    return pred_from_ens
##
