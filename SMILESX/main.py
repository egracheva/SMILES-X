import numpy as np
import pandas as pd
import os
import math
# For fixing the GPU in use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use (e.g. "0", "1", etc.)
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import matplotlib.pyplot as plt

from numpy.random import seed
seed(12345)

import GPy, GPyOpt

from keras.utils import Sequence
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics
from keras import backend as K
import tensorflow as tf
import multiprocessing

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from SMILESX import utils, token, augm, model
import pygmo as pg

np.set_printoptions(precision=3)
# 
# from sklearn.manifold import TSNE
# from keras.utils import np_utils

##
#from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

## SMILESX main pipeline
# data: provided data (pandas DataFrame of: (SMILES, property))
# data_name: dataset's name
# bayopt_bounds: bounds contraining the Bayesian search of neural architectures
# data_units: property's SI units
# k_fold_number: number of k-folds used for cross-validation (Default: 8)
# augmentation: SMILES augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Main/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# bayopt_n_epochs: number of epochs for training a neural architecture during Bayesian architecture search (Default: 10)
# bayopt_n_rounds: number of architectures to be sampled during Bayesian architecture search (initialization + optimization) (Default: 25)
# bayopt_it_factor: portion of data to be used during Bayesian architecture search (Default: 1)
# bayopt_on: Use Bayesian architecture search or not (Default: True)
# lstmunits_ref: number of LSTM units for the k_fold_index if Bayesian architecture search is off
# denseunits_ref: number of dense units for the k_fold_index if Bayesian architecture search is off
# embedding_ref: number of embedding dimensions for the k_fold_index if Bayesian architecture search is off
# batch_size_ref: batch size for neural architecture training if Bayesian architecture search is off
# alpha_ref: 10^(-alpha_ref) Adam's learning rate for neural architecture training if Bayesian architecture search is off
# n_gpus: number of GPUs to be used in parallel (Default: 1)
# bridge_type: bridge's type to be used by GPUs (e.g. 'NVLink' or 'None') (Default: 'None')
# patience: number of epochs to respect before stopping a training after minimal validation's error (Default: 25)
# n_epochs: maximum of epochs for training (Default: 1000)
# returns:
#         Tokens list (Vocabulary) -> *.txt
#         Best architecture -> *.hdf5
#         Training plot (Loss VS Epoch) -> History_*.png
#         Predictions VS Observations plot -> TrainValidTest_*.png
#
#         per k_fold (e.g. if k_fold_number = 8, 8 versions of these outputs are returned) in outdir
##
def Main(data, 
         data_name, 
         data_units,
         geom_bounds,
         seed_range,
         n_runs = 5,
         k_fold_number = 10,
         folds_of_interest = False,
         augmentation = True, 
         outdir = "./data/", 
         geomopt_on = True,
         n_gpus = 1,
         bridge_type = 'NVLink',
         lstmunits_ref = 2,
         denseunits_ref = 2,
         embedding_ref = 2,
         batch_size_ref = 64,
         alpha_ref = 2.8,
         best_seed = -0.5,
         n_epochs = 100,
         ignore_first_epochs = 1):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    save_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
        
    print("***SMILES_X starts...***\n\n")
    np.random.seed(seed=123)
    seed_list = np.random.randint(int(1e6), size = k_fold_number).tolist()
    
    # Train/validation/test data splitting - 80/10/10 % at random with diff. seeds for k_fold_number times
    kfold = KFold(k_fold_number, shuffle = True)
    # If there is a list of folds of interest defined by the user
    if folds_of_interest:
        folds = folds_of_interest
    # If there is no defined list of the folds of interest, run for all folds
    else:
        folds = [n for n in range(0, k_fold_number)]
        final_prediction = np.empty((len(data.smiles), n_runs), dtype = float)
    ifold = 0

    for train_val_idx, test_idx in kfold.split(data.smiles):
        if ifold in folds:        
            print("******")
            print("***Fold #{} initiated...***".format(ifold))
            print("******")
        
            print("***Sampling and splitting of the dataset.***\n")
            x_train, x_valid, x_test, y_train, y_valid, y_test, y_err, scaler = \
            utils.random_split(smiles_input=data.smiles,
                               prop_input=np.array(data.iloc[:,1]),
                               err_input=np.array(data.iloc[:,2]),
                               train_val_idx=train_val_idx,
                               test_idx=test_idx,                                                         
                               scaling = True)
            
            # data augmentation or not
            if augmentation == True:
                print("***Data augmentation to {}***\n".format(augmentation))
                canonical = False
                rotation = True
            else:
                print("***No data augmentation has been required.***\n")
                canonical = True
                rotation = False
                
            x_train_enum, x_train_enum_card, y_train_enum = \
            augm.Augmentation(x_train, y_train, canon=canonical, rotate=rotation)
    
            x_valid_enum, x_valid_enum_card, y_valid_enum = \
            augm.Augmentation(x_valid, y_valid, canon=canonical, rotate=rotation)
    
            x_test_enum, x_test_enum_card, y_test_enum = \
            augm.Augmentation(x_test, y_test, canon=canonical, rotate=rotation)
            
            print("Enumerated SMILES:\n\tTraining set: {}\n\tValidation set: {}\n\tTest set: {}\n".\
            format(x_train_enum.shape[0], x_valid_enum.shape[0], x_test_enum.shape[0]))
            
            print("***Tokenization of SMILES.***\n")
            # Tokenize SMILES per dataset
            x_train_enum_tokens = token.get_tokens(x_train_enum)
            x_valid_enum_tokens = token.get_tokens(x_valid_enum)
            x_test_enum_tokens = token.get_tokens(x_test_enum)
            
            print("Examples of tokenized SMILES from a training set:\n{}\n".\
            format(x_train_enum_tokens[:5]))
            
            # Vocabulary size computation
            all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens
            tokens = token.extract_vocab(all_smiles_tokens)
            vocab_size = len(tokens)
            
            train_unique_tokens = token.extract_vocab(x_train_enum_tokens)
            print("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
            valid_unique_tokens = token.extract_vocab(x_valid_enum_tokens)
            print("Number of tokens only present in a validation set: {}".format(len(valid_unique_tokens)))
            print("Is the validation set a subset of the training set: {}".\
                  format(valid_unique_tokens.issubset(train_unique_tokens)))
            print("What are the tokens by which they differ: {}\n".\
                  format(valid_unique_tokens.difference(train_unique_tokens)))
            test_unique_tokens = token.extract_vocab(x_test_enum_tokens)
            print("Number of tokens only present in a test set: {}".format(len(test_unique_tokens)))
            print("Is the test set a subset of the training set: {}".\
                  format(test_unique_tokens.issubset(train_unique_tokens)))
            print("What are the tokens by which they differ: {}".\
                  format(test_unique_tokens.difference(train_unique_tokens)))
            print("Is the test set a subset of the validation set: {}".\
                  format(test_unique_tokens.issubset(valid_unique_tokens)))
            print("What are the tokens by which they differ: {}\n".\
                  format(test_unique_tokens.difference(valid_unique_tokens)))
            
            print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))
            
            # Save the vocabulary for re-use
            token.save_vocab(tokens, save_dir+data_name+'_tokens_set_fold_'+str(ifold)+'.txt')
            # Tokens as a list
            tokens = token.get_vocab(save_dir+data_name+'_tokens_set_fold_'+str(ifold)+'.txt')
            # Add 'pad', 'unk' tokens to the existing list
            tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)
            
            # Maximum of length of SMILES to process
            max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
            print("Maximum length of tokenized SMILES: {} tokens (termination spaces included)\n".format(max_length))
            
            print("***Optimization of the SMILESX's architecture.***\n")
            # Transformation of tokenized SMILES to vector of intergers and vice-versa
            token_to_int = token.get_tokentoint(tokens)
            int_to_token = token.get_inttotoken(tokens)
            
            if geomopt_on:
                # Operate the first step of the optimization:
                # geometry optimization (number of units in LSTM and Dense layers)
                def create_mod_geom(params, seed_range):
                    # Prepare the data for the evaluation
                    x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = x_train_enum_tokens, 
                                                   max_length = max_length+1, 
                                                   vocab = tokens)
                    x_valid_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = x_valid_enum_tokens, 
                                                   max_length = max_length+1, 
                                                   vocab = tokens)
                    # Evaluation of the untrained model is performed on both training and validation data merged together
                    x_geom_enum_tokens_tointvec = np.concatenate((x_train_enum_tokens_tointvec, x_valid_enum_tokens_tointvec), axis = 0)
                    y_geom_enum                 = np.concatenate((y_train_enum, y_valid_enum), axis = 0)
                    pred_scores = []
                    for i, weight in enumerate(seed_range):
                        #Changing weight into seed for the test
                        # weight = weight*2+5
                        # print(weight)
                        K.clear_session()
    
                        if n_gpus > 1:
                            if bridge_type == 'NVLink':
                                model_geom = model.LSTMAttModelNoTrain.create(inputtokens=max_length+1, 
                                                                              vocabsize=vocab_size,
                                                                              weight=weight,
                                                                              lstmunits=int(params[0]), 
                                                                              denseunits=int(params[1]), 
                                                                              embedding=int(params[2])
                                                                              )
                            else:
                                with tf.device('/cpu'): # necessary to multi-GPU scaling
                                    model_geom = model.LSTMAttModelNoTrain.create(inputtokens = max_length+1, 
                                                                                  vocabsize=vocab_size,
                                                                                  weight=weight,
                                                                                  lstmunits=int(params[0]), 
                                                                                  denseunits=int(params[1]), 
                                                                                  embedding=int(params[2])
                                                                                  )
                                    
                            multi_model = model.ModelMGPU(model_geom, gpus=n_gpus, bridge_type=bridge_type)
                        else: # single GPU
                            model_geom = model.LSTMAttModelNoTrain.create(inputtokens = max_length+1, 
                                                                          vocabsize=vocab_size,
                                                                          weight=weight, 
                                                                          lstmunits=int(params[0]), 
                                                                          denseunits=int(params[1]), 
                                                                          embedding=int(params[2])
                                                                          )
                            
                            multi_model = model_geom
                        # Compiling the model
                        multi_model.compile(loss='mse', optimizer='sgd')
                        y_geom_pred = model_geom.predict(x_geom_enum_tokens_tointvec, verbose=0)
                        score = np.sqrt(mean_squared_error(scaler.inverse_transform(y_geom_enum), scaler.inverse_transform(y_geom_pred)))
                        pred_scores.append(score)
                    mean_score = np.mean(pred_scores)
                    best_score = np.min(pred_scores)
                    sigma_score = np.std(pred_scores)
                    best_seed = seed_range[np.argmin(pred_scores)]
                    n_nodes = model_geom.count_params()
                    return [mean_score, sigma_score, best_score, best_seed, n_nodes]
    
                print("***Geometry search.***")
                # start = time.time()
                # Test each geometry using the single shared weight (all the weights are set to constant)
                # The score is evaluated over the mean of the sulting predictions
                # This is the way to test each geometry for "compatibility" with the data -- without training the model
                # Read more in David Ha's article
                scores = []
                for n_lstm in geom_bounds[0]:
                    for n_dense in geom_bounds[1]:
                        for n_embed in geom_bounds[2]:
                          scores.append([n_lstm, n_dense, n_embed] + create_mod_geom([n_lstm, n_dense, n_embed], seed_range))
                scores = np.array(scores)
                ## Multistage sorting procedure
                ## Firstly, sort based on mean score and best score over weights
                points = scores[:, [3, 4]].tolist()
                sort_ind = pg.sort_population_mo(points)
                print("Unordered scores")
                print(pd.DataFrame(scores))
                print("Re-ordered scores")
                print(pd.DataFrame(scores[sort_ind]))
    
                # Select the best geometry for further learning rate and batch size optimisation
                best_geom = scores[sort_ind[0], :3]
                best_seed = scores[sort_ind[0], 6]
                print("The best untrained RMSE is:")
                print(scores[sort_ind[0], 5])
                print("Which is achieved using the weight of {}".format(best_seed))
                print("\nThe best selected geometry is:\n\tLSTM units: {}\n\tDense units: {}\n\tEmbedding {}".\
                      format(best_geom[0], best_geom[1], best_geom[2]))
                best_arch = best_geom.tolist()
            else:
                best_arch = [lstmunits_ref, denseunits_ref, embedding_ref, batch_size_ref, alpha_ref]
                
            print("\nThe architecture for this datatset is:\n\tLSTM units: {}\n\tDense units: {}\n\tEmbedding dimensions {}".\
                 format(int(best_arch[0]), int(best_arch[1]), int(best_arch[2])))
            
            print("***Training the best model.***\n")
            prediction_train_bag = np.zeros((x_train.shape[0], n_runs))
            prediction_valid_bag = np.zeros((x_valid.shape[0], n_runs))
            prediction_test_bag = np.zeros((x_test.shape[0], n_runs))
            for run in range(n_runs):
                print("*** Run #{} ***".format(run+1))
                # Train the model and predict
                K.clear_session()  
                # Define the multi-gpus model if necessary
                if n_gpus > 1:
                    if bridge_type == 'NVLink':
        
                        model_train = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                                vocabsize = vocab_size,
                                                                weight=best_seed,
                                                                lstmunits= int(best_arch[0]), 
                                                                denseunits = int(best_arch[1]), 
                                                                embedding = int(best_arch[2]))
                    else:
                        with tf.device('/cpu'):
                            model_train = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                                    vocabsize = vocab_size,
                                                                    weight=best_seed,
                                                                    lstmunits= int(best_arch[0]), 
                                                                    denseunits = int(best_arch[1]), 
                                                                    embedding = int(best_arch[2]))
        
                    print("Best model summary:\n")
                    print(model_train.summary())
                    print("\n")
                    multi_model = model.ModelMGPU(model_train, gpus=n_gpus, bridge_type=bridge_type)
                else:
        
                    model_train = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                            vocabsize = vocab_size,
                                                            weight=best_seed,
                                                            lstmunits= int(best_arch[0]), 
                                                            denseunits = int(best_arch[1]), 
                                                            embedding = int(best_arch[2]))
        
                    print("Best model summary:\n")
                    print(model_train.summary())
                    print("\n")
                    multi_model = model_train
        
                # We define the optimal batch size based on B ~ learning_rate*N_training empiric rule
                # Read more on it in the paper from Google Brain
                # "Don't decay the learning rate, increase the batch size" by S.Smith, Q.Le
                # https://arxiv.org/abs/1711.00489
                learning_rate = math.pow(10,-3)
                batch_size_max = np.ceil(len(x_train_enum)*learning_rate)
                batch_size_schedule = [int(np.ceil(batch_size_max/9)), int(np.ceil(batch_size_max/3)), int(np.ceil(batch_size_max))]
                n_epochs_schedule = [int(n_epochs/3), int(n_epochs/3), n_epochs - 2*int(n_epochs/3)]
                custom_adam = Adam(lr=learning_rate)
                # Compile the model
                multi_model.compile(loss="mse", optimizer=custom_adam, metrics=[metrics.mae,metrics.mse])
                
                # Checkpoint, Early stopping and callbacks definition
                filepath=save_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(ifold)+'.hdf5'
        
                # Fit the model applying the batch size schedule:
                n_epochs_done = 0
                best_val = np.Inf
                losses = []
                val_losses = []
                for i, batch_size in enumerate(batch_size_schedule):
                    n_epochs_part = n_epochs_schedule[i]
                    ignorebeginning = model.IgnoreBeginningSaveBest(filepath=filepath,
                                                                    best=best_val,
                                                                    ignore_first_epochs=ignore_first_epochs)
                    callbacks_list = [ignorebeginning]  
                    print("Schedule step number {0}\nNumber of epochs: {1}, batch size: {2}".format(i+1, n_epochs_part, batch_size))
                    history = multi_model.fit_generator(generator = DataSequence(x_train_enum_tokens,
                                                                                 vocab = tokens, 
                                                                                 max_length = max_length, 
                                                                                 props_set = y_train_enum, 
                                                                                 batch_size = batch_size), 
                                                        validation_data = DataSequence(x_valid_enum_tokens,
                                                                                       vocab = tokens, 
                                                                                       max_length = max_length, 
                                                                                       props_set = y_valid_enum, 
                                                                                       batch_size = batch_size),
                                                        shuffle = True,
                                                        initial_epoch = n_epochs_done,
                                                        epochs = n_epochs_done+n_epochs_part,
                                                        callbacks = callbacks_list,
                                                        verbose = 0)
                    losses += history.history['loss']
                    val_losses += history.history['val_loss']
                    best_val = ignorebeginning.best
                    n_epochs_done += n_epochs_part
    
                # Summarize history for losses per epoch
                plt.plot(losses)
                plt.plot(val_losses)
                plt.title('')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                plt.savefig(save_dir+'History_fit_LSTMAtt_'+data_name+'_model_fold_'+str(ifold)+'_run_'+str(run+1)+'.png', bbox_inches='tight')
                plt.close()
    
                print("***Predictions from the best model.***\n")
                model_train.load_weights(save_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(ifold)+'.hdf5')
                model_train.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])
        
                # predict and compare for the training, validation and test sets
                x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = x_train_enum_tokens, 
                                                                    max_length = max_length+1, 
                                                                    vocab = tokens)
                x_valid_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = x_valid_enum_tokens, 
                                                                    max_length = max_length+1, 
                                                                    vocab = tokens)
                x_test_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = x_test_enum_tokens, 
                                                                   max_length = max_length+1, 
                                                                   vocab = tokens)
        
                y_pred_train = model_train.predict(x_train_enum_tokens_tointvec)
                y_pred_valid = model_train.predict(x_valid_enum_tokens_tointvec)
                y_pred_test = model_train.predict(x_test_enum_tokens_tointvec)
        
                # compute a mean per set of augmented SMILES
                y_pred_train_mean, _ = utils.mean_median_result(x_train_enum_card, y_pred_train)
                y_pred_valid_mean, _ = utils.mean_median_result(x_valid_enum_card, y_pred_valid)
                y_pred_test_mean, _ = utils.mean_median_result(x_test_enum_card, y_pred_test)
                
                prediction_train_bag[:,run] = scaler.inverse_transform(y_pred_train_mean.reshape(-1,1)).ravel()
                prediction_valid_bag[:,run] = scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1)).ravel()
                prediction_test_bag[:,run] = scaler.inverse_transform(y_pred_test_mean.reshape(-1,1)).ravel()

                final_prediction[test_idx, run] = scaler.inverse_transform(y_pred_test_mean.reshape(-1,1)).ravel()
            
            # Performing inverse transformation (unscaling)
            y_true = {}
            y_true['train'] = scaler.inverse_transform(y_train).ravel()
            y_true['valid'] = scaler.inverse_transform(y_valid).ravel()
            y_true['test'] = scaler.inverse_transform(y_test).ravel()

            #y_train_pred = scaler.inverse_transform(y_pred_train_mean.reshape(-1,1))
            #y_valid_pred = scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1))
            y_preds_mean = {}
            y_preds_mean['train'] = np.mean(prediction_train_bag, axis = 1)
            y_preds_mean['valid'] = np.mean(prediction_valid_bag, axis = 1)
            y_preds_mean['test'] = np.mean(prediction_test_bag, axis = 1)

            y_preds_sigma = {}
            y_preds_sigma['train'] = np.std(prediction_train_bag, axis = 1)
            y_preds_sigma['valid'] = np.std(prediction_valid_bag, axis = 1)
            y_preds_sigma['test'] = np.std(prediction_test_bag, axis = 1)
            
            for name in ['train', 'valid', 'test']:
                print(name.capitalize() + ' set:')
                N = float(y_true[name].shape[0])
                print(y_true[name].shape)
                sstot = np.sum(np.square(y_true[name] - np.mean(y_true[name])), axis=0)
                ssres = np.sum(np.square(y_true[name] - y_preds_mean[name]), axis=0)
                
                # R2-score
                r2 = r2_score(y_true[name], y_preds_mean[name])
                # Error on R2-score when taking into account error on predictions only
                d_r2 = 2/sstot*np.sqrt(np.square(y_true[name]-y_preds_mean[name]).dot(np.square(y_preds_sigma[name])))
                # Error on R2-score when taking into account both errors on predictions and true data
                d_r2_exp = (2 / np.square(ssres) *
                            np.sqrt(np.square(sstot) * np.square(y_true[name]-y_preds_mean[name])
                            .dot(np.square(y_preds_sigma[name])) +
                            ((y_true[name] - np.mean(y_true[name])) * ssres - (y_true[name]-y_preds_mean[name]) * sstot)
                            .dot(np.square(y_err[name])))
                            )

                # RMSE
                rmse = np.sqrt(mean_squared_error(y_true[name], y_preds_mean[name]))
                # Setup the precision of the displayed error to print it cleanly
                precision_rmse = (np.abs(np.floor(np.log10(rmse)))+3)/10
                # Error on RMSE when taking into account error on predictions only
                d_rmse = np.sqrt(np.square(y_true[name]-y_preds_mean[name]).dot(np.square(y_preds_sigma[name]))/N/ssres)
                # Error on RMSE when taking into account both errors on predictions and true data
                d_rmse_exp = np.sqrt(np.square(y_true[name]-y_preds_mean[name]).dot(np.square(y_preds_sigma[name]) + np.square(y_err[name]))/N/ssres)
                print(np.square(y_true[name]-y_preds_mean[name]).shape)
                print((np.square(y_preds_sigma[name]) + np.square(y_err[name])).shape)
                # MAE
                mae = mean_absolute_error(y_true[name], y_preds_mean[name])
                # Setup the precision of the displayed error to print it cleanly
                precision_mae = (np.abs(np.floor(np.log10(mae)))+3)/10
                # Error on RMSE when taking into account error on predictions only
                d_mae = np.sqrt(np.sum(np.square(y_preds_sigma[name])))/N
                # Error on RMSE when taking into account both errors on predictions and true data
                d_mae_exp = np.sqrt(np.sum(np.square(y_preds_sigma[name]) + np.square(y_err[name])))/N

                print("Averaged R^2: {0:0.4f}+-{1:0.4f}({2:0.4f})".format(r2, d_r2, d_r2_exp[0]))
                print("Averaged RMSE: {0:{3}f}+-{1:{3}f}({2:{3}f})".format(rmse, d_rmse, d_rmse_exp[0], precision_rmse))
                print("Averaged MAE: {0:{3}f}+-{1:{3}f}({2:{3}f})\n".format(mae, d_mae, d_mae_exp, precision_mae))
            
    
            # Changed colors, scaling and sizes
            plt.figure(figsize=(12, 8))
    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Setting plot limits
            y_true_min = min(np.min(y_true['train']), np.min(y_true['valid']), np.min(y_true['test']))
            y_true_max = max(np.max(y_true['train']), np.max(y_true['valid']), np.max(y_true['test']))
            y_pred_min = min(np.min(y_preds_mean['train']), np.min(y_preds_mean['valid']), np.min(y_preds_mean['test']))
            y_pred_max = max(np.max(y_preds_mean['train']), np.max(y_preds_mean['valid']), np.max(y_preds_mean['test']))
            # Expanding slightly the canvas around the data points (by 10%)
            axmin = y_true_min-0.1*(y_true_max-y_true_min)
            axmax = y_true_max+0.1*(y_true_max-y_true_min)
            aymin = y_pred_min-0.1*(y_pred_max-y_pred_min)
            aymax = y_pred_max+0.1*(y_pred_max-y_pred_min)
    
            plt.xlim(min(axmin, aymin), max(axmax, aymax))
            plt.ylim(min(axmin, aymin), max(axmax, aymax))
                            
            plt.errorbar(y_true['train'], 
                        y_preds_mean['train'],
                        xerr = y_err['train'],
                        yerr = y_preds_sigma['train'],
                        fmt='o',
                        label="Train",
                        ecolor='#519fc4',
                        elinewidth = 0.5, 
                        ms=5,
                        mfc='#519fc4',
                        markeredgewidth = 0,
                        alpha=0.7)
            plt.errorbar(y_true['valid'], 
                        y_preds_mean['valid'],
                        xerr = y_err['valid'],
                        yerr = y_preds_sigma['valid'],
                        ecolor='#db702e',
                        elinewidth = 0.5,
                        fmt='o',
                        label="Validation", 
                        ms=5, 
                        mfc='#db702e',
                        markeredgewidth = 0,
                        alpha=0.7)
            plt.errorbar(y_true['test'], 
                        y_preds_mean['test'],
                        xerr = y_err['test'],
                        yerr = y_preds_sigma['test'],
                        ecolor='#cc1b00',
                        elinewidth = 0.5,
                        fmt='o',
                        label="Test", 
                        ms=5, 
                        mfc='#cc1b00',
                        markeredgewidth = 0,
                        alpha=0.7)
    
    
            # Plot X=Y line
            plt.plot([max(plt.xlim()[0], plt.ylim()[0]), 
                      min(plt.xlim()[1], plt.ylim()[1])],
                     [max(plt.xlim()[0], plt.ylim()[0]), 
                      min(plt.xlim()[1], plt.ylim()[1])],
                     ':', color = '#595f69')
            
            plt.xlabel('Observations ' + data_units, fontsize = 12)
            plt.ylabel('Predictions ' + data_units, fontsize = 12)
            plt.legend()
    
            plt.savefig(save_dir+'TrainValid_Plot_LSTMAtt_'+data_name+'_model_fold_'+str(ifold)+'.png', bbox_inches='tight', dpi=80)
            plt.close()
            print("Finished the " + str(ifold) + " fold")
            print("----------------------------------------------------")
            ifold += 1
        else:
            print("Skipped the " + str(ifold) + " fold")
            print("----------------------------------------------------")  
            ifold += 1
    if final_prediction:
        final_prediction_mean = np.mean(final_prediction, axis = 1)
        final_prediction_sigma = np.std(final_prediction, axis = 1)
        data_values = np.array(data.iloc[:,1])
        data_error = np.array(data.iloc[:,2])
        
        mae_final = np.mean(np.absolute(final_prediction_mean - data_values))
        mse_final = np.mean(np.square(final_prediction_mean - data_values))
        corrcoef_final = r2_score(final_prediction_mean, data_values)

        # Calculating the R2 for the averaged training results
        N_final = data_values.shape[0]
        sstot_final = np.sum(np.square(data_values - np.mean(data_values)), axis=0)
        ssres_final = np.sum(np.square(data_values - final_prediction_mean), axis=0)

        r2_final = r2_score(data_values, final_prediction_mean)
        d_r2_final = 2/sstot_final*np.sqrt(np.square(data_values-final_prediction_mean).dot(np.square(final_prediction_sigma)))
        d_r2_final_exp = (2 / np.square(ssres_final) *
                          np.sqrt(np.square(sstot_final) * np.square(data_values-final_prediction_mean)
                          .dot(np.square(final_prediction_sigma)) +
                          ((data_values - np.mean(data_values)) * ssres_final - (data_values-final_prediction_mean) * sstot_final)
                          .dot(np.square(data_error)))
                          )

        rmse_final = np.sqrt(mean_squared_error(data_values, final_prediction_mean))
        d_rmse_final = np.sqrt(np.square(data_values-final_prediction_mean).dot(np.square(final_prediction_sigma))*sstot_final/N_final)
        d_rmse_final_exp = np.sqrt(np.square(data_values-final_prediction_mean).dot(np.square(final_prediction_sigma) + np.square(data_error))/N_final/ssres_final)

        mae_final = mean_absolute_error(data_values, final_prediction_mean)
        d_mae_final = np.sqrt(np.sum(np.square(final_prediction_sigma)))/N_final
        d_mae_final_exp = np.sqrt(np.sum(np.square(final_prediction_sigma) + np.square(data_error)))/N_final

        print("Final averaged R^2:")
        print("{0:0.4f}+-{1:0.4f}({2:0.4f})".format(r2_final, d_r2_final, d_r2_final_exp[0]))
        print("Final averaged RMSE:")
        print("{0:{3}f}+-{1:{3}f}({2:{3}f})".format(rmse_final, d_rmse_final, d_rmse_final_exp[0], precision_rmse))
        print("Final averaged MAE:")
        print("{0:{3}f}+-{1:{3}f}({2:{3}f})\n".format(mae_final, d_mae_final, d_mae_final_exp, precision_mae))
        
        # Changed colors, scaling and sizes
        plt.figure(figsize=(12, 8))
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setting plot limits
        y_true_min = np.min(data_values)
        y_true_max = np.max(data_values)
        y_pred_min = np.min(final_prediction)
        y_pred_max = np.max(final_prediction)
        # Expanding slightly the canvas around the data points (by 10%)
        axmin = y_true_min-0.1*(y_true_max-y_true_min)
        axmax = y_true_max+0.1*(y_true_max-y_true_min)
        aymin = y_pred_min-0.1*(y_pred_max-y_pred_min)
        aymax = y_pred_max+0.1*(y_pred_max-y_pred_min)
        
        plt.xlim(min(axmin, aymin), max(axmax, aymax))
        plt.ylim(min(axmin, aymin), max(axmax, aymax))
        
        plt.errorbar(data_values,
                     final_prediction_mean,
                     xerr = data_error,
                     yerr = final_prediction_sigma,
                     fmt='o',
                     label="",
                     ecolor='#595f69',
                     elinewidth = 0.5, 
                     ms=5,
                     mfc='#519fc4',
                     markeredgewidth = 0,
                     alpha=0.7)
        
        
        # Plot X=Y line
        plt.plot([max(plt.xlim()[0], plt.ylim()[0]), 
                  min(plt.xlim()[1], plt.ylim()[1])],
                 [max(plt.xlim()[0], plt.ylim()[0]), 
                  min(plt.xlim()[1], plt.ylim()[1])],
                 ':', color = '#595f69')
        
        plt.xlabel('Observations ' + data_units, fontsize = 12)
        plt.ylabel('Predictions ' + data_units, fontsize = 12)
        
        plt.savefig(save_dir+'TrainValid_Plot_LSTMAtt_'+data_name+'_model_FinalPrediction.png', bbox_inches='tight', dpi=80)
        plt.close()
        
        
##        


## Data sequence to be fed to the neural network during training through batches of data
class DataSequence(Sequence):
    # Initialization
    # smiles_set: array of tokenized SMILES of dimensions (number_of_SMILES, max_length)
    # vocab: vocabulary of tokens
    # max_length: maximum length for SMILES in the dataset
    # props_set: array of targeted property
    # batch_size: batch's size
    # soft_padding: pad tokenized SMILES at the same length in the whole dataset (False), or in the batch only (True) (Default: False)
    # returns: 
    #         a batch of arrays of tokenized and encoded SMILES, 
    #         a batch of SMILES property
    def __init__(self, smiles_set, vocab, max_length, props_set, batch_size, soft_padding = False):
        self.smiles_set = smiles_set
        self.vocab = vocab
        self.max_length = max_length
        self.props_set = props_set
        self.batch_size = batch_size
        self.iepoch = 0
        self.soft_padding = soft_padding

    def on_epoch_end(self):
        self.iepoch += 1
        
    def __len__(self):
        return int(np.ceil(len(self.smiles_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        tokenized_smiles_list_tmp = self.smiles_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        # self.max_length + 1 padding
        if self.soft_padding:
            max_length_tmp = np.max([len(ismiles) for ismiles in tokenized_smiles_list_tmp])
        else:
            max_length_tmp = self.max_length
        batch_x = token.int_vec_encode(tokenized_smiles_list = tokenized_smiles_list_tmp, 
                                 max_length = max_length_tmp+1,
                                 vocab = self.vocab)
        #batch_x = self.batch[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.props_set[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)
##

