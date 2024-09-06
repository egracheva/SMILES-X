"""Add main docstring discription

"""

import os
import time
import math
import logging
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import GPy, GPyOpt

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD

from SMILESX import utils, augm, token, model, trainutils

def bayopt_run(smiles, prop, extra, train_val_idx, smiles_concat, tokens, max_length, check_smiles, augmentation, data_skew, hyper_bounds, hyper_opt, dense_depth, bo_rounds, bo_epochs, bo_runs, bayopt_vis, window_size, strategy, model_type, output_n_nodes, scale_output, pretrained_model=None):
    '''Bayesian optimization of hyperparameters.

    Parameters
    ----------
    smiles: np.array
        Input SMILES.
    prop: np.array
        Input property values.
    extra: np.array
        Additional data input.
    train_val_idx: list
        List of indices for training and validation data for the current fold.
    tokens: list
        List of tokens contained within the dataset.
    max_length: int
        Maximum length of SMILES in training and validation data.
    check_smiles: bool
        Whether to check SMILES validity with RDKit.
    augmentation: bool
        Whether to perform data augmentation during bayesian optimization process.
    data_skew: bool
        Whether the classes in the input data are imbalanced.
    hyper_bounds: dict
        A dictionary of bounds {"param":[bounds]}, where parameter `"param"` can be
        embedding, LSTM, time-distributed dense layer units, batch size or learning
        rate, and `[bounds]` is a list of possible values to be tested during
        Bayesian optimization for a given parameter.
    hyper_opt: dict
        A dictionary of bounds {"param":val}, where parameter `"param"` can be
        embedding, LSTM, time-distributed dense layer units, batch size or learning
        rate, and `val` is default value for a given parameter.
    dense_depth: int
        Number of additional dense layers to be appended after attention layer.
    bo_rounds: int
        Number of rounds to be used during Bayesian optimization.
    bo_epochs: int
        Number of epochs required for training within the optimization frame.
    bo_runs: int
        Number of training repetitions with random train/val split.
    bayopt_vis: bool
        Whether to show the learning curves for each tried hyperparameters combinations
    window_size: int
        Window size for running average.
    strategy:
        GPU memory growth strategy.
    model_type: str
        Type of the model to be used. Can be either 'regression', 'binary_classification', or 'multiclass_classification'.
    output_n_nodes: int
        Number of output nodes. (Default: 1 for regression and binary classification)
        It equals to n_class (number of possible classes per output label) for multiclass classification.
    scale_output: bool
        Whether to scale the output property values or not. For binary classification tasks, it is recommended not to scale 
        the categorical (e.g. 0, 1) output values. For regression tasks, this is preferable to guarantee quicker 
        training convergence.
    pretrained_model:
        Pretrained model in case of the transfer learning (`train_mode='finetune'`).
        (Default: None)
            
    Returns
    -------
    hyper_opt: dictdata_prop
        Dictionary with hyperparameters updated with optimized values
    '''
    # Get the logger for smooth logging
    logger = logging.getLogger()

    logging.info("*** Bayesian optimization ***")
    logging.info("")

    # Identify which parameters to optimize via Bayesian optimisation
    if not any(hyper_bounds.values()):
        logging.warning("ATTENTION! Bayesian optimisation is requested, but no bounds are given.")
        logging.info("")
        logging.warning("Specify at least one of the following:")
        logging.warning("      - embed_bounds")
        logging.warning("      - lstm_bounds")
        logging.warning("      - tdense_bounds")
        logging.warning("      - bs_bounds")
        logging.warning("      - lr_bounds")
        logging.info("")
        logging.warning("If no Bayesian optimisation is desired, set `bayopt_mode='off'`.")
        logging.info("")
        logging.warning("The SMILES-X execution is aborted.")
        raise utils.StopExecution

    bayopt_bounds = []
    logging.info('Bayesian optimisation is requested for:')
    for key in hyper_bounds.keys():
        if hyper_bounds[key] is not None:
            logging.info('      - {}'.format(key))
            # Setup GPyOpt bounds format
            bayopt_bounds.append({'name': key, 'type': 'discrete', 'domain': hyper_bounds[key]})
    logging.info('*Note: selected hyperparameters will be optimized simultaneously.')
    logging.info("")

    # The function to be optimized during Bayesian optimization
    # It is nested because GPyOpt optimizes all the passed parameters,
    # but we only need to optimize a part of architecture
    def bayopt_func(params):
        # Reverse for popping
        params = params.flatten().tolist()[::-1]
        logging.info('Model: {}'.format(params))

        # Setting up the requested parameters for the optimization
        if extra is not None:
            extra_dim = extra.shape[1]
        else:
            extra_dim = None

        hyper_bo = hyper_opt
        for key in hyper_bounds.keys():
            if hyper_bounds[key] is not None:
                hyper_bo[key] = params.pop()

        score_valids = []
        histories_train = []
        histories_val = []
        histories_val_avg = []
        for irun in range(bo_runs):
            # Preparing the data for optimization
            # Random train/val splitting for every run to assure better generalizability of the optimized parameters
            x_train, x_valid, extra_train, extra_valid, y_train, y_valid = utils.rand_split(smiles_input=smiles,
                                                                                            prop_input=prop,
                                                                                            extra_input=extra,
                                                                                            err_input=None,
                                                                                            train_val_idx=train_val_idx,
                                                                                            test_idx=None,
                                                                                            bayopt=True)
            # Scale the outputs
            if scale_output:
                y_train_scaled, y_valid_scaled, y_test_scaled, scaler = utils.robust_scaler(train=y_train,
                                                                                            valid=y_valid,
                                                                                            test=None,
                                                                                            file_name=None,
                                                                                            ifold=None)
            else:
                y_train_scaled, y_valid_scaled, y_test_scaled, scaler = y_train, y_valid, None, None
            # Check/augment the data if requested
            train_augm = augm.augmentation(x_train,
                                       train_val_idx,
                                       extra_train,
                                       y_train_scaled,
                                       check_smiles,
                                       augmentation)

            valid_augm = augm.augmentation(x_valid,
                                       train_val_idx,
                                       extra_valid,
                                       y_valid_scaled,
                                       check_smiles,
                                       augmentation)
            
            x_train_enum, extra_train_enum, y_train_enum, y_train_clean, x_train_enum_card, _ = train_augm
            x_valid_enum, extra_valid_enum, y_valid_enum, y_valid_clean, x_valid_enum_card, _ = valid_augm
            
            # Concatenate multiple SMILES into one via 'j' joint
            if smiles_concat:
                x_train_enum = utils.smiles_concat(x_train_enum)
                x_valid_enum = utils.smiles_concat(x_valid_enum)
                
            x_train_enum_tokens = token.get_tokens(x_train_enum)
            x_valid_enum_tokens = token.get_tokens(x_valid_enum)
            x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_train_enum_tokens,
                                                                max_length=max_length + 1,
                                                                vocab=tokens)
            x_valid_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list=x_valid_enum_tokens,
                                                                max_length=max_length + 1,
                                                                vocab=tokens)

            K.clear_session()
            #TODO(Guillaume): Check pretraining case
            if pretrained_model is not None:
                # Load the pretrained model
                model_train = pretrained_model.model_dic['Fold_{}'.format(ifold)][run]
                # Freeze encoding layers
                #TODO(Guillaume): Check if this is the best way to freeze the layers as layers' name may differ
                for layer in model_train.layers:
                    if layer.name in ['embedding', 'bidirectional', 'time_distributed']:
                        layer.trainable = False

                logging.info("Retrieved model summary:")
                model_train.summary(print_fn=logging.info)
                logging.info("\n")
            else:
                with strategy.scope():
                    model_opt = model.LSTMAttModel.create(input_tokens=max_length + 1,
                                                          extra_dim=extra_dim,
                                                          vocab_size=len(tokens),
                                                          embed_units=hyper_bo['Embedding'],
                                                          lstm_units=hyper_bo['LSTM'],
                                                          tdense_units=hyper_bo['TD dense'],
                                                          dense_depth=dense_depth, 
                                                          model_type=model_type, 
                                                          output_n_nodes=output_n_nodes)

            if model_type == 'regression':
                model_loss = 'mse'
                model_metrics = [metrics.mae, metrics.mse]
                hist_val_name = 'val_mean_squared_error'
                hist_train_name = 'mean_squared_error'
            else:
                if model_type == 'binary_classification':
                    model_loss = 'binary_crossentropy'
                    model_metrics = ['accuracy']
                elif model_type == 'multiclass_classification':
                    model_loss = 'sparse_categorical_crossentropy'
                    model_metrics = ['accuracy']

                if data_skew:
                    hist_train_name = 'precision_at_recall'
                    hist_val_name = 'val_precision_at_recall'
                    with strategy.scope():
                        model_metrics = [tf.keras.metrics.PrecisionAtRecall(0.5)]
                else:
                    hist_train_name = 'auc'
                    hist_val_name = 'val_auc'
                    with strategy.scope():
                        model_metrics = [tf.keras.metrics.AUC()]

            with strategy.scope():
                batch_size = int(hyper_bo['Batch size']) * strategy.num_replicas_in_sync
                batch_size_val = min(len(x_train_enum_tokens_tointvec), batch_size)
                custom_adam = Adam(learning_rate=math.pow(10,-float(hyper_bo['Learning rate'])))
                running_loss = trainutils.RunningAverageLoss(window_size=window_size, model_type=model_type, warm_up=int(bo_epochs/2), data_skew=data_skew)
                callbacks_list = [running_loss]
                if data_skew:
                    model_opt.compile(loss=trainutils.FocalLossCustom(alpha=0.2, gamma=2.0), optimizer=custom_adam, metrics=model_metrics)
                else:
                    model_opt.compile(loss=model_loss, optimizer=custom_adam, metrics=model_metrics)

                history = model_opt.fit_generator(generator=\
                                                  trainutils.DataSequence(x_train_enum_tokens_tointvec,
                                                                          extra_train_enum,
                                                                          y_train_enum,
                                                                          batch_size),
                                                  validation_data=\
                                                  trainutils.DataSequence(x_valid_enum_tokens_tointvec,
                                                                          extra_valid_enum,
                                                                          y_valid_enum,
                                                                          batch_size_val),
                                                  epochs=bo_epochs,
                                                  callbacks=callbacks_list,
                                                  shuffle=True,
                                                  initial_epoch=0,
                                                  verbose=0)
                histories_train.append(history.history[hist_train_name])
                histories_val.append(history.history[hist_val_name])
                histories_val_avg.append(running_loss.running_avg_val_loss)

        # Skip the first half of epochs during evaluation
        # Ignore the noisy burn-in period of training
        # Minimize the metric for regression problems and maximize it for classification

        histories_val_avg = np.array(histories_val_avg)
        mean_histories_val_avg = running_loss.inverse*histories_val_avg.mean(axis=0)
        
        score_valid = np.min(running_loss.inverse*mean_histories_val_avg)

        if math.isnan(score_valid): # treat diverging architectures (rare event)
            score_valid = math.inf

        logging.info('Average best validation score: {0:0.4f}'.format(score_valid))
        
        if bayopt_vis:
            histories_train = np.array(histories_train)
            histories_val = np.array(histories_val)
            

            fig, ax = plt.subplots(figsize=(5, 3))

            x = range(1, histories_train.shape[1]+1)
            sns.lineplot(x=x,
                         y=histories_train.mean(axis=0),
                         ax=ax,
                         label='Training Loss',
                         color='#3783AD',
                         linewidth=2.5)
            sns.lineplot(x=x,
                         y=histories_val.mean(axis=0),
                         ax=ax,
                         label='Running Validation Loss',
                         color='#F7A95E',
                         linewidth=2.5)
            
            sns.lineplot(x=x[int(bo_epochs/2)-1:],
                         y=histories_val_avg.mean(axis=0)[int(bo_epochs/2)-1:],
                         ax=ax,
                         label='Running Average Validation Loss',
                         color='#E06D00',
                         linewidth=2.5)
            
            sns.lineplot(x=x[:int(bo_epochs/2)],
                         y=histories_val_avg.mean(axis=0)[:int(bo_epochs/2)],
                         ax=ax,
                         color='#E06D00',
                         linewidth=2.5,
                         dashes=(2, 2))

            plt.fill_between(x,
                             histories_train.mean(axis=0) - histories_train.std(axis=0),
                             histories_train.mean(axis=0) + histories_train.std(axis=0),
                             color='#3783AD', alpha=0.2, linewidth=0.0)

            plt.fill_between(x,
                             histories_val.mean(axis=0) - histories_val.std(axis=0),
                             histories_val.mean(axis=0) + histories_val.std(axis=0),
                             color='#F7A95E', alpha=0.2, linewidth=0.0)
            
            plt.fill_between(x,
                             histories_val_avg.mean(axis=0) - histories_val_avg.std(axis=0),
                             histories_val_avg.mean(axis=0) + histories_val_avg.std(axis=0),
                             color='#E06D00', alpha=0.2, linewidth=0.0)

            ax.set_xlabel('Epochs', fontsize=14)
            ax.set_ylabel('Loss', fontsize=14)

            plt.legend()
            plt.show()

        return score_valid

    start_bo = time.time()

    logging.info("~~~~~")
    logging.info("Random initialization:")
    Bayes_opt = GPyOpt.methods.BayesianOptimization(f=bayopt_func,
                                                    domain=bayopt_bounds,
                                                    acquisition_type='EI',
                                                    acquisition_jitter=0.1,
                                                    initial_design_numdata=bo_rounds,
                                                    exact_feval=True,
                                                    normalize_Y=False,
                                                    num_cores=1)
    logging.info("~~~~~")
    logging.info("Optimization:")
    Bayes_opt.run_optimization(max_iter=bo_rounds)
    opt_params = Bayes_opt.x_opt.tolist()[::-1] # reverse the list for popping from head later
    for key in hyper_bounds.keys():
        if hyper_bounds[key] is not None:
            hyper_opt[key] = opt_params.pop()

    end_bo = time.time()
    elapsed_bo = end_bo - start_bo

    logging.info("")
    logging.info("*** Bayesian hyperparameters optimization is completed ***")
    logging.info("")
    logging.info("Bayesian optimisation duration: {}".format(str(datetime.timedelta(seconds=elapsed_bo))))
    logging.info("")

    return hyper_opt
##