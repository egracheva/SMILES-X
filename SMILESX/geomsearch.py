def findBestArch(data, 
                 data_name, 
                 bayopt_bounds,
                 geom_bounds,
                 weight_range,
                 n_opt_runs = 5,
                 k_fold_number = 5, 
                 augmentation = True, 
                 outdir = "./data/", 
                 bayopt_n_epochs = 20,
                 bayopt_n_rounds = 50,
                 bayopt_it_factor = 1,                          
                 n_gpus = 1,
                 bridge_type = 'NVLink')
    best_geoms = []
    for ifold in range(k_fold_number):
        
        print("******")
        print("***Fold #{} initiated...***".format(ifold))
        print("******")
        
        print("***Sampling and splitting of the dataset.***\n")
        x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = \
        utils.random_split(smiles_input=data.smiles, 
                           prop_input=np.array(data.iloc[:,1]), 
                           random_state=seed_list[ifold],
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

        # Operate the first step of the optimization:
        # geometry optimization (number of units in LSTM and Dense layers)
        def create_mod_geom(params, weight_range):
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
            for i, weight in enumerate(weight_range):
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
            best_weight = weight_range[np.argmin(pred_scores)]
            n_nodes = model_geom.count_params()
            return [mean_score, sigma_score, best_score, best_weight, n_nodes]

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
                  scores.append([n_lstm, n_dense, n_embed] + create_mod_geom([n_lstm, n_dense, n_embed], weight_range))
        scores = np.array(scores)
        ## Multistage sorting procedure
        ## Firstly, sort based on mean score and best score over weights
        points = scores[:, [3, 4, 7]].tolist()
        sort_ind = pg.sort_population_mo(points)
        print("Unordered scores")
        print(pd.DataFrame(scores))
        print("Re-ordered scores")
        print(pd.DataFrame(scores[sort_ind]))

        # Select the best geometry for further learning rate and batch size optimisation
        selected_geom = scores[sort_ind[0], :3]
        best_weight = scores[sort_ind[0], 6]
        print("The best untrained RMSE is:")
        print(scores[sort_ind[0], 5])
        print("Which is achieved using the weight of {}".format(best_weight))
        print("\nThe selected geometry for current fold is:\n\tLSTM units: {}\n\tDense units: {}\n\tEmbedding {}".\
              format(selected_geom[0], selected_geom[1], selected_geom[2]))
        selected_geoms.append(selected_geom)


    # Find the best geometry: select the most popular geometry among the folds
    mostCommonGeom = Counter(tuple(geom) for geom in selected_geoms).most_common(1)
    mostCommonGeom = list(mostCommonGeom)
    print("\nThe best selected geometry for current fold is:\n\tLSTM units: {}\n\tDense units: {}\n\tEmbedding {}".\
                  format(mostCommonGeom[0], mostCommonGeom[1], mostCommonGeom[2]))

    # Finding the best learning rate, and batch size for a random split 
    # through Bayesiam optimisation
    print("***Bayesian Optimization of the training hyperparameters.***\n")
    def create_mod(params):
        print('Model: {}'.format(params))
        # Run the model several times (here: 5) to get rid of the noise effect
        for run in range(0, n_opt_runs):
            K.clear_session()

            if n_gpus > 1:
                if bridge_type == 'NVLink':
                    model_opt = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                          vocabsize = vocab_size, 
                                                          weight=best_weight,
                                                          lstmunits=int(best_geom[0]), 
                                                          denseunits = int(best_geom[1]), 
                                                          embedding = int(best_geom[2]))
                else:
                    with tf.device('/cpu'): # necessary to multi-GPU scaling
                        model_opt = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                              vocabsize = vocab_size,
                                                              weight=best_weight,
                                                              lstmunits=int(best_geom[0]), 
                                                              denseunits = int(best_geom[1]), 
                                                              embedding = int(best_geom[2]))
                        
                multi_model = model.ModelMGPU(model_opt, gpus=n_gpus, bridge_type=bridge_type)
            else: # single GPU

                model_opt = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                      vocabsize = vocab_size,
                                                      weight=best_weight,
                                                      lstmunits=int(best_geom[0]), 
                                                      denseunits = int(best_geom[1]), 
                                                      embedding = int(best_geom[2]))
                
                multi_model = model_opt

            batch_size = int(params[:,0][0])
            custom_adam = Adam(lr=math.pow(10,-float(params[:,1][0])))
            multi_model.compile(loss='mse', optimizer=custom_adam, metrics=[metrics.mae,metrics.mse])
            # The same training and validation data is used as for the last performed split
            # during the geometry search
            history = multi_model.fit_generator(generator = DataSequence(x_train_enum_tokens,
                                                                         vocab = tokens, 
                                                                         max_length = max_length, 
                                                                         props_set = y_train_enum, 
                                                                         batch_size = batch_size), 
                                                                         steps_per_epoch = math.ceil(len(x_train_enum_tokens)/batch_size)//bayopt_it_factor, 
                                                validation_data = DataSequence(x_valid_enum_tokens,
                                                                               vocab = tokens, 
                                                                               max_length = max_length, 
                                                                               props_set = y_valid_enum, 
                                                                               batch_size = min(len(x_valid_enum_tokens), batch_size)),
                                                validation_steps = math.ceil(len(x_valid_enum_tokens)/min(len(x_valid_enum_tokens), batch_size))//bayopt_it_factor, 
                                                epochs = bayopt_n_epochs, 
                                                shuffle = True,
                                                initial_epoch = 0, 
                                                verbose = 0)

            best_epoch = np.argmin(history.history['val_loss'])
            # mae_valid = history.history['val_mean_absolute_error'][best_epoch]
            mse_valid = history.history['val_mean_squared_error'][best_epoch]
            if math.isnan(mse_valid): # discard diverging architectures (rare event)
                # mae_valid = math.inf
                mse_valid = math.inf
            mseValidRuns.append(mse_valid)
        meanMseValid = np.mean(mseValidRuns)
        print('RMSE: {1:0.4f}'.format(meanMseValid))

        return meanMseValid

    print("Random initialization:\n")
    Bayes_opt = GPyOpt.methods.BayesianOptimization(f=create_mod, 
                                                    domain=bayopt_bounds, 
                                                    acquisition_type = 'EI',
                                                    initial_design_numdata = bayopt_n_rounds,
                                                    exact_feval = False,
                                                    normalize_Y = True,
                                                    num_cores = multiprocessing.cpu_count()-1)
    print("Optimization:\n")
    Bayes_opt.run_optimization(max_iter=bayopt_n_rounds)
    best_arch = best_geom.tolist() + Bayes_opt.x_opt.tolist()      

    print("\nThe architecture for this datatset is:\n\tLSTM units: {}\n\tDense units: {}\n\tEmbedding dimensions {}".\
         format(int(best_arch[0]), int(best_arch[1]), int(best_arch[2])))
    print("\tBatch size: {0:}\n\tLearning rate: 10^-({1:.1f})\n".format(int(best_arch[3]), float(best_arch[4])))
    return best_arch