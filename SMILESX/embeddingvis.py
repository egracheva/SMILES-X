import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras import metrics

from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import KFold

from itertools import cycle
from adjustText import adjust_text

from SMILESX import model, utils, token, augm

## Visualization of the Embedding layer 
# data: provided data (numpy array of: (SMILES, property))
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for cross-validation
# k_fold_index: k-fold index to be used for visualization
# augmentation: SMILES's augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Embedding_Vis/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# affinity_propn: Affinity propagation tagging (Default: True)
# returns:
#         PCA visualization of a representation of SMILES tokens from the embedding layer

def Embedding_Vis(data, 
                  data_name, 
                  data_units = '',
                  k_fold_number = 8,
                  k_fold_index = 0,
                  run_index = 0,
                  augmentation = False, 
                  outdir = "./data/", 
                  affinity_propn = True, 
                  verbose = 0):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    input_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    save_dir = outdir+'Embedding_Vis/'+'{}/{}/'.format(data_name,p_dir_temp)
    input_dir_run = input_dir+'fold_{}/run_{}/'.format(k_fold_index, run_index+1)
    os.makedirs(input_dir_run, exist_ok=True)
    
    print("***SMILES_X for embedding visualization starts...***\n\n")
    np.random.seed(seed=123)
    # The following line is not necessary for the code, but as soon as I used the SMILES-X for 
    # LTE prediction with this code, to keep the kfold split the same,
    # I should preserve this line so far
    seed_list = np.random.randint(int(1e6), size = k_fold_number).tolist()

    # Train/validation/test data splitting - 80/10/10 % at random with diff. seeds for k_fold_number times
    kfold = KFold(k_fold_number, shuffle = True)
    ifold = 0
    for train_val_idx, test_idx in kfold.split(data.smiles):
        if ifold == k_fold_index:
            print("******")
            print("***Fold #{} initiated...***".format(k_fold_index))
            print("******")

            print("***Sampling and splitting of the dataset.***\n")
            # Reproducing the data split of the requested fold (k_fold_index)
            x_train, x_valid, x_test, y_train, y_valid, y_test, y_err, scaler = \
            utils.random_split(smiles_input=data.smiles,
                               prop_input=np.array(data.iloc[:,1]),
                               err_input=np.array(data.iloc[:,2]),
                               train_val_idx=train_val_idx,
                               test_idx=test_idx,                                                         
                               scaling = True)
          
            # data augmentation or not
            if augmentation == True:
                print("***Data augmentation.***\n")
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

            train_unique_tokens = list(token.extract_vocab(x_train_enum_tokens))
            print(train_unique_tokens)
            print("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
            train_unique_tokens.insert(0,'pad')
            
            # Tokens as a list
            tokens = token.get_vocab(input_dir+data_name+'_tokens_set_fold_'+str(k_fold_index)+'.txt')
            # Add 'pad', 'unk' tokens to the existing list
            tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)
            
            print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))

            # Maximum of length of SMILES to process
            max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
            print("Maximum length of tokenized SMILES: {} tokens (termination spaces included)\n".format(max_length))

            # Transformation of tokenized SMILES to vector of intergers and vice-versa
            token_to_int = token.get_tokentoint(tokens)
            int_to_token = token.get_inttotoken(tokens)

            # Gives me an init error. Fixed the get_config, but need to retrain to make it work.
            # model_train = load_model(input_dir_run+data_name+'_model.best_fold_'+str(k_fold_index)+'_run_'+str(run_index)+'.hdf5',
            #                        custom_objects={'AttentionMNotrain': model.AttentionM(seed=5)})
            # So far will just rebuild the model and load the weights

            model_train = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                    vocabsize = vocab_size,
                                                    seed=0,
                                                    lstmunits= int(16), 
                                                    denseunits = int(16), 
                                                    embedding = int(32)
                                                    )
            model_train.load_weights(input_dir_run+data_name+'_model.best_fold_'+str(k_fold_index)+'_run_'+str(run_index)+'.hdf5')

            print("Chosen model summary:\n")
            print(model_train.summary())
            print("\n")

            print("***Embedding of the individual tokens from the chosen model.***\n")
            #model_train.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

            model_embed_weights = model_train.layers[1].get_weights()[0]
            #print(model_embed_weights.shape)
            #tsne = TSNE(perplexity=30, early_exaggeration=120 , n_components=2, random_state=123, verbose=0)
            pca = PCA(n_components=2, random_state=123)
            transformed_weights = pca.fit_transform(model_embed_weights)
            #transformed_weights = tsne.fit_transform(model_embed_weights)    
            
            f = plt.figure(figsize=(9, 9))
            ax = plt.subplot(aspect='equal')
            
            if affinity_propn:
                # Compute Affinity Propagation
                af = AffinityPropagation().fit(model_embed_weights)
                cluster_centers_indices = af.cluster_centers_indices_
                labels = af.labels_
                n_clusters_ = len(cluster_centers_indices)
                # Plot it
                colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
                for k, col in zip(range(n_clusters_), colors):
                    class_members = np.where(np.array(labels == k) == True)[0].tolist()
                    for ilabpt in class_members:
                        alpha_tmp = 0.5 if tokens[ilabpt] in train_unique_tokens else 0.5
                        line_tmp = 1 if tokens[ilabpt] in train_unique_tokens else 5
                        marker_tmp = 'o' if tokens[ilabpt] in train_unique_tokens else 'x'
                        edge_color_tmp = 'black' if tokens[ilabpt] in train_unique_tokens else col
                        ax.plot(transformed_weights[ilabpt, 0], 
                                transformed_weights[ilabpt, 1], col, 
                                marker=marker_tmp, markeredgecolor = edge_color_tmp, markeredgewidth=line_tmp, 
                                alpha=alpha_tmp, markersize=10)
            else:
                # Black and white plot
                for ilabpt in range(vocab_size):
                    alpha_tmp = 0.5 if tokens[ilabpt] in train_unique_tokens else 0.2
                    size_tmp = 40 if tokens[ilabpt] in train_unique_tokens else 20
                    ax.scatter(transformed_weights[ilabpt,0], transformed_weights[ilabpt,1], 
                               lw=1, s=size_tmp, facecolor='black', marker='o', alpha=alpha_tmp)
            
            annotations = []
            weight_tmp = 'bold'
            ilabpt = 0
            for ilabpt, (x_i, y_i) in enumerate(zip(transformed_weights[:,0].tolist(), 
                                                    transformed_weights[:,1].tolist())):
                weight_tmp = 'black' if tokens[ilabpt] in train_unique_tokens else 'normal'
                tokens_tmp = tokens[ilabpt]
                if tokens_tmp == ' ':
                    tokens_tmp = 'space'
                elif tokens_tmp == '.':
                    tokens_tmp = 'dot'
                annotations.append(plt.text(x_i,y_i, tokens_tmp, fontsize=12, weight=weight_tmp))
            adjust_text(annotations,
                        x=transformed_weights[:,0].tolist(),y=transformed_weights[:,1].tolist(), 
                        arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
            
            plt.xticks([])
            plt.yticks([])
            ax.axis('tight')
            
            plt.savefig(save_dir+'Visualization_'+data_name+'_Embedding_fold_'+str(k_fold_index)+'.png', bbox_inches='tight')
            plt.show()
            ifold += 1
        else:
            ifold += 1

##
