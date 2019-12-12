import os
import math
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


from rdkit import Chem
from rdkit.Chem import Draw

from keras.models import Model
from keras.models import load_model
from keras import metrics
from keras import backend as K
import tensorflow as tf

from SMILESX import utils, model, token, augm

##
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

## Interpretation of the SMILESX predictions
# data: provided data (numpy array of: (SMILES, property))
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for cross-validation
# k_fold_index: k-fold index to be used for visualization
# augmentation: SMILES's augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Interpretation/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# smiles_toviz: targeted SMILES to visualize (Default: 'CCC')
# font_size: font's size for writing SMILES tokens (Default: 15)
# font_rotation: font's orientation (Default: 'horizontal')
# returns:
#         The 1D and 2D attention maps 
#             The redder and darker the color is, 
#             the stronger is the attention on a given token. 
#         The temporal relative distance Tdist 
#             The closer to zero is the distance value, 
#             the closer is the temporary prediction on the SMILES fragment to the whole SMILES prediction.
def Interpretation(data, 
                   data_name, 
                   data_units = '',
                   k_fold_number = 10,
                   k_fold_index=0,
                   run_index = 0,
                   augmentation = False, 
                   outdir = "./data/", 
                   smiles_list_toviz = ['CCC'], 
                   font_size = 15, 
                   font_rotation = 'horizontal'):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    input_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    input_dir_run = input_dir+'fold_{}/run_{}/'.format(k_fold_index, run_index)
    save_dir = outdir+'Interpretation/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if the vocabulary for current dataset exists already
    if os.path.exists(input_dir+data_name+'_Vocabulary.txt'):
        tokens = token.get_vocab(input_dir+data_name+'_Vocabulary.txt')
    else:
        "Vocabulary file does not exist. Are you sure you have already trained the model?"
    vocab_size = len(tokens)

    # Add 'pad', 'unk' tokens to the existing list
    tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)

    # Transformation of tokenized SMILES to vector of intergers and vice-versa
    token_to_int = token.get_tokentoint(tokens)
    int_to_token = token.get_inttotoken(tokens)

    print("***SMILES_X Interpreter starts...***\n\n")
    np.random.seed(seed=123)

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

            np.savetxt(save_dir+'smiles_train.txt', np.asarray(x_train), newline="\n", fmt='%s')
            np.savetxt(save_dir+'smiles_valid.txt', np.asarray(x_valid), newline="\n", fmt='%s')
            np.savetxt(save_dir+'smiles_test.txt', np.asarray(x_test), newline="\n", fmt='%s')
            
            smiles_toviz_x = []
            smiles_toviz_y = []
            for smiles_toviz in smiles_list_toviz:
                mol_toviz = Chem.MolFromSmiles(smiles_toviz)
                if mol_toviz != None:
                    smiles_toviz_can = Chem.MolToSmiles(mol_toviz)
                    smiles_toviz_x.append(smiles_toviz_can)
                    if smiles_toviz_can in data.smiles.values:
                        smiles_toviz_y.append(data[data.smiles==smiles_toviz_can].iloc[:,1].values)
                    else:
                        smiles_toviz_y.append(np.nan)
                else:
                    print("***Process of visualization automatically aborted!***")
                    print("The smiles_toviz is incorrect and cannot be canonicalized by RDKit.")
                    return

            smiles_toviz_x = np.array(smiles_toviz_x)
            smiles_toviz_y = np.array(smiles_toviz_y)

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
            
            smiles_toviz_x_enum, smiles_toviz_x_enum_card, smiles_toviz_y_enum = \
            augm.Augmentation(smiles_toviz_x, smiles_toviz_y, canon=canonical, rotate=rotation)

            print("Enumerated SMILES:\n\tTraining set: {}\n\tValidation set: {}\n\tTest set: {}\n".\
            format(x_train_enum.shape[0], x_valid_enum.shape[0], x_test_enum.shape[0]))

            print("***Tokenization of SMILES.***\n")
            # Tokenize SMILES per dataset
            x_train_enum_tokens = token.get_tokens(x_train_enum)
            x_valid_enum_tokens = token.get_tokens(x_valid_enum)
            x_test_enum_tokens = token.get_tokens(x_test_enum)
            

            smiles_toviz_x_enum_tokens = token.get_tokens(smiles_toviz_x_enum)

            print("Examples of tokenized SMILES from a training set:\n{}\n".\
            format(x_train_enum_tokens[:3]))

            train_unique_tokens = list(token.extract_vocab(x_train_enum_tokens))
            print(train_unique_tokens)
            print("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
            train_unique_tokens.insert(0,'pad')
            
            print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))

            # Maximum of length of SMILES to process
            all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens
            max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
            print("Maximum length of tokenized SMILES: {} tokens\n".format(max_length))

            # Best architecture to visualize from
            model_topredict = load_model(input_dir_run+data_name+'_model.best_fold_'+str(k_fold_index)+'_run_'+str(run_index)+'.hdf5',
                                   custom_objects={'AttentionM': model.AttentionM(seed=0)})

            best_arch = [model_topredict.layers[2].output_shape[-1]/2, 
                         model_topredict.layers[3].output_shape[-1], 
                         model_topredict.layers[1].output_shape[-1]]

            # Architecture to return attention weights
            model_att = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                  vocabsize = vocab_size,
                                                  seed = 0,
                                                  lstmunits= int(best_arch[0]), 
                                                  denseunits = int(best_arch[1]), 
                                                  embedding = int(best_arch[2]), 
                                                  return_proba = True)

            print("Best model summary:\n")
            print(model_att.summary())
            print("\n")

            print("***Interpretation from the best model.***\n")
            model_att.load_weights(input_dir_run+data_name+'_model.best_fold_'+str(k_fold_index)+'_run_'+str(run_index)+'.hdf5')

            smiles_toviz_x_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list= smiles_toviz_x_enum_tokens, 
                                                                       max_length = max_length+1,
                                                                       vocab = tokens)
            
            intermediate_layer_model = Model(inputs=model_att.input,
                                             outputs=model_att.layers[-2].output)
            intermediate_output = intermediate_layer_model.predict(smiles_toviz_x_enum_tokens_tointvec)
            
            smiles_toviz_x_card_cumsum_viz = np.cumsum(smiles_toviz_x_enum_card)
            smiles_toviz_x_card_cumsum_shift_viz = shift(smiles_toviz_x_card_cumsum_viz, 1, cval=0)

            for mols_id in range(len(smiles_list_toviz)):
                ienumcard = smiles_toviz_x_card_cumsum_shift_viz[mols_id]
            
                smiles_len_tmp = len(smiles_toviz_x_enum_tokens[ienumcard])
                intermediate_output_tmp = intermediate_output[ienumcard,-smiles_len_tmp+1:-1].flatten().reshape(1,-1)
                max_intermediate_output_tmp = np.max(intermediate_output_tmp)

                plt.matshow(intermediate_output_tmp, 
                            cmap='Reds')
                plt.tick_params(axis='x', bottom = False)
                plt.xticks([ix for ix in range(smiles_len_tmp-2)])
                plt.xticks(range(smiles_len_tmp-2), 
                           [int_to_token[iint].replace('pad','') \
                            for iint in smiles_toviz_x_enum_tokens_tointvec[ienumcard,-smiles_len_tmp+1:-1]], 
                           fontsize = font_size, 
                           rotation = font_rotation)
                plt.yticks([])
                plt.savefig('{}Interpretation_1D_{}_fold_{}_run_{}_mol_{}.png'.format(save_dir, data_name, k_fold_index, run_index, mols_id), bbox_inches='tight')
            
                y_pred_test_tmp = model_topredict.predict(smiles_toviz_x_enum_tokens_tointvec[ienumcard].reshape(1,-1))[0,0]
                y_test_tmp = smiles_toviz_y_enum[ienumcard, 0]
                y_pred_test_tmp_scaled = scaler.inverse_transform(y_pred_test_tmp.reshape(1, -1))[0][0]
                # Getting automatic precision computed
                if np.log10(y_pred_test_tmp_scaled)>0:
                    if np.log10(y_pred_test_tmp_scaled)<3:
                        precision = 1+(3-np.floor(np.log10(y_pred_test_tmp_scaled)))/10
                    else:
                        precision = 1.0
                else:
                    precision = (np.abs(np.floor(np.log10(y_pred_test_tmp_scaled)))+3)/10
                    
                if not np.isnan(y_test_tmp):
                    print("True value: {0:{2}f} Predicted: {1:{2}f}".format(y_test_tmp,
                                                                            y_pred_test_tmp_scaled,
                                                                            precision))
                else:
                    print("Predicted: {0:{1}f}".format(y_pred_test_tmp_scaled,
                                                       precision))

                smiles_tmp = smiles_toviz_x_enum[ienumcard]
                mol_tmp = Chem.MolFromSmiles(smiles_tmp)
                smiles_len_tmp = len(smiles_toviz_x_enum_tokens[ienumcard])
                mol_df_tmp = pd.DataFrame([smiles_toviz_x_enum_tokens[ienumcard][1:-1],intermediate_output[ienumcard].\
                                          flatten().\
                                          tolist()[-smiles_len_tmp+1:-1]]).transpose()
                bond = ['-','=','#','$','/','\\','.','(',')']
                mol_df_tmp = mol_df_tmp[~mol_df_tmp.iloc[:,0].isin(bond)]
                mol_df_tmp = mol_df_tmp[[not itoken.isdigit() for itoken in mol_df_tmp.iloc[:,0].values.tolist()]]

                minmaxscaler = MinMaxScaler(feature_range=(0,1))
                norm_weights = minmaxscaler.fit_transform(mol_df_tmp.iloc[:,1].values.reshape(-1,1)).flatten().tolist()
                if not np.isnan(y_test_tmp):
                    fig = GetSimilarityMapFromWeights(mol=mol_tmp, 
                                                     size = (250,250), 
                                                     scale=-1,  
                                                     sigma=0.05,
                                                     weights=norm_weights,
                                                     pred_val = "{0:{1}f}".format(y_pred_test_tmp_scaled, precision),
                                                     true_val = "{0:{1}f}".format(y_test_tmp, precision),
                                                     colorMap='Reds', 
                                                     contourLines = 10,
                                                     alpha = 0.25)
                else:
                    fig = GetSimilarityMapFromWeights(mol=mol_tmp, 
                                                     size = (250,250), 
                                                     scale=-1,  
                                                     sigma=0.05,
                                                     weights=norm_weights,
                                                     pred_val = "{0:{1}f}".format(y_pred_test_tmp_scaled, precision),
                                                     colorMap='Reds', 
                                                     contourLines = 10,
                                                     alpha = 0.25)
                fig.savefig('{}Interpretation_2D_{}_fold_{}_run_{}_mol_{}.png'.format(save_dir, data_name, k_fold_index, run_index, mols_id), bbox_inches='tight')                
                
                smiles_len_tmp = len(smiles_toviz_x_enum_tokens[ienumcard])
                diff_topred_list = list()
                diff_totrue_list = list()
                for csubsmiles in range(1,smiles_len_tmp):
                    isubsmiles = smiles_toviz_x_enum_tokens[ienumcard][:csubsmiles]+[' ']
                    isubsmiles_tointvec= token.int_vec_encode(tokenized_smiles_list = [isubsmiles], 
                                                              max_length = max_length+1, 
                                                              vocab = tokens)
                    predict_prop_tmp = model_topredict.predict(isubsmiles_tointvec)[0,0]
                    diff_topred_tmp = (predict_prop_tmp-y_pred_test_tmp)/np.abs(y_pred_test_tmp)
                    diff_topred_list.append(diff_topred_tmp)
                    diff_totrue_tmp = (predict_prop_tmp-y_test_tmp)/np.abs(y_test_tmp)
                    diff_totrue_list.append(diff_totrue_tmp)
                max_diff_topred_tmp = np.max(diff_topred_list)
                max_diff_totrue_tmp = np.max(diff_totrue_list)

                plt.figure(figsize=(15,7))
                markers, stemlines, baseline = plt.stem([ix for ix in range(smiles_len_tmp-1)], 
                                                        diff_topred_list, 
                                                        'k.-', 
                                                         use_line_collection=True)
                plt.setp(baseline, color='k', linewidth=2, linestyle='--')
                plt.setp(markers, linewidth=1, marker='o', markersize=10, markeredgecolor = 'black')
                plt.setp(stemlines, color = 'k', linewidth=0.5, linestyle='-')
                plt.xticks(range(smiles_len_tmp-1), 
                           smiles_toviz_x_enum_tokens[ienumcard][:-1],
                           fontsize = font_size, 
                           rotation = font_rotation)
                plt.yticks(fontsize = 20)
                plt.ylabel('Temporal relative distance', fontsize = 25, labelpad = 15)
                plt.savefig('{}Interpretation_temporal_{}_fold_{}_run_{}_mol_{}.png'.format(save_dir, data_name, k_fold_index, run_index, mols_id), bbox_inches='tight')
            ifold +=1
        else:
            ifold += 1
##


## Attention weights depiction
# from https://github.com/rdkit/rdkit/blob/24f1737839c9302489cadc473d8d9196ad9187b4/rdkit/Chem/Draw/SimilarityMaps.py
# returns:
#         a similarity map for a molecule given the attention weights
def GetSimilarityMapFromWeights(mol, weights, pred_val, true_val=None, colorMap=None, scale=-1, size=(250, 250),
                                sigma=None, coordScale=1.5, step=0.01, colors='k', contourLines=10,
                                alpha=0.5, **kwargs):
    """
    Generates the similarity map for a molecule given the atomic weights.
    Parameters:
    mol -- the molecule of interest
    colorMap -- the matplotlib color map scheme, default is custom PiWG color map
    scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                          scale = double -> this is the maximum scale
    size -- the size of the figure
    sigma -- the sigma for the Gaussians
    coordScale -- scaling factor for the coordinates
    step -- the step for calcAtomGaussian
    colors -- color of the contour lines
    contourLines -- if integer number N: N contour lines are drawn
                    if list(numbers): contour lines at these numbers are drawn
    alpha -- the alpha blending value for the contour lines
    kwargs -- additional arguments for drawing
    """
    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")
    fig = Draw.MolToMPL(mol, coordScale=coordScale, size=size, **kwargs)
    ax = fig.gca()

    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = 0.3 * math.sqrt(
                    sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i])**2 for i in range(2)]))
        else:
            sigma = 0.3 * math.sqrt(sum([(mol._atomPs[0][i] - mol._atomPs[1][i])**2 for i in range(2)]))
        sigma = round(sigma, 2)
    x, y, z = Draw.calcAtomGaussians(mol, sigma, weights=weights, step=step)
    # scaling
    if scale <= 0.0:
        maxScale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
        minScale = min(math.fabs(np.min(z)), math.fabs(np.max(z)))
    else:
        maxScale = scale
    
    # fig.axes[0].imshow(z, cmap=colorMap, interpolation='bilinear', origin='lower',
    #                  extent=(0, 1, 0, 1), vmin=minScale, vmax=maxScale)
    ax.imshow(z, cmap=colorMap, interpolation='bilinear', origin='lower',
                     extent=(0, 1, 0, 1), vmin=minScale, vmax=maxScale)
    # contour lines
    # only draw them when at least one weight is not zero
    if len([w for w in weights if w != 0.0]):
        contourset = ax.contour(x, y, z, contourLines, colors=colors, alpha=alpha, **kwargs)
        for j, c in enumerate(contourset.collections):
            if contourset.levels[j] == 0.0:
                c.set_linewidth(0.0)
            elif contourset.levels[j] < 0:
                c.set_dashes([(0, (3.0, 3.0))])
    if true_val:
        ax.text(0.97, 0.02, "Predicted value: "+pred_val,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=12)
        ax.text(0.97, 0.06, "True value: "+true_val,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=12)
    else:
        ax.text(0.97, 0.02, "Predicted value: "+pred_val,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=12)
    ax.set_axis_off()
    return fig
##
