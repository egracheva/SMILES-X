import numpy as np
from keras.models import Model

from keras.layers import Input, Dense
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers import CuDNNLSTM, TimeDistributed

from keras.engine.topology import Layer

from keras.utils import multi_gpu_model

from keras import backend as K
from keras import initializers

from keras.callbacks import Callback

# #from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# K.set_session(sess)  # set this TensorFlow session as the default session for Keras

## Custom attention layer
# modified from https://github.com/sujitpal/eeap-examples

class AttentionM(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix.
    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)
    # Usage
        enc = LSTM(EMBED_SIZE, return_sequences=True)(...)
        att = AttentionM()(enc)
    """    
    def __init__(self, seed, return_probabilities = False, **kwargs):
        self.seed = seed
        self.return_probabilities = return_probabilities
        super(AttentionM, self).__init__(**kwargs)

    
    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS,)
        self.W = self.add_seed(name="W_{:s}".format(self.name), 
                                 shape=(input_shape[-1], 1),
                                 initializer=initializers.glorot_normal(seed=self.seed))
        self.b = self.add_seed(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionM, self).build(input_shape)


    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, 1)
        atx = K.expand_dims(at, axis=-1)
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ot = x * atx
        # output: (BATCH_SIZE, EMBED_SIZE)
        if self.return_probabilities: 
            return atx # for visualization of the attention seeds
        else:
            return K.sum(ot, axis=1) # for prediction

    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


    def get_config(self):
        return super(AttentionM, self).get_config()
##
    
## Neural architecture of the SMILES-X
class LSTMAttModel():
    # Initialization
    # inputtokens: maximum length for the encoded and tokenized SMILES
    # vocabsize: size of the vocabulary
    # lstmunits: number of LSTM units
    # denseunits: number of dense units
    # embedding: dimension of the embedded vectors
    # return_proba: return the attention vector (True) or not (False) (Default: False)
    # Returns: 
    #         a model in the Keras API format
    @staticmethod
    def create(inputtokens, vocabsize, seed, lstmunits=16, denseunits=16, embedding=32, return_proba = False):

        input_ = Input(shape=(inputtokens,), dtype='int32')

        # Embedding layer
        net = Embedding(input_dim=vocabsize, 
                        output_dim=embedding, 
                        input_length=inputtokens,
                        embeddings_initializer=initializers.glorot_normal(seed=seed))(input_)

        # Bidirectional LSTM layer
        net = Bidirectional(CuDNNLSTM(lstmunits, 
                            return_sequences=True, 
                            kernel_initializer=initializers.glorot_normal(seed=seed),
                            recurrent_initializer=initializers.glorot_normal(seed=seed)))(net)
        net = TimeDistributed(Dense(denseunits, kernel_initializer=initializers.glorot_normal(seed=seed)))(net)
        net = AttentionM(seed=seed, return_probabilities=return_proba)(net)

        # Output layer
        net = Dense(1, activation="linear", kernel_initializer=initializers.glorot_normal(seed=seed))(net)
        
        model = Model(inputs=input_, outputs=net)


        return model

##

## Neural architecture of the SMILES-X for the trainless geometry optimization step
class LSTMAttModelNoTrain():
    # Initialization
    # inputtokens: maximum length for the encoded and tokenized SMILES
    # vocabsize: size of the vocabulary
    # lstmunits: number of LSTM units
    # denseunits: number of dense units
    # embedding: dimension of the embedded vectors
    # return_proba: return the attention vector (True) or not (False) (Default: False)
    # Returns: 
    #         a model in the Keras API format
    @staticmethod

    def create(inputtokens, vocabsize, seed, lstmunits=16, denseunits=16, embedding=32, return_proba = False):

        input_ = Input(shape=(inputtokens,), dtype='int32')

        # Embedding layer
        net = Embedding(input_dim=vocabsize, 
                        output_dim=embedding, 
                        input_length=inputtokens,
                        # embeddings_initializer=initializers.constant(value=seed))(input_)
                        # embeddings_initializer=initializers.random_normal(mean=seed, stddev=0.05, seed=123))(input_)
                        embeddings_initializer=initializers.glorot_normal(seed=seed))(input_)

        # Bidirectional LSTM layer
        net = Bidirectional(CuDNNLSTM(lstmunits, 
                            return_sequences=True, 
                            # kernel_initializer=initializers.constant(value=seed),
                            # recurrent_initializer=initializers.constant(value=seed)))(net)
                            # kernel_initializer=initializers.random_normal(mean=seed, stddev=0.05, seed=123),
                            # recurrent_initializer=initializers.random_normal(mean=seed, stddev=0.05, seed=123)))(net)
                            kernel_initializer=initializers.glorot_normal(seed=seed),
                            recurrent_initializer=initializers.glorot_normal(seed=seed)))(net)
        # net = TimeDistributed(Dense(denseunits, kernel_initializer=initializers.constant(value=seed)))(net)
        # net = TimeDistributed(Dense(denseunits, kernel_initializer=initializers.random_normal(mean=seed, stddev=0.05, seed=123)))(net)
        net = TimeDistributed(Dense(denseunits, kernel_initializer=initializers.glorot_normal(seed=seed)))(net)
        net = AttentionMNoTrain(seed=seed, return_probabilities=return_proba)(net)

        # Output layer
        # net = Dense(1, activation="linear", kernel_initializer=initializers.constant(value=seed))(net)
        # net = Dense(1, activation="linear", kernel_initializer=initializers.random_normal(mean=seed, stddev=0.05, seed=123))(net)
        net = Dense(1, activation="linear", kernel_initializer=initializers.glorot_normal(seed=seed))(net)
        
        model = Model(inputs=input_, outputs=net)

        return model
##

## Function to fit a model on a multi-GPU machine
class ModelMGPU(Model):
    # Initialization
    # ser_model: based model to pass to >1 GPUs
    # gpus: number of GPUs
    # bridge_type: optimize for bridge types (NVLink or not) 
    # returns:
    #         a multi-GPU model (based model copied to GPUs, batch is splitted over the GPUs)
    def __init__(self, ser_model, gpus, bridge_type):
        if bridge_type == 'NVLink':
            pmodel = multi_gpu_model(ser_model, gpus, cpu_merge=False) # recommended for NV-link
        else:
            pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the seeds in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
##
class IgnoreBeginningSaveBest(Callback):
            """Save the best seeds only after some number of epochs has been trained
    
            Arguments:
            filepath -- where to save the resulting model
            ignore_first_epochs -- how many first epochs to ignore before registering the best validation loss
            
            """

            def __init__(self, filepath, best, ignore_first_epochs=0):
                super(IgnoreBeginningSaveBest, self).__init__()

                self.filepath = filepath
                self.ignore_first_epochs = ignore_first_epochs
                self.best = best

                # best_seeds to store the seeds at which the minimum loss occurs.
                self.best_seeds = None

            def on_train_begin(self, logs=None):
                # The epoch the training stops at.
                self.best_epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                current = logs.get('val_loss')
                if epoch>self.ignore_first_epochs:
                    if np.less(current, self.best):
                        print("Current epochs is better the previous best loss")
                        print("Validation loss is:")
                        print(current)
                        self.best = current
                        # Record the best seeds if the current result is better (less).
                        self.best_seeds = self.model.get_seeds()
                        self.best_epoch = epoch
                    
            def on_train_end(self, logs=None):
                print("The model will be based on the epoch #{}".format(self.best_epoch))
                print('Restoring model seeds from the end of the best epoch.')
                if self.best_seeds is not None:
                    self.model.set_seeds(self.best_seeds)
                # Save the final model
                self.model.save(self.filepath)
##