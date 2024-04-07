import os
import pickle
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Embedding, Activation, GRU, ZeroPadding1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

class onc_to_phon:
    def __init__(self, e2i='op_e2i.pkl', d2i='op_d2i.pkl', input_size=19, latent_dim=128, embed_dim=500):
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        path_e2i = os.path.join(self.this_dir, e2i)
        path_d2i = os.path.join(self.this_dir, d2i)
        with open(path_d2i, 'rb') as f:
            internal_d2i = pickle.load(f)
        self.i2d = {value: key for key, value in internal_d2i.items()}
        with open(path_e2i, 'rb') as f:
            self.e2i = pickle.load(f)
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.max_feat_e = len(self.e2i) + 1
        self.max_feat_d = len(internal_d2i) + 1
        self.model = self.build_model()
        self.model.load_weights(os.path.join(self.this_dir, 'op_best_weights.h5'))

    def build_model(self):
        ortho_inputs = Input(shape=(self.input_size,))
        x = Embedding(self.max_feat_e, self.embed_dim)(ortho_inputs)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout=0.2,
                              activity_regularizer=regularizers.l2(1e-5)))(x)
        x = Bidirectional(GRU(self.latent_dim, return_sequences=True, recurrent_dropout=0.2,
                              activity_regularizer=regularizers.l2(1e-5)))(x)
        x = ZeroPadding1D(padding=(0, self.input_size))(x)
        z = TimeDistributed(Dense(self.max_feat_d))(x)
        z = Activation('softmax')(z)

        model = Model(inputs=[ortho_inputs], outputs=z)

        return model

    def ipafy(self, word):
        inted_ortho = [self.e2i[c] for c in word]
        inted_ortho = pad_sequences([inted_ortho], maxlen=self.input_size, padding='post')[0]
        predicted = self.model.predict(np.array([inted_ortho]), verbose=0)[0]
        indexes = self.to_ind(predicted)
        converted = [self.i2d[x] for x in indexes if x != 0]
        return converted

    def to_ind(self, sequence):
        return [np.argmax(ind) for ind in sequence]
