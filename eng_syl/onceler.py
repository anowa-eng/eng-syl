from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Embedding, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import pickle
import numpy as np
import os


class Onceler:
    def __init__(self, input_size=11, e2i='e2i_onc.pkl', latent_dim=500, embed_dim=500, max_feat=61):
        self.e2i = e2i
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        path_e2i = os.path.join(self.this_dir, e2i)
        path_clean = os.path.join(self.this_dir, 'clean_onc.pkl')
        with open(path_clean, 'rb') as f:
            self.clean = pickle.load(f)
        with open(path_e2i, 'rb') as f:
            self.e2i = pickle.load(f)
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.max_feat = max_feat + 1  # include dim for padding value 0 (no corresponding index in dict)
        self.model = self.build_model()
        self.model.load_weights(os.path.join(self.this_dir, 'onceler_best_weights.h5'))

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_size,)))
        model.add(Embedding(self.max_feat, self.embed_dim))
        model.add(Bidirectional(LSTM(self.latent_dim, return_sequences=True, recurrent_dropout=0.4)))
        model.add(TimeDistributed(Dense(3)))
        model.add(Activation('softmax'))
        return model

    def ignore_class_accuracy(self, to_ignore=0):
        def ignore_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)

            ignore_mask = K.cast(K.not_equal(y_true_class, 0), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy

        return ignore_accuracy

    def fit(self, x_tr, y_tr, x_test, y_test, ep, batch_size, save_filename):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy', self.ignore_class_accuracy()])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        ck = ModelCheckpoint(filepath=save_filename, monitor='val_accuracy', verbose=1, save_best_only=True,
                             mode='max')
        callbacks = [es, ck]
        self.model.fit(x_tr, y_tr, epochs=ep, callbacks=callbacks, batch_size=batch_size,
                       validation_data=(x_test, y_test))

    def onc_split(self, word):
        if word in self.clean:
            return self.clean[word]
        else:
            inted = [self.e2i[c] for c in word.lower()]
            inted = pad_sequences([inted], maxlen=self.input_size, padding='post')[0]
            predicted = self.model.predict(np.array([inted]), verbose=0)[0]
            converted = self.to_ind(predicted)
            return self.insert_syl(word, converted)

    def to_ind(self, sequence):
        return [np.argmax(ind) for ind in sequence]

    def insert_syl(self, word, indexes):
        index_list = np.where(np.array(indexes) == 2)[0]
        word_array = list(word)
        for i, index in enumerate(index_list):
            word_array.insert(index + i + 1, '-')
        return ''.join(word_array)
