from keras.preprocessing import sequence
import keras

def pad_seq(sequence_list, maxlen):
    return sequence.pad_sequences(sequence_list, maxlen)


def get_one_hot(labels_list, num_classes):
    return keras.utils.np_utils.to_categorical(labels_list, num_classes)

def get_adam_optim(lr=0.001):
    return keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0)

def load_model(filepath):
    return keras.models.load_model(filepath)

def get_model_checkpoint(model_path, verbose=1, save_best_only=True):
    return keras.callbacks.ModelCheckpoint(filepath=model_path,
                                           verbose=verbose,
                                           save_best_only=save_best_only)


def get_early_stopping_cbk(monitor='val_loss', patience=5):
    return keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)