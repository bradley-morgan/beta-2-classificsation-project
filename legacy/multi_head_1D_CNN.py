from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from tools.DatasetCompiler import DatasetCompiler
import numpy as np
from tools.anonymousClass import Obj
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dropout, Dense, Input, concatenate, GlobalMaxPool1D
from tqdm import tqdm
import matplotlib as plt
import seaborn as sns

# TODO Make 1D cnn
# TODO Create cross validation loop with MCC scoring the test fold
# TODO Handle Data imbalances
# TODO Try Creating an Embedding One hot to vec approuch folllowed by Random Forest or PCA

config = Obj(
    src='../data/processed/non-filtered/lrg_clean_data.pickle',
    k_folds=2,
    n_repeats=1,
    epochs=55,
    batch_size=128
)


def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def cnn_head(input, filters, kernel_size):
    output = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(input)
    output = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(output)
    # output = Conv1D(filters=filters*3, kernel_size=kernel_size, activation='relu')(output)
    output = GlobalMaxPool1D()(output)
    return output


# Load Data_set
data = DatasetCompiler.load_from_local(config.src)

#Balance Data or set Class Weights
vals, counts = np.unique(data.y_train, return_counts=True)
scale_pos_weight=int(np.round(max(counts) / min(counts), decimals=0)),

#Cross validation

scores = []
training_hist = []
val_history = []
show_summary = True
with tqdm(total=config.n_repeats*config.k_folds, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
          desc='n_repeats') as progress_bar:
    for r in range(config.n_repeats):
        # setup stratified K-fold cross validation
        cv = StratifiedKFold(n_splits=config.k_folds, shuffle=True)
        for train_idx, test_idx in cv.split(data.x_train, data.y_train):
            # extract hold out test set
            train_x, val_x = data.x_train[train_idx], data.x_train[test_idx]
            train_y, val_y = data.y_train[train_idx], data.y_train[test_idx]

            train_x = np.expand_dims(train_x, axis=2)
            train_y = np.expand_dims(train_y, axis=1)
            val_x = np.expand_dims(val_x, axis=2)
            val_y = np.expand_dims(val_y, axis=1)


            # Fit and Validate models
            # model.add(Dense(136, activation='relu'))
            inputs = Input(shape=(train_x.shape[1], 1))
            cnn_head2 = cnn_head(inputs, 32, 8)
            cnn_head3 = cnn_head(inputs, 128, 16)

            merge = concatenate(inputs=[cnn_head2, cnn_head3])
            x = Dropout(rate=0.1)(merge)
            x = Dense(1000, activation='relu')(x)
            x = Dropout(rate=0.1)(x)
            x = Dense(450, activation='relu')(x)
            outputs = Dense(1, activation='sigmoid')(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[matthews_correlation_coefficient])
            if show_summary:
                model.summary()
                show_summary = False
            history = model.fit(x=train_x, y=train_y, epochs=config.epochs, batch_size=config.batch_size, verbose=1, validation_split=0.1, class_weight={0:4, 1:1})
            _, mcc = model.evaluate(x=val_x, y=val_y, batch_size=config.batch_size, verbose=0)

            training_hist.append(history.history['matthews_correlation_coefficient'])
            val_history.append(history.history['val_matthews_correlation_coefficient'])
            scores.append(mcc)
            progress_bar.update(1)

mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}

sns.set()
plt.figure(figsize=(12, 8), dpi=100)
plt.plot(mean(np.asarray(training_hist), axis=0))
plt.plot(mean(np.asarray(val_history), axis=0))
plt.title(f'Average CNN Performance: Mean={mean_s} std={std_s} ste={ste_s}  K={config.k_folds} R={config.n_repeats}')
plt.legend(['Training mcc', 'Validation mcc'])
plt.xlabel('Epochs')
plt.ylabel('Matthews_correlation_coefficient')
plt.show()