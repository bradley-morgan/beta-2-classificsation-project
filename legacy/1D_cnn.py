from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from DatasetCompiler import DatasetCompiler
import numpy as np
from tools.anonymousClass import Obj
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Input
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# TODO Make 1D cnn
# TODO Create cross validation loop with MCC scoring the test fold
# TODO Handle Data imbalances
# TODO Try Creating an Embedding One hot to vec approuch folllowed by Random Forest or PCA

config = Obj(
    src='../data/processed/lrg_clean_data.pickle',
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

# Load Data_set
data = DatasetCompiler.load_from_pickle(config.src)

#Balance Data or set Class Weights
vals, counts = np.unique(data.y_train, return_counts=True)
scale_pos_weight=int(np.round(max(counts) / min(counts), decimals=0)),

#Cross validation

scores = []
show_summary = True
training_hist = []
val_history = []
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
            model = Sequential()
            # model.add(Dense(136, activation='relu'))
            model.add(Input(shape=(train_x.shape[1], 1)))
            # model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
            model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))

            model.add(Dropout(0.9))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(1000, activation='relu'))
            model.add(Dense(450, activation='relu'))
            model.add(Dropout(0.45))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[matthews_correlation_coefficient])
            if show_summary:
                model.summary()
                show_summary = False
            history = model.fit(x=train_x, y=train_y, epochs=config.epochs, batch_size=config.batch_size,validation_split=0.1, verbose=1, class_weight={0:4, 1:1})
            training_hist.append(history.history['matthews_correlation_coefficient'])
            val_history.append(history.history['val_matthews_correlation_coefficient'])

            _, mcc = model.evaluate(x=val_x, y=val_y, batch_size=config.batch_size, verbose=0)

            scores.append(mcc)
            progress_bar.update(1)

mean_s = np.round(mean(scores), decimals=2)
std_s = np.round(std(scores), decimals=2)
ste_s = np.round(sem(scores), decimals=2)

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
