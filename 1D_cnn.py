from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from DatasetCompiler import DatasetCompiler
import numpy as np
from tools.anonymousClass import Obj
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense


config = Obj(
    src='./data/processed/lrg_clean_data.pickle',
    k_folds=5,
    repeats=1,
    epochs=200,
    batch_size=64
)

# Load Data_set
data = DatasetCompiler.load_from_pickle(config.src)
cv = RepeatedStratifiedKFold(n_splits=config.k_folds, n_repeats=config.repeats)

#  vals, counts = np.unique(data.y_train, return_counts=True)
#  scale_pos_weight=int(np.round(max(counts) / min(counts), decimals=0)),

# TODO Make 1D cnn
# TODO Create cross validation loop with MCC scoring the test fold
# TODO Handle Data imbalances
# TODO Try Creating an Embedding One hot to vec approuch folllowed by Random Forest or PCA

n_samples, n_features = data.x_train.shape[1], data.x_train.shape[2],
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_samples, n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x='X DATA', y='Y DATA',epochs=config.epochs, batch_size=config.batch_size, verbose=0)



score_func = make_scorer(matthews_corrcoef)
scores = 'Get MCC after model training'


mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
