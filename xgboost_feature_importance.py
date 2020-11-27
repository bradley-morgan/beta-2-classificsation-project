import wandb
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from DatasetCompiler import DatasetCompiler
from xgboost import XGBClassifier, plot_importance
import numpy as np
from tools.anonymousClass import Obj
import matplotlib.pyplot as plt


# Setup
config = Obj(
    src='./data/processed/lrg_clean_data.pickle',
    n_estimators=824,
    max_depth=10,
    learning_rate=0.0491,
    subsample=0.8955,
    colsample_bytree=0.9889,
    booster='gbtree',
    gamma=0.1448,
    eta=0.6901,
    min_child_weight=1.837,
    max_delta_step=2,
    reg_alpha=0.5011,
    reg_lambda=3.562,
    scale_pos_weight=1
)

# Load Data_set
data = DatasetCompiler.load_from_pickle(config.src)

#  vals, counts = np.unique(data.y_train, return_counts=True)
#  scale_pos_weight=int(np.round(max(counts) / min(counts), decimals=0)),
model = XGBClassifier(
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    learning_rate=config.learning_rate,
    subsample=config.subsample,
    colsample_bytree=config.colsample_bytree,
    booster=config.booster,
    gamma=config.gamma,
    eta=config.eta,
    min_child_weight=config.min_child_weight,
    max_delta_step=config.max_delta_step,
    reg_alpha=config.reg_alpha,
    reg_lambda=config.reg_lambda,
    scale_pos_weight=config.scale_pos_weight,
    tree_method='gpu_hist'
)

model.fit(data.x_train, data.y_train)

plot_importance(model)
plt.show()
