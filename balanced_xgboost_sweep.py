import wandb
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix
from DatasetCompiler import DatasetCompiler
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler


# Setup
parameters = dict(
    src='./data/processed/lrg_clean_data.pickle',
    project_name='b2ar-no-filter-rfc-optimisation',
    notes='Full Bayes Optimisation on XGBoost GPU Accelerated',
    k_folds=5,
    repeats=1,
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

wandb.init(
           config=parameters,
           project=parameters['project_name'],
           notes=parameters['notes'],
           allow_val_change=True,
           name='jackpot test'
          )
config = wandb.config

# Load Data_set
data = DatasetCompiler.load_from_pickle(config.src)
cv = RepeatedStratifiedKFold(n_splits=config.k_folds, n_repeats=config.repeats)

vals, counts = np.unique(data.y_hold_out, return_counts=True)
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

over_sample = RandomOverSampler(sampling_strategy='minority')
x_train, y_train = over_sample.fit_resample(data.x_train, data.y_train)

score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=x_train, y=y_train,
                         scoring=score_func, cv=cv, n_jobs=-1)

model.fit(x_train, y_train)
y_preds = model.predict(data.x_hold_out)

hold_out_conf_mat = confusion_matrix(data.y_hold_out, y_preds)
hold_out_score = matthews_corrcoef(y_true=data.y_hold_out, y_pred=y_preds)

mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
wandb.log(metrics)
