import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix
from DatasetCompiler import DatasetCompiler
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from tools.ImageSaver import ImageSaver

def test_model_on_hold_out(run, model):
    model.fit(x_train, y_train)
    y_preds = model.predict(data.x_hold_out)

    hold_out_conf_mat = confusion_matrix(data.y_hold_out, y_preds)
    hold_out_score = matthews_corrcoef(y_true=data.y_hold_out, y_pred=y_preds)

    run.log({'Hold Out MCC': hold_out_score})
    ax = sns.heatmap(hold_out_conf_mat,
                     annot=True,
                     cbar=False,
                     fmt='d')
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.close(ax.get_figure())
    plot = ax.get_figure()
    ImageSaver().save(plot=plot, run=run, name='Hold Out Set: Median confusion_matrix', format='png')
    plt.clf()


# Setup
parameters = dict(
    src='./data/processed/lrg_clean_data.pickle',
    project_name='b2ar-no-filter-rfc-optimisation',
    notes='Full Bayes Optimisation on XGBoost GPU Accelerated',
    k_folds=3,
    repeats=1,
    n_estimators=979,
    max_depth=19,
    learning_rate=0.07944,
    subsample=0.851,
    colsample_bytree=0.7626,
    booster='gbtree',
    gamma=0.2376,
    eta=0.4664,
    min_child_weight=0.08316,
    max_delta_step=10,
    reg_alpha=0.01635,
    reg_lambda=0.6584,
    scale_pos_weight=1

)

run = wandb.init(
    config=parameters,
    project=parameters['project_name'],
    notes=parameters['notes'],
    allow_val_change=True,
    name='test run'
)
config = wandb.config

# Load Data_set
data = DatasetCompiler.load_from_pickle(config.src)
cv = RepeatedStratifiedKFold(n_splits=config.k_folds, n_repeats=config.repeats)

#  scale_pos_weight=int(np.round(max(counts) / min(counts), decimals=0)),
# n_estimators = config.n_estimators,
# max_depth = config.max_depth,
# learning_rate = config.learning_rate,
# subsample = config.subsample,
# colsample_bytree = config.colsample_bytree,
# booster = config.booster,
# gamma = config.gamma,
# eta = config.eta,
# min_child_weight = config.min_child_weight,
# max_delta_step = config.max_delta_step,
# reg_alpha = config.reg_alpha,
# reg_lambda = config.reg_lambda,
# scale_pos_weight = config.scale_pos_weight,
model = XGBClassifier(
    tree_method='gpu_hist'
)

over_sample = RandomOverSampler(sampling_strategy='minority')
x_train, y_train = over_sample.fit_resample(data.x_train, data.y_train)

vals, counts = np.unique(y_train, return_counts=True)
print(f'Class Distribution For Run: Classes: {vals}, Class Totals: {counts}')

score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=x_train, y=y_train,
                         scoring=score_func, cv=cv, n_jobs=-1)

mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
run.log(metrics)

# TODO LOG Performance as graph
# TODO Save Model
# TODO class balances before and after RandomOverSampling

test_model_on_hold_out(run, model)



