from pca.pca import pca
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from tools.DatasetCompiler import DatasetCompiler
from xgboost import XGBClassifier

data = DatasetCompiler.load_from_local('../data/processed/non-filtered/lrg_clean_data_v2.pickle')

variance_threshold = 0.40
model = pca(n_components=variance_threshold,  )
results = model.fit_transform(data.x_train)
pc_components = pd.DataFrame.from_dict(results['PC'])

# PCA Loadings
top_fet = results['topfeat']
top_features = top_fet[top_fet['type'] == 'best'].copy(deep=True)

feature_nums = top_fet[top_fet['type'] == 'best']['feature'].to_numpy(dtype='int64')
feature_nums = np.asarray([x - 1 for x in feature_nums])
feature_names = data.feature_names[feature_nums]
names = pd.Series(feature_names)
top_features['feature'] = names

x_train_pca = pc_components.to_numpy()

# Setup
parameters = dict(
    src='../data/processed/non-filtered/lrg_clean_data.pickle',
    project_name='b2ar-no-filter-rfc-optimisation',
    notes='Full Bayes Optimisation on XGBoost GPU Accelerated',
    k_folds=5,
    repeats=1,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.5,
    colsample_bytree=0.3,
    booster='gbtree',
    gamma=1,
    eta=0,
    min_child_weight=1,
    max_delta_step=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1
)


# Load Data_set
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1)

#  vals, counts = np.unique(data.y_train, return_counts=True)
#  scale_pos_weight=int(np.round(max(counts) / min(counts), decimals=0)),
model = XGBClassifier(
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
    scale_pos_weight=1,
    tree_method='gpu_hist'
)

score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=x_train_pca, y=data.y_train,
                         scoring=score_func, cv=cv, n_jobs=-1)


mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
























# # Plot the Explained Variance
# model.plot()
# a = 0

# fig, ax = model.biplot(n_feat=5, figsize=(12, 8))
#
# plt.figure(figsize=(12, 8), dpi=100)
# component1 = 1
# component2 = 2
#
# plt.scatter(x=results['PC'][f'PC{component1}'].values, y=results['PC'][f'PC{component2}'].values, c=data.y_train)
# plt.xlabel(f'PC{component1} ({np.round(results["model"].explained_variance_ratio_[component1-1]*100, decimals=4)}%)')
# plt.ylabel(f'PC{component2} ({np.round(results["model"].explained_variance_ratio_[component2-1]*100, decimals=4)}%)')
# plt.show()
