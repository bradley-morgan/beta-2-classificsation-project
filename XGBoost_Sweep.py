import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from DatasetCompiler import DatasetCompiler
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from tools.ImageSaver import ImageSaver
import tools.model_tools as m_tools
from tools.anonymousClass import Obj
from tqdm import tqdm
from scipy.stats import wilcoxon
import copy

# TODO Detect when sweep is being run and make sure not to log Model artifacts and save models locally

def get_median_confusion_matrix(conf_mat:list):
    mat = np.asarray(conf_mat)
    median_confusion_matrix = np.median(mat, axis=0)
    return median_confusion_matrix.astype('int64')


def plot_confusion_matrix(conf_mat):
    ax = sns.heatmap(conf_mat,
                     annot=True,
                     cbar=False,
                     fmt='d')
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.close(ax.get_figure())
    plot = ax.get_figure()
    return plot


def plot_performance(scores, stats: Obj):
    sns.set()
    plt.plot(scores, linewidth=2.0)
    plt.title(
        f"Validation MCC Model Performance: Mean={np.round(stats.mean, 4)}  "
        f"Std={np.round(stats.std, 4)}  Standard Error={np.round(stats.ste, 4)}")
    plt.xlabel('K Folds')
    plt.ylabel('Score')
    return plt.gcf()


def get_stats(scores, theoretical_median, alpha=0.05, alternative_hypothesis='two-sided'):
    # Get descriptives
    mean_s = np.mean(scores)
    median_s = np.median(scores)
    std_s = np.std(scores)
    ste_s = sem(scores)
    # Get inferential
    theoretical_median_scores = [theoretical_median] * len(scores)
    statistic, pvalue = wilcoxon(x=scores, y=theoretical_median_scores, alternative=alternative_hypothesis)
    # Get one tailed pvalue
    is_significant = pvalue < alpha
    # Wandb does not like bool types due to not being JSON serializable so changed to strings
    return Obj(
        test=' Wilcoxon signed-rank',
        alternative_hypothesis=alternative_hypothesis,
        mean=mean_s,
        median=median_s,
        std=std_s,
        ste=ste_s,
        statistic=statistic,
        pvalue=pvalue,
        alpha=alpha,
        theoretical_median=theoretical_median,
        is_significant='true' if is_significant else 'false',
        run_time_errors='true' if np.isnan(statistic) or np.isnan(pvalue) else 'false'
    )


def resample(X, y, resample_percentage):
    over_sample = RandomOverSampler(sampling_strategy=resample_percentage)
    resampled_x_train, resampled_y_train = over_sample.fit_resample(X, y)

    return resampled_x_train, resampled_y_train


def get_data(src):
    data = DatasetCompiler.load_from_pickle(src)
    vals, counts_before_sampling = np.unique(data.y_train, return_counts=True)

    return Obj(
        x_train=data.x_train,
        y_train=data.y_train,
        y_labels=vals,
        counts_before_sampling=counts_before_sampling,
    )


def make_model(model_config, test):
    if test:
        return XGBClassifier(n_estimators=10,
                             tree_method='gpu_hist'
                             )

    return XGBClassifier(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        learning_rate=model_config.learning_rate,
        subsample=model_config.subsample,
        colsample_bytree=model_config.colsample_bytree,
        booster=model_config.booster,
        gamma=model_config.gamma,
        eta=model_config.eta,
        min_child_weight=model_config.min_child_weight,
        max_delta_step=model_config.max_delta_step,
        reg_alpha=model_config.reg_alpha,
        reg_lambda=model_config.reg_lambda,
        scale_pos_weight=model_config.scale_pos_weight,
        tree_method='gpu_hist'
    )

# Setup
meta_data = dict(
    src='data/processed/lrg_clean_data.pickle',
    project_name='b2ar-no-filter-rfc-optimisation',
    notes='Cross Validation XGBoost GPU Accelerated',
    test_mode=False,
    k_folds=3,
    n_repeats=1,
    cross_val_theoretical_median=0.8,
    alpha=0.05,
    alternative_hypothesis='greater',
    n_estimators=979,
    resample_percentage=0.5,
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
    scale_pos_weight=4
)

run = wandb.init(
    config=meta_data,
    project=meta_data['project_name'],
    notes=meta_data['notes'],
    allow_val_change=True,
    name='test'
)
# Cross Validation Optimised XGBoost Round 2
config = wandb.config

# Load Data_set & Prepare Data
data = get_data(config.src)

# Cross validation
cross_val_scores = []
cross_val_conf_matrices = []
with tqdm(total=config.n_repeats*config.k_folds, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
          desc=f'Test Cycles: K={config.k_folds} R={config.n_repeats}') as progress_bar:
    for r in range(config.n_repeats):

        cv = StratifiedKFold(n_splits=config.k_folds, shuffle=True)
        for train_idx, test_idx in cv.split(data.x_train, data.y_train):
            # extract hold out test set
            train_x, val_x = data.x_train[train_idx], data.x_train[test_idx]
            train_y, val_y = data.y_train[train_idx], data.y_train[test_idx]

            # OverSample Minority class resample is a tunable learnt hyper-param
            train_x, train_y = resample(train_x, train_y, resample_percentage=config.resample_percentage)

            cross_val_model = make_model(config, test=config.test_mode)

            # Fit & Cross validate
            cross_val_model.fit(train_x, train_y)
            y_preds = cross_val_model.predict(val_x)

            conf_mat = confusion_matrix(val_y, y_preds)
            score = matthews_corrcoef(y_true=val_y, y_pred=y_preds)
            cross_val_scores.append(score)
            cross_val_conf_matrices.append(conf_mat)
            progress_bar.update(1)


cross_val_stats = get_stats(
    cross_val_scores,
    theoretical_median=config.cross_val_theoretical_median,
    alpha=config.alpha,
    alternative_hypothesis=config.alternative_hypothesis
)

cross_val_median_conf_mat = get_median_confusion_matrix(cross_val_conf_matrices)

run.log({'Mean Performance': cross_val_stats.mean})
run.log({'Cross Validation Statistics': cross_val_stats.to_dict()})
run.log({'Cross Validation Median Confusion Matrix': cross_val_median_conf_mat})
run.log({'Class Balances Before RandomOverSampling': {'Class Labels': data.y_labels, 'Class Counts': data.counts_before_sampling}})

