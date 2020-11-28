import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix
from DatasetCompiler import DatasetCompiler
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from tools.ImageSaver import ImageSaver
import tools.model_tools as m_tools
from tools.anonymousClass import Obj
from tqdm import tqdm
from scipy.stats import ttest_1samp

def plot_confusion_matrix(y_true, y_preds):

    hold_out_conf_mat = confusion_matrix(y_true, y_preds)
    ax = sns.heatmap(hold_out_conf_mat,
                     annot=True,
                     cbar=False,
                     fmt='d')
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.close(ax.get_figure())
    plot = ax.get_figure()
    return plot


def log_performance(scores, mean_s, std_s, ste_s):
    sns.set()
    plt.plot(scores, linewidth=2.0)
    plt.title(
        f"Validation MCC Model Performance: Mean={np.round(mean_s, 4)}  "
        f"Std={np.round(std_s, 4)}  Standard Error={np.round(ste_s, 4)}")
    plt.xlabel('K Folds')
    plt.ylabel('Score')
    return plt.gcf()

def get_stats(scores, population_mean, alpha=0.05):
    # Get descriptives
    mean_s = mean(scores)
    std_s = std(scores)
    ste_s = sem(scores)
    # Get inferential
    tscore, pvalue = ttest_1samp(scores, population_mean)
    # Get one tailed pvalue
    pvalue = pvalue/2
    is_significant = pvalue < alpha
    return Obj(
        type='One Tailed',
        mean=mean_s,
        std=std_s,
        ste=ste_s,
        tscore=tscore,
        pvalue=pvalue,
        alpha=alpha,
        population_mean=population_mean,
        is_significant=is_significant
    )


def get_data(src):

    data = DatasetCompiler.load_from_pickle(src)
    _, counts_before_sampling = np.unique(data.y_train, return_counts=True)
    over_sample = RandomOverSampler(sampling_strategy='minority')
    resampled_x_train, resampled_y_train = over_sample.fit_resample(data.x_train, data.y_train)
    vals, counts_after_sampling = np.unique(resampled_y_train, return_counts=True)
    print(f'Class Distribution For Run: Classes: {vals}, Class Totals: {counts_after_sampling}')

    return Obj(
        x_train=data.x_train,
        y_train=data.y_train,
        resampled_x_train=resampled_x_train,
        resampled_y_train=resampled_y_train,
        x_hold_out=data.x_hold_out,
        y_hold_out=data.y_hold_out,
        y_labels=vals,
        counts_before_sampling=counts_before_sampling,
        counts_after_sampling=counts_after_sampling
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
parameters = dict(
    src='./data/processed/lrg_clean_data.pickle',
    project_name='b2ar-no-filter-rfc-optimisation',
    notes='Full Bayes Optimisation on XGBoost GPU Accelerated',
    k_folds=5,
    n_repeats=30,
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
    name='Cross validated & Optimised XGBoostClassifier'
)
config = wandb.config

# Load Data_set & Prepare Data
data = get_data(config.src)

# Cross validation
cross_val_scores = []
cross_val_conf_matrices = []
hold_out_scores = []
hold_out_conf_matrices = []
with tqdm(total=config.n_repeats*config.k_folds, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
          desc=f'Test Cycles: K={config.k_folds} R={config.n_repeats}') as progress_bar:
    for r in range(config.n_repeats):

        cv = StratifiedKFold(n_splits=config.k_folds, shuffle=True)
        for train_idx, test_idx in cv.split(data.resampled_x_train, data.resampled_y_train):
            # extract hold out test set
            train_x, val_x = data.resampled_x_train[train_idx], data.resampled_x_train[test_idx]
            train_y, val_y = data.resampled_y_train[train_idx], data.resampled_y_train[test_idx]

            cross_val_model = make_model(config, test=True)

            # Fit & Cross validate
            cross_val_model.fit(train_x, train_y)
            y_preds = cross_val_model.predict(val_x)

            conf_mat = confusion_matrix(val_y, y_preds)
            score = matthews_corrcoef(y_true=val_y, y_pred=y_preds)
            cross_val_scores.append(score)
            cross_val_conf_matrices.append(conf_mat)
            progress_bar.update(1)

        # Hold out Evaluation: Train model on whole data-set then do final unseen test
        hold_out_model = make_model(config, True)

        hold_out_model.fit(data.resampled_x_train, data.resampled_y_train)
        hold_out_y_preds = hold_out_model.predict(data.x_hold_out)

        hold_out_conf_mat = confusion_matrix(data.y_hold_out, hold_out_y_preds)
        hold_out_score = matthews_corrcoef(y_true=data.y_hold_out, y_pred=hold_out_y_preds)
        hold_out_scores.append(hold_out_score)
        hold_out_conf_matrices.append(hold_out_conf_mat)


cross_val_stats = get_stats(cross_val_scores, population_mean=0.6171, alpha=0.05)
hold_out_stats = get_stats(hold_out_scores, population_mean=0.5, alpha=0.05)

run.log({'Cross Validation Statistics': cross_val_stats})
run.log({'Hold Out Test Statistics': hold_out_stats})
run.log({'Class Balances Before RandomOverSampling': {'Class Labels': data.y_labels, 'Class Counts': data.counts_before_sampling}})
run.log({'Class Balances After RandomOverSampling': {'Class Labels': data.y_labels, 'Class Counts': data.counts_after_sampling}})

image_saver = ImageSaver(run)
image_saver.save(plot=log_performance(scores, mean_s, std_s, ste_s),
                 name='Hold Out Set: Median confusion_matrix', format='png')

image_saver.save(plot=test_model_on_hold_out(run, model, data),
                 name=f'Cross-Validation Performance: Repeats={config.repeats} K-folds={config.k_folds}',
                 format='png')

m_tools.save_model(hold_out_model, 'XGBoostClassifier', parameters)
