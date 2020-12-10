import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from tools.DatasetCompiler import DatasetCompiler
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from tools.ImageSaver import ImageSaver
import tools.model_tools as m_tools
from tqdm import tqdm
import copy

# TODO Detect when sweep is being run and make sure not to log Model artifacts and save models locally


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


def resample(X, y, resample_percentage):

    Error = False
    resampled_x_train=None
    resampled_y_train=None
    try:
        over_sample = RandomOverSampler(sampling_strategy=resample_percentage)
        resampled_x_train, resampled_y_train = over_sample.fit_resample(X, y)

    except ValueError:
        Error = True

    return resampled_x_train, resampled_y_train, Error


# Setup
meta_data = dict(
    src='../data/processed/non-filtered/lrg_clean_data_v2.pickle',
    project_name='b2ar-no-filter-rfc-optimisation',
    notes='Cross Validation XGBoost GPU Accelerated',
    test_mode=True,
    k_folds=5,
    n_repeats=5,
    cross_val_theoretical_median=0.78,
    hold_out_theoretical_median=0.78,
    alpha=0.05,
    alternative_hypothesis='greater',
    # Model Parameters
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
    scale_pos_weight=4
)

run = wandb.init(
    config=meta_data,
    project=meta_data['project_name'],
    notes=meta_data['notes'],
    allow_val_change=True,
    name='Cross Validation test'
)
config = wandb.config

# Load Data_set & Prepare Data
data = DatasetCompiler.load_from_local(config.src)

# Cross validation
cross_val_scores = []
cross_val_conf_matrices = []
hold_out_scores = []
hold_out_conf_matrices = []
with tqdm(total=config.n_repeats*config.k_folds, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
          desc=f'Test Cycles: K={config.k_folds} R={config.n_repeats}') as progress_bar:
    for r in range(config.n_repeats):

        cv = StratifiedKFold(n_splits=config.k_folds, shuffle=True)
        for train_idx, test_idx in cv.split(data.x_train, data.y_train):
            # extract hold out test set
            train_x, val_x = data.x_train[train_idx], data.x_train[test_idx]
            train_y, val_y = data.y_train[train_idx], data.y_train[test_idx]

            cross_val_model = make_model(config, test=config.test_mode)

            # Fit & Cross validate
            cross_val_model.fit(train_x, train_y)
            y_preds = cross_val_model.predict(val_x)

            conf_mat = confusion_matrix(val_y, y_preds)
            score = matthews_corrcoef(y_true=val_y, y_pred=y_preds)
            cross_val_scores.append(score)
            cross_val_conf_matrices.append(conf_mat)
            progress_bar.update(1)

        # Hold out Evaluation: Train model on whole data-set then do final unseen test
        hold_out_model = make_model(config, config.test_mode)

        hold_out_model.fit(data.x_train, data.y_train)
        hold_out_y_preds = hold_out_model.predict(data.x_hold_out)

        hold_out_conf_mat = confusion_matrix(data.y_hold_out, hold_out_y_preds)
        hold_out_score = matthews_corrcoef(y_true=data.y_hold_out, y_pred=hold_out_y_preds)
        hold_out_scores.append(hold_out_score)
        hold_out_conf_matrices.append(hold_out_conf_mat)


cross_val_stats = m_tools.get_wilcoxon_stats(
    cross_val_scores,
    scores_B=config.cross_val_theoretical_median,
    alpha=config.alpha,
    alternative_hypothesis=config.alternative_hypothesis
)
hold_out_stats = m_tools.get_wilcoxon_stats(
    hold_out_scores,
    scores_B=config.hold_out_theoretical_median,
    alpha=config.alpha,
    alternative_hypothesis=config.alternative_hypothesis
)
cross_val_median_conf_mat = m_tools.get_median_confusion_matrix(cross_val_conf_matrices)
hold_out_median_conf_mat = m_tools.get_median_confusion_matrix(hold_out_conf_matrices)

run.log({'Cross Validation Statistics': cross_val_stats.to_dict()})
run.log({'Hold Out Test Statistics': hold_out_stats.to_dict()})
run.log({'Cross Validation Median Confusion Matrix': cross_val_median_conf_mat})
run.log({'Hold Out Median Confusion Matrix': hold_out_median_conf_mat})
labels, counts = np.unique(data.y_train, return_counts=True)
run.log({'Class Balances Training Set': {'Class Labels': labels, 'Class Counts': counts}})

image_saver = ImageSaver(run)
image_saver.save(plot=m_tools.plot_performance(cross_val_scores, cross_val_stats),
                 name='Cross Validation Performance', format='png')
image_saver.save(plot=m_tools.plot_performance(hold_out_scores, hold_out_stats),
                 name='Hold Out Test Performance', format='png')
image_saver.save(plot=m_tools.plot_confusion_matrix(cross_val_median_conf_mat),
                 name='Cross Validation: Median confusion_matrix', format='png')
image_saver.save(plot=m_tools.plot_confusion_matrix(hold_out_median_conf_mat),
                 name='Hold Out Test: Median confusion_matrix', format='png')

meta_data_cloud = copy.deepcopy(meta_data)
meta_data['last run stats'] = dict(cross_val=cross_val_stats.to_dict(), hold_out=hold_out_stats.to_dict())
meta_data['last run scores'] = dict(cross_val=cross_val_scores, hold_out=hold_out_scores)
model_file_path = m_tools.local_save_model(hold_out_model, 'XGBoostClassifier_Cross_Val', meta_data, return_path=True)
model_artifact = wandb.Artifact(name='XGBoostClassifier_Cross_Val', type='models', metadata=meta_data_cloud)
model_artifact.add_file(model_file_path)
run.log_artifact(model_artifact)
