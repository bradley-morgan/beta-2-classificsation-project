import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score
from tools.DatasetCompiler import DatasetCompiler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tools.ImageSaver import ImageSaver
import tools.model_tools as m_tools
from tqdm import tqdm
import copy

def make_model(model_config, test):
    if test:
        return DecisionTreeClassifier()

    # Need to run sweep on filtered data to get optimised model params
    return DecisionTreeClassifier()

meta_data = dict(
    src='data/processed/non-filtered/lrg_clean_data_v2.pickle',
    project_name='b2ar-filtered',
    notes='Cross Validation Decision Tree',
    test_mode=True,
    k_folds=10,
    n_repeats=100,
    alpha=0.05,
    alternative_hypothesis='greater',
    artifact_name='FilteredDecisionTreeClassifier'
    # Model Parameters
)

run = wandb.init(
    config=meta_data,
    project=meta_data['project_name'],
    notes=meta_data['notes'],
    allow_val_change=True,
    name='Cross Validation TEST'
)
config = wandb.config

data = DatasetCompiler.load_from_pickle(config.src)

# Cross validation
cross_val_mcc_scores = []
cross_val_acc_scores=[]
cross_val_conf_matrices = []
hold_out_mcc_scores = []
hold_out_acc_scores = []
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
            mcc_score = matthews_corrcoef(y_true=val_y, y_pred=y_preds)
            acc_score = accuracy_score(y_true=val_y, y_pred=y_preds)

            cross_val_mcc_scores.append(mcc_score)
            cross_val_acc_scores.append(acc_score)
            cross_val_conf_matrices.append(conf_mat)
            progress_bar.update(1)

        # Hold out Evaluation: Train model on whole data-set then do final unseen test
        hold_out_model = make_model(config, config.test_mode)

        hold_out_model.fit(data.x_train, data.y_train)
        hold_out_y_preds = hold_out_model.predict(data.x_hold_out)

        hold_out_conf_mat = confusion_matrix(data.y_hold_out, hold_out_y_preds)
        hold_out_mcc_score = matthews_corrcoef(y_true=data.y_hold_out, y_pred=hold_out_y_preds)
        hold_out_acc_score = accuracy_score(y_true=data.y_hold_out, y_pred=hold_out_y_preds)

        hold_out_mcc_scores.append(hold_out_mcc_score)
        hold_out_acc_scores.append(hold_out_acc_score)
        hold_out_conf_matrices.append(hold_out_conf_mat)

mcc_confidence_interval = m_tools.get_resampled_confidence_interval(
    cross_val_mcc_scores, score_range=[-1,1], alpha=5.0
)

acc_confidence_interval = m_tools.get_resampled_confidence_interval(
    cross_val_acc_scores, score_range=[0,1], alpha=5.0
)

cross_val_median_conf_mat = m_tools.get_median_confusion_matrix(cross_val_conf_matrices)
hold_out_median_conf_mat = m_tools.get_median_confusion_matrix(hold_out_conf_matrices)

# run.log({'Cross Validation Statistics': cross_val_stats.to_dict()})
# run.log({'Hold Out Test Statistics': hold_out_stats.to_dict()})
run.log({'Cross Validation Median Confusion Matrix': cross_val_median_conf_mat})
run.log({'Hold Out Median Confusion Matrix': hold_out_median_conf_mat})
labels, counts = np.unique(data.y_train, return_counts=True)
run.log({'Class Balances Training Set': {'Class Labels': labels, 'Class Counts': counts}})

image_saver = ImageSaver(run)
image_saver.save_graphviz(
    graph_name='Decision Tree Structure',
    model=hold_out_model,
    feature_names=data.feature_names,
    class_names=['ant', 'ag']
)
# image_saver.save(plot=m_tools.plot_performance(cross_val_scores, cross_val_stats),
#                  name='Cross Validation Performance', format='png')
# image_saver.save(plot=m_tools.plot_performance(hold_out_scores, hold_out_stats),
#                  name='Hold Out Test Performance', format='png')
image_saver.save(plot=m_tools.plot_confusion_matrix(cross_val_median_conf_mat),
                 name='Cross Validation: Median confusion_matrix', format='png')
image_saver.save(plot=m_tools.plot_confusion_matrix(hold_out_median_conf_mat),
                 name='Hold Out Test: Median confusion_matrix', format='png')

meta_data_cloud = copy.deepcopy(meta_data)
# meta_data['last run stats'] = dict(cross_val=cross_val_stats.to_dict(), hold_out=hold_out_stats.to_dict())
meta_data['last run scores'] = dict(cross_val=cross_val_mcc_scores, hold_out=hold_out_mcc_scores)
model_file_path = m_tools.local_save_model(hold_out_model, config.artifact_name, meta_data, return_path=True)
model_artifact = wandb.Artifact(name=config.artifact_name, type='models', metadata=meta_data_cloud)
model_artifact.add_file(model_file_path)
run.log_artifact(model_artifact)