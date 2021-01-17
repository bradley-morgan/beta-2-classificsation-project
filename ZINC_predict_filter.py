from tools.DatasetCompiler import DatasetCompiler
from transforms.merge_datasets import MergeDatasets
from transforms.change_nans import ChangeNans
from transforms.clean_feature_names import CleanFeatureNames
from tools.model_performance_estimation import CrossValidation
from transforms.remove_features import RemoveFeatures
from tools.make_models import xgboost_constructor
from tools.anonymousClass import Obj
import wandb
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef
from tools.make_models import make_model, decision_tree_contrustor
import tools.model_tools as m_tools
from tools.ImageSaver import ImageSaver

run = wandb.init(
    project='test'
)
image_saver = ImageSaver(run)
# Model Parameters
model_params = Obj(
    test_mode=False,
    model='decision_tree',
    load_model_from='train',
    criterion='gini',
    splitter='best',
    max_depth=10,
    max_features=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
)

# Get features from data model originally trained on
og_data = DatasetCompiler.load_from_local('./data/processed/filtered/dataset1-2percent-hold-out.pickle')
og_features = og_data.feature_names

original_model = decision_tree_contrustor(m_config=Obj(
    load_model_from='cloud',
    global_load_run_path='bradamorg/B2AR-Filtered/23xzu26f',
    model_file_name='v15_CV-DecisionTree.joblib'
))

split_node_idxs = [f for f in original_model.tree_.feature if f >= 0]
split_node_feature_names = og_features[split_node_idxs]
og_x_train_reduced = pd.DataFrame(og_data.x_train, columns=og_features)[split_node_feature_names]
og_x_hold_out_reduced = pd.DataFrame(og_data.x_hold_out, columns=og_features)[split_node_feature_names]

reduced_dataset = Obj(
    x_train=og_x_train_reduced.to_numpy().astype('int64'),
    y_train=og_data.y_train,
    x_hold_out=og_x_hold_out_reduced.to_numpy().astype('int64'),
    y_hold_out=og_data.y_hold_out
)

reduced_results = CrossValidation(
    k_folds=10, n_repeats=3, data=reduced_dataset,
    make_model_func=make_model(model_params.model), model_parameters=model_params
).run()
reduced_stats= m_tools.get_descriptive_stats(reduced_results.cross_val_mcc_scores)
y_pred = reduced_results.model.predict_proba(reduced_dataset.x_train)

zinc_model = reduced_results.model

image_saver.save_graphviz(
    zinc_model,
    split_node_feature_names,
    ['ant', 'ag'],
    'Reduced Tree'
)

# Process ZINC Data then reduce the features to only features the model used to split original data on
src = 'data/ZINC/filtered'
data_sets = DatasetCompiler(src=src, y_labels="target", test_size=0.02)
data_sets.load()
# data_sets.remove_feature(feature_name='Ligand_Pose2')
data_sets = CleanFeatureNames(dict(exceptions=['Ligand_Pose2']))(data_sets)
data_sets = MergeDatasets(config=dict(
    merge_all=True,
    merge_all_name='ZINC-Merged',
    groups=[],
    group_names=['3sn6-3sn6', '4lde-4lde', '5jqh-5jqh']
))(data_sets)
data_sets = ChangeNans(config=dict(value=0))(data_sets)
zinc_ligand_poses = data_sets.datasets['ZINC-Merged']['data']['Ligand_Pose2'].to_numpy()
data_sets.datasets['ZINC-Merged']['data'].drop(['Ligand_Pose2'], axis=1, inplace=True)
zinc_features = data_sets.datasets['ZINC-Merged']['data'].columns.to_numpy()
shared_features = list(set(og_features).intersection(zinc_features))
zinc_data = data_sets.datasets['ZINC-Merged']['data'][split_node_feature_names].to_numpy().astype('int64')

zinc_preds = zinc_model.predict(zinc_data)
zinc_y_probs = zinc_model.predict_proba(zinc_data)

zinc_df = pd.DataFrame(zip(zinc_ligand_poses, zinc_preds, zinc_y_probs[:,0], zinc_y_probs[:, 1]),
                               columns=['Ligand Pose','Class Prediction', 'Ant Probability', 'Ag Probability'])


# run.log({'zinc model predictions': wandb.Table(dataframe=zinc_df)})

zinc_df.to_csv('./analysis/V2_decision_tree_zinc_predictions.csv')
