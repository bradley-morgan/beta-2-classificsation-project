from tools.DatasetCompiler import DatasetCompiler
from transforms.merge_datasets import MergeDatasets
from transforms.change_nans import ChangeNans
from transforms.clean_feature_names import CleanFeatureNames
from transforms.remove_features import RemoveFeatures
from tools.make_models import xgboost_constructor
from tools.anonymousClass import Obj
import wandb
import pandas as pd

wandb.init()
# Get features from data model originally trained on
og_data = DatasetCompiler.load_from_local('./data/processed/non-filtered/dataset1-2percent-hold-out.pickle')
og_features = og_data.feature_names

src = 'data/ZINC/unfiltered'
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
zinc_data = data_sets.datasets['ZINC-Merged']['data'].to_numpy().astype('int64')

list_comp = [i for i, j in zip(og_features, zinc_features) if i == j]

xgboost_model = xgboost_constructor(m_config=Obj(
    load_model_from='cloud',
    global_load_run_path='bradamorg/B2AR-Unfiltered/2rcr4zra',
    model_file_name='v3_CV-XGBoost.joblib'
))

xgboost_preds = xgboost_model.predict(zinc_data)
xgboost_y_probs = xgboost_model.predict_proba(zinc_data)

xgboost_zinc_df = pd.DataFrame(zip(zinc_ligand_poses, xgboost_preds, xgboost_y_probs[:,0], xgboost_y_probs[:, 1]),
                               columns=['Ligand Pose', 'Class Prediction', 'Ant Probability', 'Ag Probability'])


xgboost_zinc_df.to_csv('./analysis/V2_xgboost_zinc_predictions.csv')