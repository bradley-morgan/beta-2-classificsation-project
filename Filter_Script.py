from tools.anonymousClass import Obj
from tools.make_models import make_model

MODEL_ESTIMATION_RUN_ID = '12'
CROSS_VALIDATION_RUN_ID = '5'
FEATURE_IMPORTANCE_RUN_ID = '2'

global_src = 'data/processed/filtered/filter-resample-test.pickle'
global_project_name = 'test'
global_cloud_log = True
global_test_mode = False
global_artifact_name = 'DecisionTree'
global_model = 'decision_tree'
global_load_model_from = 'train'
global_load_run_path='bradamorg/filter-test/6'
global_model_file_name = 'v4_CV-DecisionTree.joblib'

# Model Parameters Need to match global model
criterion = 'gini'
splitter = 'best'
max_depth = 6
max_features = 'auto'
min_samples_split = 2
min_samples_leaf = 1
class_weight = 'balanced'

def get_config():
    model_estimation_config = Obj(
        # Global Parameters
        src=global_src,
        project_name=global_project_name,
        cloud_log=global_cloud_log,
        test_mode=global_test_mode,
        artifact_name=global_artifact_name,
        model=global_model,
        id=MODEL_ESTIMATION_RUN_ID,
        # Function Parameters
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        run_name='resample Test confidence 2',
        notes='notes',
        is_d_tree=True,
        test_repeats=[3, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        n_repeats=1000,
        n_samples=0.8,
        confidence_level=95,
        time_units='mins',
        time_threshold=None,
        ste_threshold=None,
        # model parameters
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight
    )

    cross_validation_config = Obj(
        # Global Parameters
        src=global_src,
        project_name=global_project_name,
        cloud_log=global_cloud_log,
        test_mode=global_test_mode,
        artifact_name=global_artifact_name,
        model=global_model,
        id=CROSS_VALIDATION_RUN_ID,
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        # Function Parameters
        notes='Decision Tree',
        run_name='dt cross val 3',
        run_sweep=False,
        k_folds=10,
        n_repeats=100,
        confidence_level=95,
        # Non-Sweep Model Parameters
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight
    )

    feature_importance_config = Obj(
        # Global Parameters
        src=global_src,
        project_name=global_project_name,
        cloud_log=global_cloud_log,
        test_mode=global_test_mode,
        artifact_name=global_artifact_name,
        model=f'FIMP-{global_model}',
        id=FEATURE_IMPORTANCE_RUN_ID,
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        # Function Parameters
        run_name='Decision Tree Feature Importance',
        notes='Notes',
        n_jobs=1,
        target_datasets=[('x_train', 'y_train'), ('x_hold_out', 'y_hold_out')],
        n_repeats=3,
        confidence_level=95,
        run_threshold_method=True,
        # model parameters
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight
    )

    return Obj(
        model_estimation_config=model_estimation_config,
        cross_validation_config=cross_validation_config,
        feature_importance_config=feature_importance_config
    )
