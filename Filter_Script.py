from tools.anonymousClass import Obj
from tools.make_models import make_model

EXPERIMENT = 'unfiltered'


def get_config():

    if EXPERIMENT.lower() == 'filtered':
        return get_filtered_config()

    elif EXPERIMENT.lower() == 'unfiltered':
        return get_unfiltered_config()


def get_filtered_config():
    global_src = 'data/processed/filtered/dataset1-10percent-hold-out.pickle'
    global_project_name = 'B2AR-Filtered'
    global_cloud_log = True
    global_test_mode = False
    global_artifact_name = 'DecisionTree'
    global_model = 'decision_tree'
    global_load_model_from = 'train'
    global_load_run_path = ' '
    global_model_file_name = ' '

    # Model Parameters Need to match global model
    # DEFAULT
    # criterion = 'gini'
    # splitter = 'best'
    # max_depth = None
    # max_features = 'auto'
    # min_samples_split = 2
    # min_samples_leaf = 1
    # class_weight = 'balanced'

    # LOW COMPLEXITY OPTIMISED Low Complexity Cross Validation Round 1
    criterion = 'gini'
    splitter = 'best'
    max_depth = None
    max_features = 'auto'
    min_samples_split = 2
    min_samples_leaf = 1
    class_weight = None

    # # MEDIUM COMPLEXITY OPTIMISED
    # criterion = 'entropy'
    # splitter = 'random'
    # max_depth = 16
    # max_features = 139
    # min_samples_split = 32
    # min_samples_leaf = 10
    # class_weight = 'balanced'
    #
    # # HIGH COMPLEXITY OPTIMISED
    # criterion = 'gini'
    # splitter = 'random'
    # max_depth = 122
    # max_features = 139
    # min_samples_split = 17
    # min_samples_leaf = 1
    # class_weight = 'balanced'

    model_estimation_config = Obj(
        # Global Parameters
        src=global_src,
        project_name=global_project_name,
        cloud_log=global_cloud_log,
        test_mode=global_test_mode,
        artifact_name=global_artifact_name,
        model=global_model,
        # id=MODEL_ESTIMATION_RUN_ID,
        # Function Parameters
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        run_name='XGBoost Variance Estimation',
        notes='notes',
        is_d_tree=True,
        test_repeats=[3, 5, 10, 30, 100, 250],
        n_repeats=200,
        n_samples=1.0,
        confidence_level=99,
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
        # id=CROSS_VALIDATION_RUN_ID,
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        # Function Parameters
        notes='Decision Tree',
        run_name='D_Tree Cross Validation',
        run_sweep=False,
        sweep_name='D_Tree Cross Validation Round 1',
        k_folds=10,
        n_repeats=3,
        confidence_level=99,
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
        model=global_model,
        # id=FEATURE_IMPORTANCE_RUN_ID,
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        # Function Parameters
        run_name='D-Tree Low Complexity Feature Importance Round 1',
        notes='Notes',
        n_jobs=-1,
        target_datasets=[('x_train', 'y_train'), ('x_hold_out', 'y_hold_out')],
        n_repeats=200,
        confidence_level=99,
        run_threshold_method=False,
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


def get_unfiltered_config():
    global_src = 'data/processed/non-filtered/dataset1-10percent-hold-out.pickle'
    global_project_name = 'B2AR-Unfiltered'
    global_cloud_log = True
    global_test_mode = False
    global_artifact_name = 'DecisionTree'
    global_model = 'xgb'
    global_load_model_from = 'local'
    global_load_run_path = 'bradamorg/B2AR-Filtered/27p42aap'
    global_model_file_name = './saved_models/XGBoostClassifier_Cross_Val.joblib'

    # Model Parameters Need to match global model
    # DEFAULT
    # criterion = 'gini'
    # splitter = 'best'
    # max_depth = None
    # max_features = 'auto'
    # min_samples_split = 2
    # min_samples_leaf = 1
    # class_weight = 'balanced'

    # LOW COMPLEXITY OPTIMISED Low Complexity Cross Validation Round 1
    criterion = 'gini'
    splitter = 'best'
    max_depth = None
    max_features = 'auto'
    min_samples_split = 2
    min_samples_leaf = 1
    class_weight = None

    model_estimation_config = Obj(
        # Global Parameters
        src=global_src,
        project_name=global_project_name,
        cloud_log=global_cloud_log,
        test_mode=global_test_mode,
        artifact_name=global_artifact_name,
        model=global_model,
        # id=MODEL_ESTIMATION_RUN_ID,
        # Function Parameters
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        run_name='XGBoost Variance Estimation',
        notes='notes',
        is_d_tree=True,
        test_repeats=[3, 5, 10, 30, 100, 250],
        n_repeats=200,
        n_samples=1.0,
        confidence_level=99,
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
        # id=CROSS_VALIDATION_RUN_ID,
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        # Function Parameters
        notes='Decision Tree',
        run_name='XGBoost Round 1',
        run_sweep=False,
        sweep_name='XgBoost Round 1',
        k_folds=10,
        n_repeats=3,
        confidence_level=99,
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
        model=global_model,
        # id=FEATURE_IMPORTANCE_RUN_ID,
        load_model_from=global_load_model_from,
        global_load_run_path=global_load_run_path,
        model_file_name=global_model_file_name,
        # Function Parameters
        run_name='D-Tree Low Complexity Feature Importance Round 1',
        notes='Notes',
        n_jobs=-1,
        target_datasets=[('x_train', 'y_train'), ('x_hold_out', 'y_hold_out')],
        n_repeats=200,
        confidence_level=99,
        run_threshold_method=False,
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

