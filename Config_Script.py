from tools.anonymousClass import Obj
from tools.make_models import make_model

EXPERIMENT = 'filtered'


def get_config():
    if EXPERIMENT.lower() == 'filtered':
        return get_filtered_config()

    elif EXPERIMENT.lower() == 'unfiltered':
        return get_unfiltered_config()


def get_filtered_config():
    global_src = 'data/processed/filtered/filter-train-processor_98.pickle'
    global_project_name = 'B2AR-unfilt-get-shaps'
    global_cloud_log = True
    global_test_mode = False
    global_artifact_name = 'RandomForest-98-46'
    global_model = 'random_forest'
    global_load_model_from = 'cloud'
    global_load_run_path = 'bradamorg/B2AR-Hotfix/3ifm2as1'
    global_model_file_name = 'CV-RandomForest-98-46.joblib'

    sweep_config = dict(
        method='grid',
        metric=dict(
            goal='maximize',
            name='mean'
        ),
        parameters=dict(
            max_depth=dict(
                values=[i for i in range(1, 31)]
            ),
            max_features=dict(
                values=[i for i in range(1, 162)]
            )
        )
    )

    # Model Parameters Need to match global model
    # DEFAULT
    max_depth = 5
    max_features = 5
    n_estimators = 2000
    n_jobs=12

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
        run_name='Uncertainty Estimation RFC-98-46',
        notes='notes',
        test_repeats=[10, 100, 500, 1000, 1500, 2000],
        n_repeats=8,
        n_samples=1.0,
        confidence_level=99,
        time_units='mins',
        time_threshold=None,
        ste_threshold=None,
        # model parameters
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
        n_jobs=n_jobs
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
        notes='Random Forest',
        run_name='Cross validation RFC-98-3-fold',
        run_sweep=False,
        k_folds=3,
        n_repeats=8,
        confidence_level=99,
        # Non-Sweep Model Parameters
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
        n_jobs=n_jobs
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
        run_name='Feature Importance RFC-98-46',
        notes='Notes',
        n_jobs=n_jobs,
        target_datasets=[('x_train', 'y_train'), ('x_hold_out', 'y_hold_out')],
        method='shap',  # shap or default
        n_repeats=30,
        confidence_level=99,
        run_threshold_method=False,
        # model parameters
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
    )

    return Obj(
        model_estimation_config=model_estimation_config,
        cross_validation_config=Obj(config=cross_validation_config, sweep_config=sweep_config),
        feature_importance_config=feature_importance_config
    )


def get_unfiltered_config():
    global_src = 'data/processed/non-filtered/dataset1-2percent-hold-out.pickle'
    global_project_name = 'B2AR-unfilt-get-shaps'
    global_cloud_log = True
    global_test_mode = False
    global_artifact_name = 'XGBoost'
    global_model = 'xgb'
    global_load_model_from = 'cloud'
    global_load_run_path = 'bradamorg/B2AR-Unfiltered/2rcr4zra'
    global_model_file_name = 'v3_CV-XGBoost.joblib'

    # Model Parameters Need to match global model
    # DEFAULT
    # n_estimators = 979
    # max_depth = 19
    # learning_rate = 0.07944
    # subsample = 0.851
    # colsample_bytree = 0.7626
    # booster = 'gbtree'
    # gamma = 0.2376
    # eta = 0.4664
    # min_child_weight = 0.08316
    # max_delta_step = 10
    # reg_alpha = 0.01635
    # reg_lambda = 0.6584
    # scale_pos_weight = 4

    n_estimators = 1150
    max_depth = 8
    learning_rate = 0.05873
    subsample = 0.7532
    colsample_bytree = 0.466
    booster = 'gbtree'
    gamma = 0.2458
    eta = 0.4306
    min_child_weight = 1.246
    max_delta_step = 7
    reg_alpha = 0.06229
    reg_lambda = 2.904
    scale_pos_weight = 1
    objective = 'multi:softmax'
    num_class = 2

    sweep_config = dict(
        program='Cross_Validation.py',
        method='bayes',
        metric=dict(
            goal='maximize',
            name='mean_mcc'
        ),
        name='XGBoost Sweep 2',
        description='XGBoost Sweep Test',
        parameters=dict(
            booster=dict(
                values=['gbtree', 'dart']
            ),
            colsample_bytree=dict(
                distribution='uniform',
                max=1,
                min=0
            ),
            gamma=dict(
                distribution='uniform',
                max=10,
                min=0
            ),
            learning_rate=dict(
                distribution='uniform',
                max=0.1,
                min=0.01
            ),
            eta=dict(
                distribution='uniform',
                max=1,
                min=0.3
            ),
            max_delta_step=dict(
                distribution='int_uniform',
                max=10,
                min=1
            ),
            max_depth=dict(
                distribution='int_uniform',
                max=13,
                min=3
            ),
            min_child_weight=dict(
                distribution='uniform',
                max=10,
                min=0
            ),
            n_estimators=dict(
                distribution='int_uniform',
                max=1000,
                min=10
            ),
            reg_alpha=dict(
                distribution='uniform',
                max=10,
                min=0
            ),
            reg_lambda=dict(
                distribution='uniform',
                max=10,
                min=0
            ),
            scale_pos_weight=dict(
                distribution='int_uniform',
                max=10,
                min=1
            ),
            subsample=dict(
                distribution='uniform',
                max=1,
                min=0.5
            )
        )
    )

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
        run_name='XGBoost Optimised Uncertainty Estimation',
        notes='notes',
        test_repeats=[3, 5, 10, 30, 50, 80, 100, 250, 500, 650, 750, 850, 1000, 1250],
        n_repeats=13,
        n_samples=1.0,
        confidence_level=99,
        time_units='mins',
        time_threshold=None,
        ste_threshold=None,
        # model parameters
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        booster=booster,
        gamma=gamma,
        eta=eta,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
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
        run_name='XGBoost Default Cross Validation Round 1',
        run_sweep=False,
        sweep_name='XgBoost Round 2',
        k_folds=5,
        n_repeats=13,
        confidence_level=99,
        # Non-Sweep Model Parameters
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        booster=booster,
        gamma=gamma,
        eta=eta,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective=objective,
        num_class=num_class,
        # scale_pos_weight=scale_pos_weight
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
        run_name='XGBoost Feature Importance Agonist',
        notes='Notes',
        n_jobs=-1,
        target_datasets=[('x_train', 'y_train'), ('x_hold_out', 'y_hold_out')],
        method='shap',  # shap or default
        n_repeats=13,
        confidence_level=99,
        run_threshold_method=False,
        # model parameters
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        booster=booster,
        gamma=gamma,
        eta=eta,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight
    )

    return Obj(
        model_estimation_config=model_estimation_config,
        cross_validation_config=Obj(config=cross_validation_config, sweep_config=sweep_config),
        feature_importance_config=feature_importance_config
    )
