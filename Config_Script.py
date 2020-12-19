from tools.anonymousClass import Obj
from tools.make_models import make_model

EXPERIMENT = 'filtered'


def get_config():
    if EXPERIMENT.lower() == 'filtered':
        return get_filtered_config()

    elif EXPERIMENT.lower() == 'unfiltered':
        return get_unfiltered_config()


def get_filtered_config():
    global_src = 'data/processed/filtered//dataset1-2percent-hold-out.pickle'
    global_project_name = 'B2AR-Filtered'
    global_cloud_log = True
    global_test_mode = False
    global_artifact_name = 'DecisionTree'
    global_model = 'decision_tree'
    global_load_model_from = 'train'
    global_load_run_path = 'bradamorg/B2AR-Filtered/23xzu26f'
    global_model_file_name = 'v15_CV-DecisionTree.joblib'

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
    max_depth = 10
    max_features = None
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

    sweep_config = dict(
        program='Cross_Validation.py',
        method='bayes',
        metric=dict(
            goal='maximize',
            name='mean_mcc'
        ),
        name='D_Tree Cross Validation Round 1',
        description='Decision Tree Sweep Test',
        parameters=dict(
            # criterion=dict(
            #     values=['gini', 'entropy']
            # ),
            # splitter=dict(
            #     values=['best', 'random']
            # ),
            max_depth=dict(
                distribution='int_uniform',
                max=5,
                min=3
            ),
            # max_features=dict(
            #     distribution='int_uniform',
            #     max=161,
            #     min=1
            # ),
            # min_samples_split=dict(
            #     distribution='int_uniform',
            #     max=1000,
            #     min=1
            # ),
            # min_samples_leaf=dict(
            #     distribution='int_uniform',
            #     max=1000,
            #     min=1
            # ),
            # class_weight=dict(
            #     values=['balanced', None]
            # ),
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
        run_name='Max Depth 10',
        notes='notes',
        test_repeats=[3, 5, 10, 30, 50, 80, 100, 250, 500, 650, 750, 850, 1000, 1250, 1500, 1750, 2000],
        n_repeats=3,
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
        run_name='Max Depth 10',
        run_sweep=False,
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
        run_name='test feat imp',
        notes='Notes',
        n_jobs=-1,
        target_datasets=[('x_train', 'y_train'), ('x_hold_out', 'y_hold_out')],
        method='shap',  # shap or default
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
        cross_validation_config=Obj(config=cross_validation_config, sweep_config=sweep_config),
        feature_importance_config=feature_importance_config
    )


def get_unfiltered_config():
    global_src = 'data/processed/non-filtered/dataset1-2percent-hold-out.pickle'
    global_project_name = 'B2AR-Unfiltered-test'
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
        scale_pos_weight=scale_pos_weight
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
        scale_pos_weight=scale_pos_weight
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
        run_name='XGBoost Feature Importance',
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
