
config = {
    'project': 'dataset test',
    'wandb_key': '003bdcde0a7e623fdeb0425c3079a7aed09a32e6',

    'dataset': {
        'src': '../data/raw',
        'name': 'dataset test',
        'labels': 'target',
        'notes': 'Data contains B2in, Z, R and Merged datasets',
        'stats': dict(
            names=[],
            label_feature_name="target",
        ),
        'transforms': {
            'merge_datasets': dict(
                merge_all=True,
                merge_all_name='merged-B2in-Z-R',
                groups=[('B2in-ant', 'B2in-ag'), ('R-ant', 'R-ag'), ('Z-ant', 'Z-ag')],
                group_names=['B2in', 'R', 'Z']
            ),
            # 'drop_nans': dict(
            #     target_datasets=['B2in', 'R', 'Z']
            # )
        }
    },
    'models': {

        'decision_tree1': {
            'setup': dict(
                active=False,
                file="decision_tree",
                id="decision_tree_1",
                run_name='B2in',
                model_name="CART Model",
                dataset="B2in",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'models': dict(
                n_jobs=4,
                k_folds=10,
                learning_curve=True,
                class_names=['ant', 'ag'],
                criterion='gini',
                splitter='best',
                max_depth=None,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'random_forest1': {
            'setup': dict(
                active=True,
                file="random_forest",
                id="RDF B2in",
                run_name='RFC B2in',
                model_name="Standard RandomForest 1",
                dataset="B2in",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'models': dict(
                n_jobs=3,
                k_folds=10,
                learning_curve=True,
                n_estimators=100,
                max_features='auto',
                bootstrap=True,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'naive_model3': {
            'setup': dict(
                active=False,
                file="naive_model",
                id="naive-random-1",
                run_name='naive models random class',
                model_name="naive models random class",
                dataset="beta-2-ag-ant",
                y_labels="target",
                shuffle=True,
                dtype='int64'
            ),
            'models': dict(
                k_folds=10,
                model_type="random",
                scorer="Matthews Correlation Coefficient"
            )
        },

    }

}