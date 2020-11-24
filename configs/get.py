
def general_config():
    parameters = dict(
        project_name='test-optim',
        run_name='test run',
        src='data/no-filter',
        y_labels='target',
        test_size=0.1,
        notes='This is test script for rfc sweeps',
        clean_features=dict(
            exceptions=['Action']
        ),
        rename_features=dict(
            renames=dict(Action='target')
        ),
        merge=dict(
            merge_all=True,
            merge_all_name='merged-3sn6-4lde-5jqh',
            groups=[('3sn6-ant', '3sn6-ag'), ('4lde-ant', '4lde-ag'), ('5jqh-ant', '5jqh-ag')],
            group_names=['3sn6', '4lde', '5jqh']
        ),
        remove_features=dict(
            search_params=["Clash", "Proximal"]
        ),
        change_nans=dict(
            value=0
        ),
        k_folds=3,
        repeats=3,
        n_estimators=100,
        max_features=11,
        max_depth=200
    )

    return parameters


def random_forest_sweep_config():
    parameters = dict(
        program='RFC_sweep.py',
        project='test-optim',
        method='bayes',
        metric=dict(
            name='Mean',
            goal='maximize'
        ),
        parameters=dict(
            n_estimators=dict(
                distribution='int_uniform',
                max=200,
                min=50
            ),
            max_features=dict(
                distribution='int_uniform',
                max=22,
                min=6
            ),
            max_depth=dict(
                distribution='int_uniform',
                max=200,
                min=50
            )
        )
    )

    return parameters
