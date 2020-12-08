from tools.model_performance_estimation import CrossValidation
from tools.DatasetCompiler import DatasetCompiler
from tools import model_tools as m_tools
import tools.cloud_tools as c_tools
import matplotlib.pyplot as plt
import wandb
from tools.ImageSaver import ImageSaver
import copy
import numpy as np
import pandas as pd
from tools.make_models import make_model
import Filter_Script
import os
print(f'Program Running: {os.path.basename(__file__)}')


meta_data = Filter_Script.get_config().cross_validation_config

sweep_config = dict(
    program='Filtered_Cross_Validation.py',
    method='bayes',
    metric=dict(
        goal='maximize',
        name='mean_mcc'
    ),
    name=meta_data.sweep_name,
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

is_sweep = False


def run_cross_validation():
    # ===================================== Set Up File ===============================================
    run = c_tools.sweep_init(meta_data, is_sweep)
    config = wandb.config
    image_saver = ImageSaver(run)

    data = DatasetCompiler.load_from_pickle(config.src)

    # ================================== Run Cross Validation =======================================
    results = CrossValidation(config.k_folds, config.n_repeats, data, make_model(config), config).run()
    desc_stats = m_tools.get_descriptive_stats(results.cross_val_mcc_scores)

    # =================================== Log Data ==============================================
    if is_sweep:
        run.log({'mean_mcc': desc_stats.mean})
    else:
        # Get confidence levels
        stats = m_tools.get_normal_confidence_interval(
            results.cross_val_mcc_scores,
            confidence_level=config.confidence_level,
            score_range=(-1, 1)
        )
        # Log Performance
        val_median_conf_mat = m_tools.get_median_confusion_matrix(results.cross_val_conf_matrices)
        image_saver.save(plot=m_tools.plot_confusion_matrix(val_median_conf_mat),
                         name='Validation: Median confusion_matrix', format='png')
        image_saver.save(plot=m_tools.plot_confusion_matrix(results.hold_out_conf_matrice),
                         name='Hold Out Test: Confusion_matrix', format='png')
        labels, counts = np.unique(data.y_train, return_counts=True)
        run.log({'Class Balances Training Set': {'Class Labels': labels, 'Class Counts': counts}})
        image_saver.save(plot=m_tools.plot_performance(results.cross_val_mcc_scores, desc_stats),
                         name=f'MCC Cross Val Performance', format='png')

        # Log Confidence Interval
        col1 = ['Upper Bound', 'Lower Bound', 'Radius', 'distribution', 'method']
        col2 = [stats.upper_bound, stats.lower_bound, stats.radius, stats.distribution, stats.method]
        stat_df = pd.DataFrame(zip(col1, col2), columns=['Names', 'Values'])
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
        ax.hist(results.cross_val_mcc_scores)
        image_saver.save(plot=fig,
                         name='Cross Val of MCC Scores', format='png')
        run.log({f'{stats.method} Confidence Interval: CL={config.confidence_level}%': wandb.Table(dataframe=stat_df)})

        # Log Descriptives
        col1 = ['Validation mean', 'Validation median', 'Validation std', 'Validation ste', 'hold out MCC']
        col2 = [desc_stats.mean, desc_stats.median, desc_stats.std, desc_stats.ste, results.hold_out_mcc_score]
        stat_df = pd.DataFrame(zip(col1, col2), columns=['Names', 'Values'])
        run.log({'Cross Val MCC Descriptive Statistics': wandb.Table(dataframe=stat_df)})

        # Log Decision Tree Structure If Applicable
        image_saver.save_graphviz(
            graph_name='Decision Tree Structure',
            model=results.model,
            feature_names=data.feature_names,
            class_names=['ant', 'ag']
        )

        # Save Model Locally and in the Cloud
        meta_data(last_run_scores=dict(
            cross_val=results.cross_val_mcc_scores,
            hold_out=results.hold_out_mcc_score,
            model_performance=desc_stats.to_dict()
        ))

        meta_data_cloud = copy.deepcopy(meta_data.to_dict())
        artifact_model_name = f'CV-{config.artifact_name}'
        artifact_results = 'CV-model-results'
        artifact_data_name = 'CV-meta-data'
        path = m_tools.local_save_model(
            results.model, artifact_model_name, meta_data.to_dict(),
            return_path=True
        )

        c_tools.cloud_save(results.model, path.file_name, run)
        c_tools.cloud_save(results, f'{path.file_name}-{artifact_results}', run)
        c_tools.cloud_save(meta_data_cloud, f'{path.file_name}-{artifact_data_name}', run)


if meta_data.run_sweep:
    is_sweep = True
    sweep_id = wandb.sweep(sweep_config, project=meta_data.project_name)
    wandb.agent(sweep_id, function=run_cross_validation)
else:
    run_cross_validation()
