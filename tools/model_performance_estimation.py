from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from tools.anonymousClass import Obj
from tools.model_tools import get_model_performance
from tools.DatasetCompiler import DatasetCompiler
import tools.cloud_tools as c_tools
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tools.ImageSaver import ImageSaver
import numpy as np
import wandb
import tools.model_tools as m_tools
import pandas as pd
import copy
import matplotlib.gridspec as gridspec


class CrossValidation:

    def __init__(self, k_folds, n_repeats, data, make_model_func, model_parameters):

        self.k_folds = k_folds
        self.n_repeats = n_repeats
        self.make_model_func = make_model_func
        self.model_parameters = model_parameters
        self.data = data

    def run(self):
        # Cross validation
        cross_val_mcc_scores = []
        cross_val_acc_scores = []
        cross_val_conf_matrices = []
        with tqdm(total=self.n_repeats * self.k_folds, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                  desc=f'Cross-Validation Cycles: K={self.k_folds} R={self.n_repeats}', position=0) as progress_bar:

            for r in range(self.n_repeats):

                cv = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
                for train_idx, test_idx in cv.split(self.data.x_train, self.data.y_train):
                    # extract hold out test set
                    train_x, val_x = self.data.x_train[train_idx], self.data.x_train[test_idx]
                    train_y, val_y = self.data.y_train[train_idx], self.data.y_train[test_idx]

                    cross_val_model = self.make_model_func(self.model_parameters)

                    # Fit & Cross validate
                    cross_val_model.fit(train_x, train_y)
                    y_preds = cross_val_model.predict(val_x)

                    cross_val_scores = get_model_performance(y_true=val_y, y_preds=y_preds)

                    cross_val_mcc_scores.append(cross_val_scores.mcc_score)
                    cross_val_acc_scores.append(cross_val_scores.acc_score)
                    cross_val_conf_matrices.append(cross_val_scores.conf_mat)
                    progress_bar.update(1)

        # Hold out Evaluation: Train model on whole data-set then do final unseen test
        hold_out_model = self.make_model_func(self.model_parameters)

        hold_out_model.fit(self.data.x_train, self.data.y_train)
        hold_out_y_preds = hold_out_model.predict(self.data.x_hold_out)
        hold_out_scores = get_model_performance(y_true=self.data.y_hold_out, y_preds=hold_out_y_preds)

        return Obj(
            cross_val_mcc_scores=cross_val_mcc_scores,
            cross_val_acc_scores=cross_val_acc_scores,
            cross_val_conf_matrices=cross_val_conf_matrices,
            hold_out_mcc_score=hold_out_scores.mcc_score,
            hold_out_acc_score=hold_out_scores.acc_score,
            hold_out_conf_matrice=hold_out_scores.conf_mat,
            model=hold_out_model
        )


class BootstrapValidation:

    def __init__(self, n_repeats, n_samples, data, make_model_func, model_parameters):
        self.n_repeats = n_repeats
        self.n_samples = n_samples
        self.make_model_func = make_model_func
        self.model_parameters = model_parameters
        self.data = data

    def split(self, data: pd.DataFrame, y_label_name):
        x = data.drop([y_label_name], axis=1).to_numpy()
        y = data[y_label_name].to_numpy()
        return x, y

    def run(self):
        # Convert data to dataframes so that we can preserve the indexes. This will allow to extract the out of bag test
        data = pd.DataFrame(self.data.x_train, columns=self.data.feature_names)
        data['target'] = self.data.y_train

        cross_val_mcc_scores = []
        cross_val_acc_scores = []
        cross_val_conf_matrices = []
        with tqdm(total=self.n_repeats, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                  desc=f'Bootstrap Cycles: R={self.n_repeats}', position=0) as progress_bar:
            for r in range(self.n_repeats):
                train_n_samples = int(np.floor(len(self.data.y_train) * self.n_samples))
                test_n_samples = len(self.data.y_train) - train_n_samples

                sampled_train = resample(
                    data, n_samples=train_n_samples, stratify=data['target'].values
                )

                # Extract the out-of-bag samples for test data and resample so we get a stratified distribution
                out_bag_test = data.loc[~data.index.isin(sampled_train.index)]
                sampled_test = resample(
                    out_bag_test, stratify=out_bag_test['target'].values
                )

                train_x, train_y = self.split(sampled_train, 'target')
                val_x, val_y = self.split(sampled_test, 'target')

                cross_val_model = self.make_model_func(self.model_parameters)

                # Fit & Cross validate
                cross_val_model.fit(train_x, train_y)
                y_preds = cross_val_model.predict(val_x)
                cross_val_scores = get_model_performance(y_true=val_y, y_preds=y_preds)

                cross_val_mcc_scores.append(cross_val_scores.mcc_score)
                cross_val_acc_scores.append(cross_val_scores.acc_score)
                cross_val_conf_matrices.append(cross_val_scores.conf_mat)
                progress_bar.update(1)

        # Hold out Evaluation: Train model on whole data-set then do final unseen test
        hold_out_model = self.make_model_func(self.model_parameters)

        hold_out_model.fit(self.data.x_train, self.data.y_train)
        hold_out_y_preds = hold_out_model.predict(self.data.x_hold_out)
        hold_out_scores = get_model_performance(y_true=self.data.y_hold_out, y_preds=hold_out_y_preds)

        return Obj(
            cross_val_mcc_scores=cross_val_mcc_scores,
            cross_val_acc_scores=cross_val_acc_scores,
            cross_val_conf_matrices=cross_val_conf_matrices,
            hold_out_mcc_score=hold_out_scores.mcc_score,
            hold_out_acc_score=hold_out_scores.acc_score,
            hold_out_conf_matrice=hold_out_scores.conf_mat,
            model=hold_out_model
        )


class ModelEstimations:

    def __init__(self, config: Obj, make_model_func, cloud_log=True):
        self.meta_data = config
        self.src = config.src
        self.project_name = config.project_name
        self.run_name = config.run_name
        self.notes = config.notes
        self.artifact_name = config.artifact_name
        self.test_mode = config.test_mode
        self.make_model_func = make_model_func
        self.time_units = config.time_units
        self.time_threshold = config.time_threshold
        self.ste_threshold = config.ste_threshold
        self.is_d_tree = config.is_d_tree

        self.pse_data = Obj()
        self.m_uncertainty_data = Obj()
        self.data = DatasetCompiler.load_from_pickle(self.src)

        self.cloud_log = cloud_log

        if config.time_units == 'mins' or config.time_units == 'm':
            self.time_units = 60.0
            self.time_unit_name = 'Minutes'

        elif config.time_units == 'hours' or config.time_units == 'h':
            self.time_units = 3600.0
            self.time_unit_name = 'Hours'
        else:
            self.time_units = 1.0
            self.time_unit_name = 'Seconds'

        if self.cloud_log:
            config = self.meta_data.to_dict()
            self.run = wandb.init(
                config=config,
                project=self.project_name,
                notes=self.notes,
                allow_val_change=True,
                name=self.run_name,
            )
            self.config = wandb.config
            self.image_saver = ImageSaver(self.run)

    def get_ste(self, test_repeats, repeat_num):
        return self.pse_data.ste_scores[test_repeats.index(repeat_num)]

    def estimate_population_variance(self, test_repeats, n_samples):

        # TODO Elbow the population std estimation  Estimate the standard error start naively maybe we can fit a function to it
        ste_scores = []
        std_scores = []
        run_times = []
        for repeat in test_repeats:
            start = time.time()
            results = BootstrapValidation(
                n_repeats=repeat, n_samples=n_samples, data=self.data, make_model_func=self.make_model_func,
                model_parameters=self.config,
            ).run()

            end = time.time()
            time_elapsed = (end - start) / self.time_units
            desc_stats = m_tools.get_descriptive_stats(results.cross_val_mcc_scores)

            if self.time_threshold and time_elapsed > self.time_threshold:
                break
            elif self.ste_threshold and desc_stats.ste < self.ste_threshold:
                break

            ste_scores.append(desc_stats.ste)
            std_scores.append(desc_stats.std)
            run_times.append(time_elapsed)

        self.pse_data(
            ste_scores=ste_scores,
            std_scores=std_scores,
            run_times=run_times,
            optimal_ste=min(ste_scores),
            optimal_n_repeats_ste=test_repeats[np.argmin(ste_scores)],
            optimal_std=min(std_scores),
            optimal_n_repeats_std=test_repeats[np.argmin(std_scores)],
            is_ste_normal=m_tools.is_normal_distribution(ste_scores) if len(ste_scores) > 8 else None,
            is_std_normal=m_tools.is_normal_distribution(std_scores) if len(std_scores) > 8 else None
        )

        if not self.cloud_log:
            return
        # Log Data to Cloud
        pse_df = pd.DataFrame(
            zip(test_repeats, self.pse_data.ste_scores, self.pse_data.std_scores, self.pse_data.run_times),
            columns=['Repeats', 'Standard Error', 'Standard Deviation', 'Run Times'])

        self.run.log({'Population Variance Error Estimates': wandb.Table(dataframe=pse_df)})
        self.run.log({'Population Variance Error': self.pse_data.to_dict()})


        sns.set()
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(15, 10))
        ax = plt.subplot(gs[0, 0])
        ax.plot(ste_scores, marker='o')
        ax.set(xlabel='n repeats', ylabel='standard error')
        ax.set_xticks(range(0, len(self.config.test_repeats)))
        ax.set_xticklabels(self.config.test_repeats)
        ax.title.set_text('Standard Error Estimation')
        ax.grid(b=True)

        ax3 = plt.subplot(gs[0, 1])
        ax3.plot(std_scores, marker='o')
        ax3.set(xlabel='n repeats', ylabel='standard Deviation')
        ax3.set_xticks(range(0, len(self.config.test_repeats)))
        ax3.set_xticklabels(self.config.test_repeats)
        ax3.title.set_text('Standard Deviation Estimation')
        ax3.grid(b=True)

        ax2 = plt.subplot(gs[1, :])
        ax2.plot(run_times, marker='o')
        ax2.set(xlabel='n repeats', ylabel=f'run time ({self.time_unit_name})')
        ax2.set_xticks(range(0, len(self.config.test_repeats)))
        ax2.set_xticklabels(self.config.test_repeats)
        ax2.grid(b=True)
        ax2.title.set_text('Run Times')
        plt.rcParams['xtick.labelsize'] = 10
        plt.tight_layout()

        self.image_saver.save(plot=fig,
                              name='Elbow Method Standard Error Estimation', format='png')

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 7))
        ax.hist(ste_scores)
        ax.set(xlabel='Standard Error')
        ax.grid(b=True)

        ax2.hist(std_scores)
        ax2.set(xlabel='Standard Deviations')
        ax2.grid(b=True)
        plt.tight_layout()

        self.image_saver.save(plot=fig,
                              name='Standard Error Distribution', format='png')

    def estimate_n_repeats(self, confidence_level, population_std, margin_error_range: tuple, n_samples: int):

        margin_errors = np.linspace(margin_error_range[0], margin_error_range[1], n_samples)

        estimated_repeats = []
        for m_error in margin_errors:
            r = m_tools.get_n_repeats_estimation(confidence_level, population_std, m_error)
            estimated_repeats.append(r)

        self.m_uncertainty_data(
            estimated_repeats=estimated_repeats,
            marginal_errors=margin_errors
        )

        if not self.cloud_log:
            return

        r_df = pd.DataFrame(
            zip(estimated_repeats, margin_errors, [population_std] * len(margin_errors),
                [confidence_level] * len(margin_errors)
                ),
            columns=['Estimated Repeats', 'Marginal Errors', 'Estimated Population Std', 'Confidence Level']
        )

        self.run.log({'Estimated Number of Repeats': wandb.Table(dataframe=r_df)})
        sns.set()
        fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
        ax.plot(estimated_repeats, marker='o')
        ax.set(xlabel='Marginal Errors', ylabel='Estimated Number of Repeats')
        x = [np.format_float_scientific(x, precision=2) for x in margin_errors]
        ax.set_xticks(range(0, len(x)))
        ax.set_xticklabels(x)
        ax.title.set_text(f'Change in the estimated number of repeats as a function of marginal Error'
                          f' with a {confidence_level}% CL')
        ax.grid(b=True)
        self.image_saver.save(plot=fig,
                              name='Estimated Number of Repeats as Function of Marginal Error', format='png')

    def estimate_model_uncertainty(self, confidence_level, n_repeats, n_samples):

        results = BootstrapValidation(
            n_repeats=n_repeats, n_samples=n_samples,
            data=self.data, make_model_func=self.make_model_func, model_parameters=self.config,
        ).run()

        stats = m_tools.get_normal_confidence_interval(
            results.cross_val_mcc_scores,
            confidence_level=confidence_level,
            score_range=(-1, 1)
        )
        desc_stats = m_tools.get_descriptive_stats(results.cross_val_mcc_scores)

        # Log Performance
        val_median_conf_mat = m_tools.get_median_confusion_matrix(results.cross_val_conf_matrices)
        self.image_saver.save(plot=m_tools.plot_confusion_matrix(val_median_conf_mat),
                              name='Validation: Median confusion_matrix', format='png')
        self.image_saver.save(plot=m_tools.plot_confusion_matrix(results.hold_out_conf_matrice),
                              name='Hold Out Test: Confusion_matrix', format='png')
        labels, counts = np.unique(self.data.y_train, return_counts=True)
        self.run.log({'Class Balances Training Set': {'Class Labels': labels, 'Class Counts': counts}})
        self.image_saver.save(plot=m_tools.plot_performance(results.cross_val_mcc_scores, desc_stats),
                              name=f'MCC Bootstrap Performance', format='png')

        # Log Confidence Interval
        col1 = ['Upper Bound', 'Lower Bound', 'Radius', 'distribution', 'method']
        col2 = [stats.upper_bound, stats.lower_bound, stats.radius, stats.distribution, stats.method]
        stat_df = pd.DataFrame(zip(col1, col2), columns=['Names', 'Values'])
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
        ax.hist(results.cross_val_mcc_scores)
        self.image_saver.save(plot=fig,
                              name='Bootstrap Distribution of MCC Scores', format='png')
        self.run.log({f'Bootstrap Confidence Interval: CL={confidence_level}%': wandb.Table(dataframe=stat_df)})

        # Log Descriptives
        col1 = ['mean', 'median', 'std', 'ste']
        col2 = [desc_stats.mean, desc_stats.median, desc_stats.std, desc_stats.ste]
        stat_df = pd.DataFrame(zip(col1, col2), columns=['Names', 'Values'])
        self.run.log({'Bootstrap MCC Descriptive Statistics': wandb.Table(dataframe=stat_df)})

        # Log Decision Tree Structure If Applicable
        if self.is_d_tree:
            self.image_saver.save_graphviz(
                graph_name='Decision Tree Structure',
                model=results.model,
                feature_names=self.data.feature_names,
                class_names=['ant', 'ag']
            )

        # Save Model Locally and in the Cloud
        self.meta_data(last_run_scores=dict(
            cross_val=results.cross_val_mcc_scores,
            hold_out=results.hold_out_mcc_score,
            model_performance=desc_stats.to_dict()
        ))
        meta_data_cloud = copy.deepcopy(self.meta_data.to_dict())
        artifact_model_name = f'EU-{self.artifact_name}'
        artifact_results = 'EU-model-results'
        artifact_data_name = 'EU-meta-data'
        path = m_tools.local_save_model(
            results.model, artifact_model_name, self.meta_data.to_dict(),
            return_path=True
        )

        c_tools.cloud_save(results.model, path.file_name, self.run)
        c_tools.cloud_save(results, f'{path.file_name}-{artifact_results}', self.run)
        c_tools.cloud_save(meta_data_cloud, f'{path.file_name}-{artifact_data_name}', self.run)