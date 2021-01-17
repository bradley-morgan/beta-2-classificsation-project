from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.inspection import permutation_importance
from tools.DatasetCompiler import DatasetCompiler
from tools.ImageSaver import ImageSaver
from tools.anonymousClass import Obj
import pandas as pd
import tools.model_tools as m_tools
import wandb
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# TODO Run Feature Perm Importances for Training Data [x]
# TODO Run Feature Perm Importances for Hold out Data []
# TODO Calculate confidence intervals for each feature for training perm importances []
# TODO Calculate confidence intervals for each feature for hold out perm importances []
# TODO Thresholding method for the Feature Importances

class FeatureImportances:

    def __init__(self, config: Obj, make_model_func, target_datasets=None, cloud_log=True):

        if target_datasets is None:
            target_datasets = ['x_train', 'y_train']

        self.config = config
        self.project_name = config.project_name
        self.notes = config.notes
        self.run_name = config.run_name
        self.src = self.config.src
        self.make_model = make_model_func
        self.method = config.method
        self.target_datasets = target_datasets
        self.n_repeats = self.config.n_repeats
        self.confidence_level = self.config.confidence_level
        self.run_threshold_method = self.config.run_threshold_method
        self.n_jobs = self.config.n_jobs
        self.cloud_log = cloud_log

        self.data = DatasetCompiler.load_from_local(self.src).to_dict()

        if self.cloud_log:
            config = self.config.to_dict()
            self.run = wandb.init(
                config=config,
                project=self.project_name,
                notes=self.notes,
                allow_val_change=True,
                name=self.run_name,
            )
            self.image_saver = ImageSaver(self.run)

    def get_confidence_levels(self, feat_import_data):

        out = Obj(
            lower_bounds=[],
            upper_bounds=[],
            radii=[],
            stds=[],
            stes=[],
            confidences=[],
            repeats=[]
        )
        for score in feat_import_data.importances:
            stats = m_tools.get_normal_confidence_interval(score, self.confidence_level, (score.min(), score.max()))
            desc_stats = m_tools.get_descriptive_stats(score)
            out.lower_bounds.append(stats.lower_bound)
            out.upper_bounds.append(stats.upper_bound)
            out.radii.append(stats.radius)
            out.stds.append(desc_stats.std)
            out.stes.append(desc_stats.ste)
            out.confidences.append(self.confidence_level)
            out.repeats.append(self.n_repeats)

        return out

    def format_importances(self, stats, importances, feature_names):
        feature_importances = pd.DataFrame(
            zip(feature_names,
                importances,
                stats.lower_bounds,
                stats.upper_bounds,
                stats.radii,
                stats.stds,
                stats.stes,
                stats.confidences,
                stats.repeats
                ),
            columns=[
                'Feature Names',
                'Permutation Score MCC',
                'Lower Bound',
                'Upper Bound',
                'Confidence Interval Radius',
                'ste',
                'std',
                'CL',
                'Permutation Repeats'
            ]
        )

        feature_importances.sort_values(by='Permutation Score MCC', ascending=False, inplace=True)
        return feature_importances

    def default_feature_importance(self, model):

        feature_importance = model.feature_importances_
        impurity_imp_df = pd.DataFrame(zip(self.data['feature_names'], feature_importance),
                                       columns=['Features', 'Gini Importance'])
        impurity_imp_df.sort_values(by='Gini Importance', ascending=False, inplace=True)
        self.run.log({f'Gini Feature Importances': wandb.Table(dataframe=impurity_imp_df)})

        with tqdm(total=len(self.target_datasets), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                  desc=f'Feature Importance Cycles: {len(self.target_datasets)}, Repeats: {self.n_repeats}',
                  position=0) as progress_bar:
            for dataset in self.target_datasets:
                x = dataset[0]
                y = dataset[1]
                x_data = self.data[x]
                y_data = self.data[y]

                if self.config.load_model_from == 'train':
                    model.fit(x_data, y_data)

                # Compute Permutation Importance for training data
                scorer = make_scorer(matthews_corrcoef)
                feature_importance = permutation_importance(model,
                                                            x_data,
                                                            y_data,
                                                            scoring=scorer,
                                                            n_repeats=self.n_repeats,
                                                            n_jobs=self.n_jobs)

                # feature_importances = joblib.load('../mock_data/no-filter-d-tree-perm-imp.joblib')
                stats = self.get_confidence_levels(feature_importance)
                feature_importance = self.format_importances(
                    stats, feature_importance["importances_mean"], self.data['feature_names']
                )

                # Calculate the agreement between gini and permutation importances
                agree_dict = {'Feature Rank': [], 'Gini vs Perm': [], 'Gini Feat': [], 'Perm Feat': []}
                for idx, (perm_feat, gini_feat) in enumerate(
                        zip(feature_importance['Feature Names'].to_numpy(), impurity_imp_df['Features'].to_numpy())):

                    agree_dict['Feature Rank'].append(idx)
                    if perm_feat == gini_feat:
                        agree_dict['Gini vs Perm'].append('Agree')
                    else:
                        agree_dict['Gini vs Perm'].append('Disagree')

                    agree_dict['Gini Feat'].append(gini_feat)
                    agree_dict['Perm Feat'].append(perm_feat)

                gini_vs_perm_df = pd.DataFrame.from_dict(agree_dict)

                if self.run_threshold_method:
                    pass

                if not self.cloud_log:
                    return

                self.run.log({f'Permutation Feature Importances: {x}': wandb.Table(dataframe=feature_importance)})
                self.run.log({f'Permutation vs Gini Agreement: {x}': wandb.Table(dataframe=gini_vs_perm_df)})

                progress_bar.update(1)

    def correlate_shaps(self, shap, X, y, feature_names):
        # import matplotlib as plt
        # Make a copy of the input data
        feature_order_df = pd.DataFrame(shap, columns=feature_names).abs().mean(axis=0).sort_values(ascending=False)

        class_ant_train = pd.DataFrame(X[np.where(y == 0)[0], :], columns=feature_names)
        class_ag_train =  pd.DataFrame(X[np.where(y == 1)[0], :], columns=feature_names)

        class_ant_shap = pd.DataFrame(shap[np.where(y == 0)[0], :], columns=feature_names)
        class_ag_shap =  pd.DataFrame(shap[np.where(y == 1)[0], :], columns=feature_names)

        # Determine the correlation in order to plot with different colors
        ant_corr_list = list()
        ag_corr_list = list()
        for i in feature_names:

            ant_corr = np.corrcoef(class_ant_shap[i], class_ant_train[i])[1][0]
            ag_corr = np.corrcoef(class_ag_shap[i], class_ag_train[i])[1][0]

            ant_corr_list.append(ant_corr)
            ag_corr_list.append(ag_corr)

        corr_df = pd.concat([pd.Series(feature_names), pd.Series(ant_corr_list), pd.Series(ag_corr_list)], axis=1).fillna(0)
        # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
        corr_df.columns = ['Feature', 'Ant Corr', 'Ag Corr']
        corr_df.insert(2, 'Ant Sign', np.where(corr_df['Ant Corr'] > 0, 'Pos', 'Neg'))
        corr_df.insert(4, 'Ag Sign', np.where(corr_df['Ag Corr'] > 0, 'Pos', 'Neg'))

        corr_df.set_index('Feature', inplace=True)
        correl = corr_df.loc[feature_order_df.index.values]
        correl.insert(0, 'Mean Absolute SHAP Importance', feature_order_df)
        correl.reset_index(inplace=True)
        correl.index.name = 'Order of Importance'

        return correl

    def shap_feature_importance(self, model):

        if self.config.load_model_from == 'train':
            model.fit(self.data['x_train'], self.data['y_train'])
        
        with tqdm(total=len(self.target_datasets), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                  desc=f'Feature Importance Cycles: {len(self.target_datasets)}, Repeats: {self.n_repeats}',
                  position=0) as progress_bar:
            
            for dataset in self.target_datasets:
                x = dataset[0]
                y = dataset[1]
                x_data = self.data[x]
                y_data = self.data[y]
        
                shap_vales = shap.TreeExplainer(model).shap_values(x_data)
                # shap_vales = shap_vales[1]
                correl = self.correlate_shaps(shap_vales, x_data ,y_data, self.data['feature_names'])
        
                # m_tools.local_save_model(correl, 'd_tree_feature_imp.joblib', None, dir_path='./analysis')
                correl.to_csv('./analysis/xgboost_feature_imp_agonist.csv')
        
                self.run.log({f'Feature Importance Correlation Table {x}': wandb.Table(dataframe=correl)})

                self.image_saver.save_graphviz(
                    model, self.data['feature_names'], ['ant', 'ag'], 'Tree Structure'
                )
        
                sns.set()
                shap.summary_plot(
                    shap_values=shap_vales,
                    features=x_data,
                    feature_names=self.data['feature_names'],
                    show=False,
                    plot_type='dot',
                    max_display=10
                )
                plt.tight_layout()
                self.image_saver.save(plot=plt.gcf(),
                                      name=f'Local Feature Importance {x}', format='png')
                plt.clf()
        
                shap.summary_plot(
                    shap_vales,
                    x_data,
                    plot_type='bar',
                    feature_names=self.data['feature_names'],
                    show=False,
                    max_display=10
                )
                self.image_saver.save(plot=plt.gcf(),
                                      name=f'Global Feature Importance {x}', format='png')


                progress_bar.update()
        
                # fig2 = shap.dependence_plot(
                #     '193/CB - /1/C Hydrophobic',
                #     shap_vales,
                #     x_data,
                #     feature_names=self.data['feature_names'],
                #     x_jitter=0.05,
                #     color='#000000',
                #     show=False
                # )
                #
                # self.image_saver.save(plot=plt.gcf(),
                #                       name='Test2', format='png')

    def get_feature_importances(self):

        model = self.make_model(self.config)

        if self.method.lower() == 'default':
            self.default_feature_importance(model)

        elif self.method.lower() == 'shap':
            self.shap_feature_importance(model)

        else:
            raise ValueError(f'Invalid Method {self.method}: Select shap or default (gini & permutation)' )





