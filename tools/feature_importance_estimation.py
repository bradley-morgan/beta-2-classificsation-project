from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.inspection import permutation_importance
from tools.DatasetCompiler import DatasetCompiler
from tools.ImageSaver import ImageSaver
from tools.anonymousClass import Obj
import pandas as pd
import tools.model_tools as m_tools
import wandb
from tqdm import tqdm

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
        self.target_datasets = target_datasets
        self.n_repeats = self.config.n_repeats
        self.confidence_level = self.config.confidence_level
        self.run_threshold_method = self.config.run_threshold_method
        self.n_jobs = self.config.n_jobs
        self.cloud_log = cloud_log

        self.data = DatasetCompiler.load_from_pickle(self.src).to_dict()

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
            stats = m_tools.get_normal_confidence_interval(score, self.confidence_level, (score.min(), score.max()), force_normal=False)
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

    def get_feature_importances(self):

        model = self.make_model(self.config)

        feature_importance = model.feature_importances_
        impurity_imp_df = pd.DataFrame(zip(self.data['feature_names'], feature_importance), columns=['Features', 'Gini Importance'])
        impurity_imp_df.sort_values(by='Gini Importance', ascending=False, inplace=True)
        self.run.log({f'Gini Feature Importances': wandb.Table(dataframe=impurity_imp_df)})

        with tqdm(total=len(self.target_datasets), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                  desc=f'Feature Importance Cycles: {len(self.target_datasets)}', position=0) as progress_bar:
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
                for idx, (perm_feat, gini_feat) in enumerate(zip(feature_importance['Feature Names'].to_numpy(), impurity_imp_df['Features'].to_numpy())):

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



