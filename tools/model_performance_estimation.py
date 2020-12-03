from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from tools.anonymousClass import Obj
from tools.model_tools import get_model_performance
from sklearn.utils import resample


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
                  desc=f'Cross-Validation Cycles: K={self.k_folds} R={self.n_repeats}') as progress_bar:

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
            hold_out_conf_matrice=hold_out_scores.conf_mat
        )


class BootstrapValidation:

    def __init__(self, n_repeats, n_samples, validation_size, data, make_model_func, model_parameters):

        self.n_repeats = n_repeats
        self.n_samples = n_samples
        self.validation_size = validation_size
        self.make_model_func = make_model_func
        self.model_parameters = model_parameters
        self.data = data

    def run(self):

        cross_val_mcc_scores = []
        cross_val_acc_scores = []
        cross_val_conf_matrices = []
        with tqdm(total=self.n_repeats, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                  desc=f'Bootstrap Cycles: R={self.n_repeats}') as progress_bar:

            for r in range(self.n_repeats):

                x_data, y_data = resample(
                    self.data.x_train, self.data.y_train, n_samples=self.n_samples, stratify=self.data.y_train
                )
                train_x, val_x, train_y, val_y = train_test_split(x_data, y_data, test_size=self.validation_size)

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
            hold_out_conf_matrice=hold_out_scores.conf_mat
        )