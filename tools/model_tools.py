from joblib import dump, load
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
import numpy as np
from tools.anonymousClass import Obj
from scipy.stats import wilcoxon, norm, normaltest
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix


def local_load_model(file_name):
    model = load(file_name)
    return model


def local_save_model(model, file_name: str, mete_data: dict, overwrite=False, return_path=False):
    dir_path = './saved_models'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if not file_name.endswith('.joblib'):
        file_name = f'{file_name}.joblib'

    if not overwrite:
        if os.path.exists(os.path.join(dir_path, file_name)):
            print(f'The File {file_name} Already Exists! Creating a new version...')
            i = 2
            new_file_name = file_name
            while os.path.exists(os.path.join(dir_path, new_file_name)):
                new_file_name = f'v{i}_{file_name}'
                i += 1
            file_name = new_file_name
            print(f'New version for file: {file_name}')

    data = Obj(model=model, mete_data=mete_data)
    dump(data, os.path.join(dir_path, file_name))
    print('Model Successfully saved')

    if return_path:
        return os.path.join(dir_path, file_name)


def get_median_confusion_matrix(conf_mat: list):
    mat = np.asarray(conf_mat)
    median_confusion_matrix = np.median(mat, axis=0)
    return median_confusion_matrix.astype('int64')


def plot_confusion_matrix(conf_mat):
    ax = sns.heatmap(conf_mat,
                     annot=True,
                     cbar=False,
                     fmt='d')
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.close(ax.get_figure())
    plot = ax.get_figure()
    return plot


def plot_performance(scores, stats: Obj):
    sns.set()
    plt.plot(scores, linewidth=2.0)
    plt.title(
        f"Validation MCC Model Performance: Mean={np.round(stats.mean, 4)}  "
        f"Std={np.round(stats.std, 4)}  Standard Error={np.round(stats.ste, 4)}")
    plt.xlabel('K Folds')
    plt.ylabel('Score')
    return plt.gcf()


def get_model_performance(y_true, y_preds):
    conf_mat = confusion_matrix(y_true, y_preds)
    mcc_score = matthews_corrcoef(y_true=y_true, y_pred=y_preds)
    acc_score = accuracy_score(y_true=y_true, y_pred=y_preds)

    return Obj(
        conf_mat=conf_mat,
        mcc_score=mcc_score,
        acc_score=acc_score
    )


def is_normal_distribution(score):
    _, p = normaltest(score)
    return p < 1e-3


def get_z_score(confidence_level):
    area_under_curve = (1+confidence_level)/2
    z_score = np.round(norm.ppf(area_under_curve), decimals=2)
    return z_score


def get_descriptive_stats(scores):
    return Obj(
            mean=np.mean(scores),
            median=np.median(scores),
            std=np.std(scores),
            ste=sem(scores),
        )


def get_wilcoxon_stats(scores_A, scores_B, alpha=0.05, alternative_hypothesis='two-sided'):
    # Get descriptives
    descriptives = Obj(
        A=get_descriptive_stats(scores_A),
        B=get_descriptive_stats(scores_B)
    )

    # Get inferential
    statistic, pvalue = wilcoxon(x=scores_A, y=scores_B, alternative=alternative_hypothesis)
    # Get one tailed pvalue
    is_significant = pvalue < alpha
    # Wandb does not like bool types due to not being JSON serializable so changed to strings
    return Obj(
        test=' Wilcoxon signed-rank',
        alternative_hypothesis=alternative_hypothesis,
        descriptives=descriptives,
        statistic=statistic,
        pvalue=pvalue,
        alpha=alpha,
        theoretical_median=scores_B,
        is_significant='true' if is_significant else 'false',
        run_time_errors='true' if np.isnan(statistic) or np.isnan(pvalue) else 'false'
    )


def get_normal_confidence_interval(scores: list, confidence_level, score_range:tuple):
    # This method is based on the assumption that samples are normally distributed check this and report a warning
    # if samples are non-normal
    distribution = None
    if not is_normal_distribution(scores):
        distribution = 'DataNotNormal'
        output = get_resampled_confidence_interval(
            scores,
            score_range,
            100-confidence_level
        )
        output(distribution=distribution)
        return output

    distribution = 'DataNormal'

    confidence_level = confidence_level / 100 if confidence_level > 1.0 else confidence_level
    desc_stats = get_descriptive_stats(scores)
    z_score = get_z_score(confidence_level)
    # Normal Distribution in symmetric so we only have to calculate interval once
    confidence_interval = desc_stats.mean + z_score * desc_stats.std / np.sqrt(len(scores))
    upper_bound = desc_stats.mean + confidence_interval
    lower_bound = desc_stats.mean - confidence_interval
    radius = upper_bound - lower_bound

    return Obj(
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        radius=radius,
        distribution=distribution
    )


def get_binomial_confidence_interval(accuracy: int, sample_size: int, alpha: float):
    lower_bound, upper_bound = proportion_confint(accuracy, sample_size, alpha)
    radius = upper_bound - lower_bound
    return Obj(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        interval_radius=radius
    )


def get_resampled_confidence_interval(scores:list, score_range, alpha:int):
    lower_p = alpha / 2
    lower_bound = max(float(score_range[0]), np.percentile(scores, lower_p))

    upper_p = (100 - alpha) + (alpha / 2)
    upper_bound = min(float(score_range[1]), np.percentile(scores, upper_p))

    radius = upper_bound - lower_bound
    return Obj(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        radius=radius
    )


def get_n_repeats_estimation(confidence_level, population_standard_deviation, margin_of_error):
    """
    Estimates the number of samples, in this context, number of repeats needed in order to be sure relative to
    a given confidence level that the estimate will be within the specified margin of error range.
    :param confidence_level: E.g 95% Confidence level which is the same an alpha of 0.05 when doing p value confidence
    :param population_standard_deviation:
     Can be an estimate of the standard deviation of the population. Can be obtained by running many bootstrap experiments
     and calculating the standard error. (Standard error is an estimation of the population standard deviation)
    :param margin_of_error: Specified level of precision you can live with. E.g be accurate within 0.1 at 95%
    confidence level
    :return: Estimated Number of repeats
    """
    z_score = get_z_score(confidence_level)
    estimated_n_repeats = z_score**2 * (population_standard_deviation**2 / margin_of_error**2)
    return estimated_n_repeats



def get_elbow_precision(sample_sizes, validator, alpha, confidence_interval='resample', score_range=None ):
    """

    :param sample_sizes: Array of sample Sizes to test over
    :param validator: model_performance_estimation class Boostrap or CrossValidation
    :param alpha: Confidence level E.g 5% or 0.05 probability
    :param confidence_interval: Method used to calculate the confidence interval. For Non-Normal Distributions
     'get_resampled_confidence_interval'. For Normally distributed data 'get_binomial_confidence_interval'
    :param score_range: if confidence_interval = get_resampled_confidence_interval the define the minimum and maximum
    ranges of the model performance metric, E.g accuracy is defined in a range from 0-1.
    """
    pass