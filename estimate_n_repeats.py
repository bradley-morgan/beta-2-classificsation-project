from tools.model_performance_estimation import ModelEstimations
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tools.anonymousClass import Obj
from tools.DatasetCompiler import DatasetCompiler
from tools import model_tools as m_tools
import matplotlib.pyplot as plt
import seaborn as sns
import time
import wandb
from tools.ImageSaver import ImageSaver


config = Obj(
    src='data/processed/non-filtered/lrg_clean_data_v2.pickle',
    project_name='b2ar-no-filter-test',
    run_name='Estimate Number of Repeats for Decision Tree',
    notes='To Calculate an more optimal number of repeats I first need to determine an estimate for the population'
          'Standard Error',
    cloud_log=True,
    test_mode=True,
    artifact_name='DecisionUnoptimisedBootstrap',
    is_d_tree=True,
    test_repeats=[3, 5, 8, 10, 30, 60, 100, 250, 500, 750, 1000, 1250, 1500],
    n_repeats=100,
    n_samples=0.8,
    validation_size=0.1,
    time_units='hours',
    time_threshold=None,
    ste_threshold=None
    # model parameters
)

make_model = lambda _: DecisionTreeClassifier()
emp = ModelEstimations(config, make_model, cloud_log=config.cloud_log)
# emp.estimate_population_standard_error()
#results = emp.estimate_n_repeats(0.99,  0.0008776974715241871, (0.001, 0.0001), 15)
emp.estimate_model_uncertainty()
