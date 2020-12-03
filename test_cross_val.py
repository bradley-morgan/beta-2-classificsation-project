from tools.model_performance_estimation import CrossValidation, BootstrapValidation
from sklearn.tree import DecisionTreeClassifier
from tools.anonymousClass import Obj
from tools.DatasetCompiler import DatasetCompiler
from tools import model_tools as m_tools
import matplotlib.pyplot as plt


config = Obj(
    src='data/processed/non-filtered/lrg_clean_data_v2.pickle',
    project_name='b2ar-filtered',
    notes='Cross Validation Decision Tree',
    test_mode=True,
    k_folds=10,
    n_repeats=100,
    alpha=0.05,
    alternative_hypothesis='greater',
    artifact_name='FilteredDecisionTreeClassifier'
    # Model Parameters
)

data = DatasetCompiler.load_from_pickle(config.src)
make_model = lambda m_config: DecisionTreeClassifier() if m_config.test_mode else DecisionTreeClassifier(max_depth=5)
# results = BootstrapValidation(
#     n_repeats=100, n_samples=0.8, validation_size=0.1, data=data, make_model_func=make_model, model_parameters=config
# ).run()

results = CrossValidation(10, 10, data, make_model, config).run()



# results = CrossValidation(5, 100 , data, make_model, config).run()
plt.hist(results.cross_val_mcc_scores)
plt.show()
mcc_confidence_interval = m_tools.get_normal_confidence_interval(
    results.cross_val_mcc_scores, score_range=[-1,1], alpha=5.0
)

acc_confidence_interval = m_tools.get_resampled_confidence_interval(
    results.cross_val_acc_scores, score_range=[0,1], alpha=5.0
)

a = 0
