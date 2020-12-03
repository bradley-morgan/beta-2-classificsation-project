from tools.model_performance_estimation import CrossValidation, BootstrapValidation
from sklearn.tree import DecisionTreeClassifier
from tools.anonymousClass import Obj
from tools.DatasetCompiler import DatasetCompiler


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
results = BootstrapValidation(10, 2000, data, make_model, config).run()
# results = CrossValidation(10, 10, data, make_model, config).run()

a = 0
