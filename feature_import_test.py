from tools.model_tools import local_load_model
from DatasetCompiler import DatasetCompiler

data = DatasetCompiler.load_from_pickle('./data/processed/lrg_clean_data_v2.pickle')
model = local_load_model('./saved_models/XGBoostClassifier.joblib')
feature_importances = model.model.feature_importances_
a = 0